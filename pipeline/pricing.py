"""Book agent, step 3: price identified books in the policyholder's country and currency.

Order of search
1. The policyholder's country.
2. Other countries using the same currency (eurozone neighbours for EUR), so no conversion.
3. Large markets: US, GB, CA, AU, IN. The first price found is converted to the home currency
   with the ECB reference rate and labelled converted, with the rate, its date and source.
No price anywhere: amount stays empty, the book is excluded from totals and sent to review.

Sources (eBay Browse API only; both empty when EBAY_CLIENT_ID / EBAY_CLIENT_SECRET are not set)
- Replacement cost: the lower median asking price of current new fixed-price listings of the
  printed book on the country's eBay site. No new listing anywhere: empty, with the reason.
- Used value: the same for used listings. Asking prices, not sold prices.
Replacement cost, by the policyholder's country:
1. Country with an eBay site: eBay new listings, in the order above.
2. Country without one (India, for example): the Google Play eBook price in that country (Google
   Books API), labelled as an eBook edition (edition: "eBook") in the line, packet and report.
3. No eBook there (or Google Books unavailable): eBay new listings from the other markets above,
   converted; the line says why the eBook was not used (ebook_lookup).
The used value is always eBay used listings.

Appraisal: antiquarian works (first published before 1900) and anything priced above the
appraisal threshold (set in US dollars, converted to the policyholder's currency with the ECB rate, so
150 USD is about 12,500 INR) are not priced automatically; they get status needs_appraisal.
"""

from __future__ import annotations

import base64
import json
import os
import re
import statistics
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from rapidfuzz import fuzz

from .catalog import tokens
from .common import USER_AGENT, get_cached
from .fx import Rates

GOOGLE_BOOKS_URL = "https://www.googleapis.com/books/v1/volumes"
EUROZONE = ["DE", "AT", "FR", "NL", "BE", "IE", "ES", "IT", "FI", "PT"]
LARGE_MARKETS = ["US", "GB", "CA", "AU", "IN"]
APPRAISAL_YEAR = 1900
THRESHOLD_CURRENCY = "USD"
REPO = Path(__file__).resolve().parent.parent


def fallback_countries(home: str, currency: str) -> list[str]:
    if not re.fullmatch(r"[A-Za-z]{2}", home or ""):
        raise ValueError(f"Country must be an ISO 3166 alpha-2 code such as DE, not {home!r}")
    order = [home.upper()]
    if currency.upper() == "EUR":
        order += EUROZONE
    order += LARGE_MARKETS
    return list(dict.fromkeys(order))


def _env(name: str) -> str | None:
    if os.getenv(name):
        return os.environ[name]
    env = REPO / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            match = re.match(rf"\s*{name}\s*=\s*(.+)", line)
            if match:
                return match.group(1).strip().strip("\"'")
    return None


def load_books_key() -> str | None:
    return _env("GOOGLE_BOOKS_API_KEY")


# Google Books quota (catalogue search in catalog.py). Once Google says the quota is spent,
# waiting and retrying inside a batch only stalls reading, so searches go to Open Library until
# the pause ends.
DAILY_QUOTA_PAUSE_S = 3600
RATE_LIMIT_PAUSE_S = 60
_google_books_paused_until = 0.0


# Google Books answers 20-50% of requests with 503 "Service temporarily unavailable" at random
# (2026-11-04, same query or not), and up to two thirds when several arrive at once. Every search
# goes through one pacer, and a 503 is retried a few times quickly.
GOOGLE_BOOKS_MIN_INTERVAL_S = 0.25
GOOGLE_BOOKS_ATTEMPTS = 5
GOOGLE_BOOKS_BACKOFF_S = 0.5
_pace_lock = threading.Lock()
_last_google_books_request = 0.0


def pace_google_books() -> None:
    global _last_google_books_request
    with _pace_lock:
        wait = _last_google_books_request + GOOGLE_BOOKS_MIN_INTERVAL_S - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_google_books_request = time.monotonic()


def google_books_paused() -> bool:
    return time.time() < _google_books_paused_until


def note_google_books_error(exc: urllib.error.HTTPError, body: str) -> bool:
    """Pause Google Books if this error is a rate or quota limit. True if it was one."""
    global _google_books_paused_until
    body = body.lower()
    if exc.code != 429 and not (exc.code == 403 and ("ratelimit" in body or "quota" in body)):
        return False
    pause = DAILY_QUOTA_PAUSE_S if "per day" in body else RATE_LIMIT_PAUSE_S
    _google_books_paused_until = max(_google_books_paused_until, time.time() + pause)
    return True


class EbayPrices:
    """New or used copies for sale on a country's eBay site (Browse API, application token).

    Listings count only if their title contains the book's title and the author's surname, and
    they are not a summary or a lot of several books. Fixed-price listings only: an auction's
    current bid is not a price. Shipping is left out."""
    source = "eBay Browse API (current listings)"
    TOKEN_URL = "https://api.ebay.com/identity/v1/oauth2/token"
    SEARCH_URL = "https://api.ebay.com/buy/browse/v1/item_summary/search"
    MARKETPLACES = {"DE": "EBAY_DE", "AT": "EBAY_AT", "FR": "EBAY_FR", "NL": "EBAY_NL", "IE": "EBAY_IE",
                    "ES": "EBAY_ES", "IT": "EBAY_IT", "GB": "EBAY_GB", "US": "EBAY_US", "CA": "EBAY_CA", "AU": "EBAY_AU"}
    BOOKS_CATEGORY = "267"
    CONDITIONS = {"used": "2750|3000|4000|5000|6000",   # like new, used, very good, good, acceptable
                  "new": "1000"}                          # new
    NOT_THE_BOOK = {"summary", "zusammenfassung", "workbook", "analysis", "lot", "bundle", "konvolut", "set", "paket"}

    def __init__(self, client_id: str, client_secret: str):
        self.client_id, self.client_secret = client_id, client_secret
        self._token, self._expires = None, 0.0
        self._lock = threading.Lock()

    def _access_token(self) -> str:
        with self._lock:
            if self._token is None or time.time() > self._expires - 60:
                basic = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode()
                data = urllib.parse.urlencode({"grant_type": "client_credentials",
                                               "scope": "https://api.ebay.com/oauth/api_scope"}).encode()
                request = urllib.request.Request(self.TOKEN_URL, data=data, headers={
                    "Authorization": f"Basic {basic}", "Content-Type": "application/x-www-form-urlencoded", "User-Agent": USER_AGENT})
                with urllib.request.urlopen(request, timeout=20) as response:
                    body = json.load(response)
                self._token, self._expires = body["access_token"], time.time() + body.get("expires_in", 7200)
            return self._token

    def _search(self, params: dict, marketplace: str) -> dict:
        from .telemetry import record_call

        request = urllib.request.Request(self.SEARCH_URL + "?" + urllib.parse.urlencode(params), headers={
            "Authorization": f"Bearer {self._access_token()}", "X-EBAY-C-MARKETPLACE-ID": marketplace, "User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                body = json.load(response)
        except OSError:
            record_call("ebay", ok=False)
            raise
        record_call("ebay")
        return body

    def offers(self, title: str, author: str, country: str, condition: str = "used") -> tuple[list[dict], str]:
        marketplace = self.MARKETPLACES.get(country)
        if marketplace is None:
            return [], ""
        surname = tokens(author.split(",")[0])[-1] if author and tokens(author.split(",")[0]) else ""
        query = f"{title} {surname}".strip()
        body = self._search({"q": query, "category_ids": self.BOOKS_CATEGORY, "limit": 50,
                             "filter": f"conditionIds:{{{self.CONDITIONS[condition]}}},buyingOptions:{{FIXED_PRICE}}"}, marketplace)
        retrieved_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        title_words = set(tokens(title))
        found = []
        for item in body.get("itemSummaries", []):
            listing, price = item.get("title", ""), item.get("price") or {}
            words = set(tokens(listing))
            if not price.get("value") or fuzz.token_set_ratio(title.lower(), listing.lower()) < 90:
                continue
            if surname and surname not in words:
                continue
            if (words & self.NOT_THE_BOOK) - title_words:
                continue
            found.append({"amount": float(price["value"]), "currency": price["currency"], "country": country,
                          "listing_title": listing, "condition": item.get("condition"), "url": item.get("itemWebUrl"),
                          "retrieved_at": retrieved_at})
        return found, query


_ebay: EbayPrices | None = None


def ebay_prices() -> EbayPrices | None:
    """The shared eBay client (one token for the process), or None without credentials."""
    global _ebay
    client_id, secret = _env("EBAY_CLIENT_ID"), _env("EBAY_CLIENT_SECRET")
    if not (client_id and secret):
        return None
    if _ebay is None or _ebay.client_id != client_id:
        _ebay = EbayPrices(client_id, secret)
    return _ebay


def ebay_value(book: dict, provider, home: str, currency: str, rates: Rates, condition: str) -> dict:
    """Lower median asking price of current eBay listings in one condition ("new": replacement
    cost, "used": used value), in the fallback order of countries."""
    empty = {"amount": None, "currency": currency, "source": None, "url": None, "retrieved_at": None, "condition_assumed": None}
    if provider is None:
        return {**empty, "reason": "no price source configured (EBAY_CLIENT_ID / EBAY_CLIENT_SECRET)"}
    tried = []
    for country in fallback_countries(home, currency):
        if country not in provider.MARKETPLACES:
            continue
        try:
            offers, query = provider.offers(book["title"], book.get("author", ""), country, condition)
        except Exception as exc:   # eBay unreachable: this value stays empty and the job retries later
            return {**empty, "reason": f"eBay lookup failed in {country}: {type(exc).__name__}", "countries_tried": tried + [country],
                    "lookup_failed": True}
        tried.append(country)
        if not offers:
            continue
        # The lower median is a price someone actually listed, and the link goes to that listing, so
        # the figure can be checked by clicking it (a plain median of an even count matched none).
        median = statistics.median_low(o["amount"] for o in offers)
        chosen = next(o for o in offers if o["amount"] == median)
        conditions = sorted({o["condition"] for o in offers if o.get("condition")})
        value = {"amount": round(median, 2), "currency": chosen["currency"], "source": f"eBay Browse API (current {condition} listings)",
                 "url": chosen["url"], "retrieved_at": chosen["retrieved_at"], "country": country, "converted": False,
                 "condition_assumed": f"{condition} ({', '.join(conditions) or 'condition not stated'}); median (lower middle) asking price of "
                                      f"{len(offers)} current listing(s) of the printed book, shipping excluded, not sold prices",
                 "listings_considered": len(offers), "query": query, "countries_tried": tried,
                 "listings": [{"amount": o["amount"], "currency": o["currency"], "condition": o.get("condition"), "url": o["url"]}
                              for o in sorted(offers, key=lambda o: o["amount"])][:30]}
        if chosen["currency"] != currency:
            try:
                amount, rate = rates.convert(median, chosen["currency"], currency)
            except KeyError as exc:
                return {**value, "amount": None, "currency": currency, "reason": str(exc)}
            value.update(amount=amount, currency=currency, converted=True,
                         original={"amount": round(median, 2), "currency": chosen["currency"]},
                         fx={"rate": round(rate, 6), "date": rates.date, "source": rates.source, "url": rates.url})
        return value
    return {**empty, "reason": f"no {condition} listing found on eBay in {', '.join(tried) or 'any supported country'}",
            "countries_tried": tried}


def local_threshold(threshold: float, currency: str, rates: Rates) -> float:
    """The appraisal threshold (in THRESHOLD_CURRENCY) in the policyholder's currency. Without a rate for
    that currency the number is used as is."""
    try:
        return rates.convert(threshold, THRESHOLD_CURRENCY, currency)[0]
    except KeyError:
        return threshold


class GoogleBooksEbooks:
    """Google Play eBook prices in one country, from the Google Books API. Only editions in the
    spine's language whose title and author match; summaries and workbooks are not the book."""
    source = "Google Books API (Google Play eBook)"

    def __init__(self, api_key: str, language: str = "en"):
        self.api_key, self.language = api_key, language

    def offers(self, title: str, author: str, country: str) -> list[dict] | None:
        """Matching eBook offers; None when Google Books could not be asked (quota or error), not "no eBook"."""
        if google_books_paused():
            return None
        main = title.split(":")[0].strip()
        surname = tokens(author.split(",")[0])[-1] if author and tokens(author.split(",")[0]) else ""
        params = {"q": f'intitle:"{main}"' + (f' inauthor:"{surname}"' if surname else ""), "country": country,
                  "printType": "books", "langRestrict": self.language, "maxResults": 20, "key": self.api_key}
        try:
            body, retrieved_at = get_cached(GOOGLE_BOOKS_URL, params, cache=False, attempts=GOOGLE_BOOKS_ATTEMPTS,
                                            backoff_s=GOOGLE_BOOKS_BACKOFF_S, before_request=pace_google_books)
        except OSError as exc:
            if isinstance(exc, urllib.error.HTTPError):
                note_google_books_error(exc, exc.read().decode("utf-8", "replace"))
            return None
        title_words = set(tokens(main))
        found = []
        for volume in body.get("items", []):
            info, sale = volume.get("volumeInfo", {}), volume.get("saleInfo", {})
            price = sale.get("retailPrice") or sale.get("listPrice")
            if sale.get("saleability") != "FOR_SALE" or not price or not price.get("amount"):
                continue
            if info.get("language") != self.language or fuzz.token_set_ratio(main.lower(), info.get("title", "").lower()) < 90:
                continue
            if surname and not any(surname in tokens(a) for a in info.get("authors", [])):
                continue
            if (set(tokens(info.get("title", ""))) & EbayPrices.NOT_THE_BOOK) - title_words:
                continue
            found.append({"amount": float(price["amount"]), "currency": price["currencyCode"], "edition_title": info.get("title"),
                          "url": sale.get("buyLink") or info.get("infoLink"), "retrieved_at": retrieved_at})
        return found


def ebooks_provider() -> GoogleBooksEbooks | None:
    key = load_books_key()
    return GoogleBooksEbooks(key) if key else None


def ebook_price(book: dict, provider, home: str, currency: str) -> dict:
    """The eBook price in the policyholder's own country (no other country, no conversion): a local reference."""
    empty = {"amount": None, "currency": currency, "edition": "eBook"}
    if provider is None:
        return {**empty, "reason": "GOOGLE_BOOKS_API_KEY not set"}
    found = provider.offers(book["title"], book.get("author", ""), home)
    if found is None:
        return {**empty, "reason": "eBook price not looked up: Google Books unavailable (daily quota or an error)"}
    offers = [o for o in found if o["currency"] == currency]
    if not offers:
        return {**empty, "reason": f"no eBook for sale on Google Play in {home}"}
    median = statistics.median_low(o["amount"] for o in offers)
    chosen = next(o for o in offers if o["amount"] == median)
    return {"amount": round(median, 2), "currency": currency, "edition": "eBook", "source": provider.source,
            "country": home, "converted": False, "url": chosen["url"], "retrieved_at": chosen["retrieved_at"],
            "edition_title": chosen["edition_title"], "offers_considered": len(offers),
            "condition_assumed": f"new, eBook edition (Google Play {home}): no eBay site in {home}; median (lower middle) "
                                 f"of {len(offers)} matching eBook offer(s)"}


def price_book(book: dict, home: str, currency: str, rates: Rates, threshold: float, ebay=None, ebooks=None) -> dict:
    """Replacement cost and used value for one identified book. Replacement cost: eBay new listings
    where the policyholder's country has an eBay site; elsewhere the country's Google Play eBook price, and
    eBay new listings from other markets if there is no eBook. Used value: eBay used listings. Books
    sent to appraisal get neither. threshold is in THRESHOLD_CURRENCY. If eBay could not be reached,
    the result is marked retry so the job tries again when idle."""
    threshold = local_threshold(threshold, currency, rates)
    unpriced = {"amount": None, "currency": currency, "reason": "not priced: sent to human appraisal"}
    year = book.get("first_publish_year")
    if year and year < APPRAISAL_YEAR:
        return {"status": "needs_appraisal", "appraisal_reason": f"first published {year} (antiquarian)",
                "replacement_cost": None, "used_value": unpriced, "countries_tried": []}
    ebook_lookup = None
    if home.upper() not in EbayPrices.MARKETPLACES:      # no eBay site in the policyholder's country: its eBook price first
        new = ebook_price(book, ebooks, home.upper(), currency)
        if new["amount"] is None:
            ebook_lookup = new["reason"]
            new = {**ebay_value(book, ebay, home, currency, rates, "new"), "instead_of": f"eBook: {ebook_lookup}"}
    else:
        new = ebay_value(book, ebay, home, currency, rates, "new")
    if new["amount"] is not None and new["amount"] > threshold:
        return {"status": "needs_appraisal", "appraisal_reason": f"price {new['amount']} {currency} above appraisal "
                f"threshold {threshold} {currency}", "replacement_cost": None, "price_seen": new,
                "used_value": unpriced, "countries_tried": new.get("countries_tried", [])}
    used = ebay_value(book, ebay, home, currency, rates, "used")
    result = {"status": "identified", "replacement_cost": new, "used_value": used, "countries_tried": new.get("countries_tried", [])}
    if ebook_lookup:
        result["ebook_lookup"] = ebook_lookup
    if new.get("lookup_failed") or used.get("lookup_failed"):
        result["retry"] = True
    return result


def book_totals(priced: list[dict]) -> dict:
    """Totals from the lines, in code. Each line is one work, merged across frames; `copies` (1
    unless the policyholder confirmed more) multiplies its count and its prices."""
    def copies(p: dict) -> int:
        return p.get("copies", 1)

    costed = [p for p in priced if (p.get("replacement_cost") or {}).get("amount") is not None]
    used = [p for p in priced if (p.get("used_value") or {}).get("amount") is not None]
    return {"books_identified": sum(copies(p) for p in priced), "titles_identified": len(priced),
            "books_priced": sum(copies(p) for p in costed),
            "books_needs_appraisal": sum(copies(p) for p in priced if p["status"] == "needs_appraisal"),
            "books_replacement_cost": round(sum(p["replacement_cost"]["amount"] * copies(p) for p in costed), 2),
            "books_used_value": round(sum(p["used_value"]["amount"] * copies(p) for p in used), 2) if used else None,
            "books_with_used_value": sum(copies(p) for p in used),
            "books_converted": sum(copies(p) for p in costed if p["replacement_cost"].get("converted")),
            "excluded_from_totals": sum(copies(p) for p in priced) - sum(copies(p) for p in costed),
            "note": "identified books only, one line per work across all frames, times confirmed copies; "
                    "unidentified spines not yet counted"}


def price_books(identified: list[dict], home: str, currency: str, rates: Rates, *, threshold: float = 150.0) -> dict:
    ebay, ebooks = ebay_prices(), ebooks_provider()
    priced = []
    for book in identified:
        priced.append({"id": book["id"], "title": book["title"], "author": book["author"],
                       "frame_ref": book["frame_ref"], "id_confidence": book["id_confidence"],
                       **price_book(book, home, currency, rates, threshold, ebay, ebooks)})
    review = []
    for p in priced:
        cost = p.get("replacement_cost") or {}
        if p["status"] == "needs_appraisal":
            review.append({"ref_id": p["id"], "reason": p["appraisal_reason"]})
        elif cost.get("amount") is None:
            review.append({"ref_id": p["id"], "reason": cost.get("reason", "no price")})
        elif cost.get("converted"):
            review.append({"ref_id": p["id"], "reason": f"price converted from {cost['original']['currency']} ({cost['country']})"})
    return {"country": home.upper(), "currency": currency.upper(),
            "appraisal_threshold": {"amount": local_threshold(threshold, currency, rates), "currency": currency.upper(),
                                    "set_as": f"{threshold:g} {THRESHOLD_CURRENCY}"},
            "totals": book_totals(priced), "price_source": ebay.source if ebay else None,
            "fx": {"source": rates.source, "date": rates.date, "url": rates.url},
            "books": priced, "review_queue": review}
