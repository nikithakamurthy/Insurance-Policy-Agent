"""Book agent, step 3: price identified books in the policyholder's country and currency.

Order of search
1. The policyholder's country.
2. Other countries using the same currency (eurozone neighbours for EUR), so no conversion.
3. Large markets: US, GB, CA, AU, IN. The first price found is converted to the home currency
   with the ECB reference rate and labelled converted, with the rate, its date and source.
No price anywhere: amount stays empty, the book is excluded from totals and sent to review.

Sources
- Replacement cost: eBay Browse API, the median asking price of current new fixed-price listings
  of the printed book on the country's eBay site. Without new listings: Google Books API sale
  information, country-specific but only for Google Play eBooks, so a new eBook edition (stated
  in condition_assumed).
- Used value: eBay Browse API, the median asking price of current used fixed-price listings
  of the same title and author on the country's eBay site. Asking prices, not sold prices.
  Empty when EBAY_CLIENT_ID / EBAY_CLIENT_SECRET are not set.

Appraisal: antiquarian works (first published before 1900) and anything priced above the
appraisal threshold are not priced automatically; they get status needs_appraisal.
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


# Google Books quota: one key, shared by search (catalog.py) and pricing. Once Google says the
# quota is spent, waiting and retrying inside a batch only stalls reading (minutes per book across
# ~15 fallback countries), so calls fail fast until the pause ends; unpriced books are retried
# when the job is idle.
DAILY_QUOTA_PAUSE_S = 3600
RATE_LIMIT_PAUSE_S = 60
_google_books_paused_until = 0.0


class GoogleBooksQuotaError(RuntimeError):
    pass


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


class GoogleBooksPrices:
    """Country-specific Google Play eBook prices, restricted to one edition language.

    A store in Germany also sells German translations; a translation is a different edition
    from the copy on the shelf, so only editions in the spine's language count.
    """
    source = "Google Books API (Google Play eBooks)"
    MIN_INTERVAL_S = 0.25          # pace requests; a burst across many countries trips the rate limit
    _lock = threading.Lock()
    _last_request = 0.0

    def __init__(self, api_key: str, language: str = "en"):
        self.api_key = api_key
        self.language = language

    def _get(self, params: dict) -> tuple[dict, str]:
        if google_books_paused():
            raise GoogleBooksQuotaError("Google Books quota exhausted; pricing paused")
        with GoogleBooksPrices._lock:
            wait = GoogleBooksPrices._last_request + self.MIN_INTERVAL_S - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            GoogleBooksPrices._last_request = time.monotonic()
        try:
            return get_cached(GOOGLE_BOOKS_URL, params, cache=False)   # prices are always fetched fresh
        except urllib.error.HTTPError as exc:
            # A rate or quota limit pauses Google Books; other errors (an invalid country) are permanent.
            if note_google_books_error(exc, exc.read().decode("utf-8", "replace")):
                raise GoogleBooksQuotaError(f"Google Books quota exhausted (HTTP {exc.code}); pricing paused") from exc
            raise

    def offers(self, title: str, author: str, country: str) -> tuple[list[dict], str]:
        surname = tokens(author.split(",")[0])[-1] if author and tokens(author.split(",")[0]) else ""
        query = f'intitle:"{title}"' + (f' inauthor:"{surname}"' if surname else "")
        body, retrieved_at = self._get({"q": query, "country": country, "printType": "books",
                                        "langRestrict": self.language, "maxResults": 20, "key": self.api_key})
        found = []
        for volume in body.get("items", []):
            info, sale = volume.get("volumeInfo", {}), volume.get("saleInfo", {})
            price = sale.get("retailPrice") or sale.get("listPrice")
            if sale.get("saleability") != "FOR_SALE" or not price or info.get("language") != self.language:
                continue
            if fuzz.token_set_ratio(title.lower(), info.get("title", "").lower()) < 90:
                continue
            if surname and not any(surname in tokens(a) for a in info.get("authors", [])):
                continue
            found.append({"amount": float(price["amount"]), "currency": price["currencyCode"], "country": country,
                          "edition_title": info.get("title"), "language": info.get("language"),
                          "isbns": [i["identifier"] for i in info.get("industryIdentifiers", []) if i["type"].startswith("ISBN")],
                          "url": sale.get("buyLink") or info.get("infoLink"), "retrieved_at": retrieved_at})
        return found, query


class EbayUsedPrices:
    """Used copies for sale on a country's eBay site (Browse API, application token).

    Listings count only if their title contains the book's title and the author's surname, and
    they are not a summary or a lot of several books. Fixed-price listings only: an auction's
    current bid is not a price. Shipping is left out."""
    source = "eBay Browse API (current used listings)"
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


_ebay: EbayUsedPrices | None = None


def ebay_used_prices() -> EbayUsedPrices | None:
    """The shared eBay client (one token for the process), or None without credentials."""
    global _ebay
    client_id, secret = _env("EBAY_CLIENT_ID"), _env("EBAY_CLIENT_SECRET")
    if not (client_id and secret):
        return None
    if _ebay is None or _ebay.client_id != client_id:
        _ebay = EbayUsedPrices(client_id, secret)
    return _ebay


def used_value(book: dict, provider, home: str, currency: str, rates: Rates) -> dict:
    """Median asking price of used copies, following the same country order as replacement cost."""
    return ebay_value(book, provider, home, currency, rates, "used")


def ebay_value(book: dict, provider, home: str, currency: str, rates: Rates, condition: str) -> dict:
    """Median asking price of current eBay listings in one condition (used, or new for replacement
    cost when Google Books has no price), in the same country order as replacement cost."""
    empty = {"amount": None, "currency": currency, "source": None, "url": None, "retrieved_at": None, "condition_assumed": None}
    if provider is None:
        return {**empty, "reason": "no used-market price source configured (EBAY_CLIENT_ID / EBAY_CLIENT_SECRET)"}
    tried = []
    for country in fallback_countries(home, currency):
        if country not in provider.MARKETPLACES:
            continue
        try:
            offers, query = provider.offers(book["title"], book.get("author", ""), country, condition)
        except Exception as exc:   # an eBay failure leaves the used value empty, never the whole book
            return {**empty, "reason": f"eBay lookup failed in {country}: {type(exc).__name__}", "countries_tried": tried + [country]}
        tried.append(country)
        if not offers:
            continue
        # The lower median is a price someone actually listed, and the link goes to that listing, so
        # the figure can be checked by clicking it (a plain median of an even count matched none).
        median = statistics.median_low(o["amount"] for o in offers)
        chosen = next(o for o in offers if o["amount"] == median)
        conditions = sorted({o["condition"] for o in offers if o.get("condition")})
        value = {"amount": round(median, 2), "currency": chosen["currency"],
                 "source": provider.source if condition == "used" else "eBay Browse API (current new listings)",
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


def price_book(book: dict, provider, home: str, currency: str, rates: Rates, threshold: float,
               used_provider=None) -> dict:
    """Replacement cost and used value for one identified book. Books sent to appraisal get neither.

    Replacement cost is the median asking price of new copies on eBay (the printed book, as on the
    shelf). Without new listings it falls back to Google Books (new eBook); a Google Books quota
    failure is marked retry_replacement so the job tries again later. The used value is the median
    of used listings on eBay and never depends on Google Books."""
    year = book.get("first_publish_year")
    if year and year < APPRAISAL_YEAR:
        return _replacement(book, provider, home, currency, rates, threshold)      # antiquarian: appraisal, unpriced
    new = ebay_value(book, used_provider, home, currency, rates, "new") if used_provider is not None else None
    if new and new["amount"] is not None:
        if new["amount"] > threshold:
            return {"status": "needs_appraisal", "appraisal_reason": f"price {new['amount']} {currency} above appraisal "
                    f"threshold {threshold} {currency}", "replacement_cost": None, "price_seen": new,
                    "used_value": {"amount": None, "currency": currency, "reason": "not priced: sent to human appraisal"},
                    "countries_tried": new.get("countries_tried", [])}
        result = {"status": "identified", "replacement_cost": new, "countries_tried": new.get("countries_tried", [])}
    else:
        try:
            result = _replacement(book, provider, home, currency, rates, threshold)
        except GoogleBooksQuotaError as exc:
            result = {"status": "identified", "replacement_cost": {"amount": None, "currency": currency, "reason": str(exc)},
                      "countries_tried": [], "retry_replacement": True}
        if result["status"] != "identified":
            return result
        if new is not None and (result.get("replacement_cost") or {}).get("amount") is not None:
            result["replacement_cost"]["instead_of"] = f"eBay new copies: {new.get('reason')}"
    result["used_value"] = used_value(book, used_provider, home, currency, rates)
    return result


def _replacement(book: dict, provider, home: str, currency: str, rates: Rates, threshold: float) -> dict:
    """Replacement cost, following the fallback order."""
    used = {"amount": None, "currency": currency, "source": None, "url": None, "retrieved_at": None,
            "condition_assumed": None, "reason": "not priced: sent to human appraisal"}
    year = book.get("first_publish_year")
    if year and year < APPRAISAL_YEAR:
        return {"status": "needs_appraisal", "appraisal_reason": f"first published {year} (antiquarian)",
                "replacement_cost": None, "used_value": used, "countries_tried": []}
    if provider is None:
        return {"status": "identified", "replacement_cost": {"amount": None, "currency": currency,
                "reason": "GOOGLE_BOOKS_API_KEY not set; no price source available"}, "used_value": used, "countries_tried": []}

    tried = []
    for country in fallback_countries(home, currency):
        offers, query = provider.offers(book["title"], book.get("author", ""), country)
        tried.append(country)
        if not offers:
            continue
        median = statistics.median(o["amount"] for o in offers)
        chosen = min(offers, key=lambda o: abs(o["amount"] - median))
        cost = {"amount": chosen["amount"], "currency": chosen["currency"], "source": provider.source,
                "url": chosen["url"], "retrieved_at": chosen["retrieved_at"], "country": country,
                "converted": False, "condition_assumed": "new, eBook edition (Google Play)",
                "language_assumed": getattr(provider, "language", None),
                "edition_matched": {"title": chosen["edition_title"], "language": chosen["language"], "isbns": chosen["isbns"]},
                "offers_considered": len(offers), "query": query}
        if chosen["currency"] != currency:
            try:
                amount, rate = rates.convert(chosen["amount"], chosen["currency"], currency)
            except KeyError as exc:
                cost.update(amount=None, reason=str(exc))
                return {"status": "identified", "replacement_cost": cost, "used_value": used, "countries_tried": tried}
            cost.update(amount=amount, currency=currency, converted=True,
                        original={"amount": chosen["amount"], "currency": chosen["currency"]},
                        fx={"rate": round(rate, 6), "date": rates.date, "source": rates.source, "url": rates.url})
        if cost["amount"] > threshold:
            return {"status": "needs_appraisal", "appraisal_reason": f"price {cost['amount']} {currency} above "
                    f"appraisal threshold {threshold} {currency}", "replacement_cost": None, "price_seen": cost,
                    "used_value": used, "countries_tried": tried}
        return {"status": "identified", "replacement_cost": cost, "used_value": used, "countries_tried": tried}
    return {"status": "identified", "replacement_cost": {"amount": None, "currency": currency,
            "reason": f"no price found in {', '.join(tried)}"}, "used_value": used, "countries_tried": tried}


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


def price_books(identified: list[dict], home: str, currency: str, rates: Rates, *, threshold: float = 150.0,
                provider=None, language: str = "en") -> dict:
    if provider is None:
        key = load_books_key()
        provider = GoogleBooksPrices(key, language) if key else None
    used_provider = ebay_used_prices()
    priced = []
    for book in identified:
        priced.append({"id": book["id"], "title": book["title"], "author": book["author"],
                       "frame_ref": book["frame_ref"], "id_confidence": book["id_confidence"],
                       **price_book(book, provider, home, currency, rates, threshold, used_provider)})
    review = []
    for p in priced:
        cost = p.get("replacement_cost") or {}
        if p["status"] == "needs_appraisal":
            review.append({"ref_id": p["id"], "reason": p["appraisal_reason"]})
        elif cost.get("amount") is None:
            review.append({"ref_id": p["id"], "reason": cost.get("reason", "no price")})
        elif cost.get("converted"):
            review.append({"ref_id": p["id"], "reason": f"price converted from {cost['original']['currency']} ({cost['country']})"})
    return {"country": home.upper(), "currency": currency.upper(), "appraisal_threshold": threshold,
            "totals": book_totals(priced),
            "price_source": provider.source if provider else None, "edition_language": language,
            "used_price_source": used_provider.source if used_provider else None,
            "fx": {"source": rates.source, "date": rates.date, "url": rates.url},
            "books": priced, "review_queue": review}
