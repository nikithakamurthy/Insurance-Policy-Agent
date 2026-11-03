import json
import time
import unittest

from pipeline.catalog import query_text, score_match
from pipeline.fx import parse_ecb
from pipeline.pricing import EbayUsedPrices, fallback_countries, price_book
from pipeline.spine_ocr import group_strips, to_original

ECB_SAMPLE = """<Cube><Cube time='2026-10-30'>
<Cube currency='USD' rate='1.1172'/><Cube currency='GBP' rate='0.83380'/><Cube currency='INR' rate='106.0840'/>
</Cube></Cube>"""
RATES = parse_ecb(ECB_SAMPLE, "2026-11-01T10:00:00+0100")
LEAN = {"title": "The Lean Startup", "author_name": ["Eric Ries"]}


class SpineGeometryTest(unittest.TestCase):
    def test_rotated_boxes_map_back(self):
        # A box at the top-left of the original frame, read after each rotation.
        for got, want in ((to_original((0.0, 0.8, 0.1, 1.0), "top_to_bottom"), [0.0, 0.0, 0.2, 0.1]),
                          (to_original((0.8, 0.0, 1.0, 0.1), "bottom_to_top"), [0.0, 0.0, 0.1, 0.2])):
            for g, w in zip(got, want):
                self.assertAlmostEqual(g, w)

    def test_upside_down_box_maps_back(self):
        # A box at the top-left of the frame is at the bottom-right of the 180-degree view.
        for g, w in zip(to_original((0.8, 0.9, 1.0, 1.0), "upside_down"), [0.0, 0.0, 0.2, 0.1]):
            self.assertAlmostEqual(g, w)

    def test_lines_on_one_strip_group_together(self):
        lines = [{"text": "ERIC RIES", "box": (0.6, 0.30, 0.9, 0.34)},
                 {"text": "THE LEAN STARTUP", "box": (0.1, 0.30, 0.5, 0.35)},
                 {"text": "ZERO to ONE", "box": (0.1, 0.50, 0.4, 0.54)}]
        strips = group_strips(lines)
        self.assertEqual([[l["text"] for l in s] for s in strips], [["THE LEAN STARTUP", "ERIC RIES"], ["ZERO to ONE"]])

    def test_stacked_lines_of_neighbouring_spines_stay_apart(self):
        # Slanted neighbouring spines: vertical extents overlap, but the lines lie side by side.
        lines = [{"text": "Designing Machine Learning Systems", "box": (0.1, 0.40, 0.7, 0.45)},
                 {"text": "Building a Second Brain", "box": (0.15, 0.43, 0.6, 0.48)}]
        self.assertEqual(len(group_strips(lines)), 2)


class MatchScoringTest(unittest.TestCase):
    def test_title_and_author_read(self):
        self.assertTrue(score_match("THE LEAN STARTUP ERIC RIES", LEAN)["accepted"])

    def test_ocr_slip_still_matches(self):
        doc = {"title": "Building a Second Brain", "author_name": ["Tiago Forte"]}
        self.assertTrue(score_match("Building Lusecond Brain TIAGO FORTE", doc)["accepted"])

    def test_partial_title_is_rejected(self):
        doc = {"title": "The Lean Startup Field Guide", "author_name": ["Somebody Else"]}
        self.assertFalse(score_match("THE LEAN STARTUP", doc)["accepted"])

    def test_mirrored_gibberish_is_rejected(self):
        self.assertFalse(score_match("THL PELAV 3103 VITnr", LEAN)["accepted"])

    def test_words_run_together_still_match(self):
        doc = {"title": "A Dead Hand", "author_name": ["Paul Theroux"]}
        self.assertTrue(score_match("A DEADHAND PAUL THEROUX", doc)["accepted"])

    def test_title_and_author_with_neighbour_text_is_accepted(self):
        doc = {"title": "Designing Machine Learning Systems", "author_name": ["Chip Huyen"]}
        self.assertTrue(score_match("Designing Machine Learning Systems Huyen OREILLY FORTE Building", doc)["accepted"])

    def test_query_keeps_real_title_words(self):
        # "die" used to be dropped as a German article, leaving "not alone" and the wrong book.
        self.assertEqual(query_text("HOW TO NOT DIE ALONE"), "how not die alone")

    def test_first_name_counts_only_with_the_whole_title(self):
        zero = {"title": "Zero to One", "author_name": ["Peter Thiel"]}
        self.assertTrue(score_match("ZERO to ONE Peter", zero)["accepted"])
        affair = {"title": "After the Affair", "author_name": ["Julia Cole"]}
        self.assertTrue(score_match("AFTER THE AFFAIR JULIAC", affair)["accepted"])   # surname cut off by OCR
        partial = {"title": "Zero to One Hundred", "author_name": ["Peter Smith"]}
        self.assertFalse(score_match("ZERO to ONE Peter", partial)["accepted"])        # title not fully read

    def test_query_drops_apostrophe_letters_and_publishers(self):
        self.assertEqual(query_text("The Staff Engineer's Path O'REILLY"), "staff engineers path")

    def test_text_from_two_books_is_rejected(self):
        # Most of the reading is unexplained by the title: likely two spines merged.
        self.assertFalse(score_match("LEAN STARTUP ZERO ONE THIEL BLISS CLUB CREATIVE LOVERS", LEAN)["accepted"])


class SearchTest(unittest.TestCase):
    def setUp(self):
        from unittest import mock
        self.calls, self.google_down = [], False

        def fake_get(url, params, **kw):
            if "googleapis" in url and self.google_down:
                raise OSError("503 Service Unavailable")
            self.calls.append(params["q"])
            return ({"items": []} if "googleapis" in url else {"docs": []}), "now"
        for patcher in (mock.patch("pipeline.catalog.get_cached", side_effect=fake_get),
                        mock.patch("pipeline.pricing.load_books_key", lambda: "test-key"),
                        mock.patch("pipeline.catalog.correct_spelling", lambda q: q)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_fallback_is_capped(self):
        from pipeline.catalog import MAX_FALLBACK_QUERIES, search
        search("designing machine learning systems chip huyen", limit=10)
        self.assertEqual(len(self.calls), 1 + MAX_FALLBACK_QUERIES + 1)                # plus one Open Library query
        self.assertEqual(self.calls[1], "systems chip huyen")                       # last spine of a merged reading
        self.assertEqual(self.calls[2], "designing machine learning systems chip")   # then the last word dropped

    def test_memo_skips_repeat_queries_within_a_sweep_only(self):
        from pipeline.catalog import search
        sweep_a, sweep_b = {}, {}
        search("zero to one peter thiel", limit=10, memo=sweep_a)
        first = len(self.calls)
        search("zero to one peter thiel", limit=10, memo=sweep_a)
        self.assertEqual(len(self.calls), first)        # same sweep: answered from memory
        search("zero to one peter thiel", limit=10, memo=sweep_b)
        self.assertEqual(len(self.calls), 2 * first)    # another sweep: searched again

    def test_google_books_failure_falls_back_to_open_library(self):
        from unittest import mock
        from pipeline import catalog
        self.google_down = True
        with mock.patch.object(catalog, "_open_library", return_value=([{"key": "/works/OL1W", "title": "Zero to One", "author_name": ["Peter Thiel"]}], "now")) as ol:
            docs, _, _ = catalog.search("zero to one peter thiel", limit=10)
        self.assertEqual(docs[0]["title"], "Zero to One")
        ol.assert_called_once()


class SameWorkTest(unittest.TestCase):
    def test_entries_seen_as_duplicates_in_a_live_sweep_merge(self):
        from pipeline.catalog import same_work
        self.assertTrue(same_work("Spark: The Definitive Guide", "Matei Zaharia and Bill Chambers",
                                  "Spark The Definitive Guide", "Bill Chambers, Matei Zaharia"))
        self.assertTrue(same_work("The Intelligent Investor", "Benjamin Graham", "The Intelligent Investor, Rev. Ed", "Benjamin Graham"))
        self.assertTrue(same_work("How to Win Friends and Influence People", "Dale Carnegie",
                                  "How to Win Friends and Influence People by Dale Carnegie", "Dale Carnegie"))

    def test_sequels_and_other_authors_stay_apart(self):
        from pipeline.catalog import same_work
        self.assertFalse(same_work("Dune", "Frank Herbert", "Dune Messiah", "Frank Herbert"))
        self.assertFalse(same_work("Alone", "Richard D. Logan", "How to Not Die Alone", "Logan Ury"))
        self.assertFalse(same_work("Zero to One", "Peter Thiel", "Zero to One Hundred", "Peter Smith"))


class SpellingTest(unittest.TestCase):
    def test_ocr_slips_and_run_together_words_are_corrected(self):
        from pipeline.catalog import correct_spelling
        self.assertEqual(correct_spelling("workng backwrds"), "working backwards")
        self.assertEqual(correct_spelling("afterthe affair"), "after the affair")

    def test_corrected_query_that_finds_nothing_falls_back_to_the_reading(self):
        from unittest import mock
        from pipeline import catalog
        asked = []

        def fake_search(query, limit, memo=None, open_library=False):
            asked.append(query)
            return ([{"key": "gb:x", "title": "Designing Machine Learning Systems", "author_name": ["Chip Huyen"],
                      "source": "Google Books"}] if "huyen" in query else []), "now"
        with mock.patch.object(catalog, "correct_spelling", lambda q: q.replace("huyen", "human")), \
                mock.patch.object(catalog, "_search", side_effect=fake_search):
            docs, used, _ = catalog.search("Designing Machine Learning Systems Huyen", limit=10)
        self.assertEqual(asked[:2], ["designing machine learning systems human", "designing machine learning systems huyen"])
        self.assertEqual(used, "designing machine learning systems huyen")


class GoogleBooksSearchTest(unittest.TestCase):
    def test_editions_group_into_one_work(self):
        from unittest import mock
        from pipeline import catalog

        def volume(title, year, isbn):
            return {"volumeInfo": {"title": title, "authors": ["Tanya Reilly"], "publishedDate": year, "language": "en",
                                   "industryIdentifiers": [{"type": "ISBN_13", "identifier": isbn}]}}
        body = {"items": [volume("The Staff Engineer's Path", "2022-09-20", "1"), volume("The Staff Engineer's Path", "2023", "2"),
                          volume("Navigating the Staff Engineer's Path", "2024", "3")]}
        with mock.patch("pipeline.catalog.get_cached", return_value=(body, "now")):
            docs, _ = catalog._google_books("staf enginers path", 10, "k")
        self.assertEqual([d["title"] for d in docs], ["The Staff Engineer's Path", "Navigating the Staff Engineer's Path"])
        self.assertEqual((docs[0]["edition_count"], docs[0]["published_year_seen"], docs[0]["first_publish_year"]), (2, 2022, None))
        self.assertEqual(docs[0]["source"], "Google Books")


class FakeProvider:
    source = "fake"

    def __init__(self, prices):
        self.prices = prices   # country -> (amount, currency)

    def offers(self, title, author, country):
        if country not in self.prices:
            return [], "q"
        amount, ccy = self.prices[country]
        return [{"amount": amount, "currency": ccy, "country": country, "edition_title": title, "language": "en",
                 "isbns": [], "url": f"https://example.test/{country}", "retrieved_at": "2026-11-01"}], "q"


class PricingTest(unittest.TestCase):
    book = {"title": "The Lean Startup", "author": "Eric Ries", "first_publish_year": 2011}

    def test_country_name_instead_of_code_is_rejected(self):
        with self.assertRaises(ValueError):
            fallback_countries("Germany", "EUR")

    def test_fallback_order_for_euro(self):
        order = fallback_countries("DE", "EUR")
        self.assertEqual(order[0], "DE")
        self.assertLess(order.index("AT"), order.index("US"))

    def test_home_price_is_not_converted(self):
        result = price_book(self.book, FakeProvider({"DE": (12.99, "EUR")}), "DE", "EUR", RATES, 150)
        self.assertEqual(result["replacement_cost"]["amount"], 12.99)
        self.assertFalse(result["replacement_cost"]["converted"])

    def test_same_currency_neighbour_is_used_before_conversion(self):
        result = price_book(self.book, FakeProvider({"AT": (11.0, "EUR"), "US": (9.0, "USD")}), "DE", "EUR", RATES, 150)
        self.assertEqual(result["replacement_cost"]["country"], "AT")
        self.assertFalse(result["replacement_cost"]["converted"])

    def test_foreign_price_is_converted_with_ecb_rate(self):
        result = price_book(self.book, FakeProvider({"US": (11.172, "USD")}), "DE", "EUR", RATES, 150)
        cost = result["replacement_cost"]
        self.assertTrue(cost["converted"])
        self.assertAlmostEqual(cost["amount"], 10.0, places=2)
        self.assertEqual(cost["original"], {"amount": 11.172, "currency": "USD"})
        self.assertEqual(cost["fx"]["date"], "2026-10-30")

    def test_no_price_anywhere_leaves_amount_empty(self):
        result = price_book(self.book, FakeProvider({}), "DE", "EUR", RATES, 150)
        self.assertIsNone(result["replacement_cost"]["amount"])
        self.assertIn("no price found", result["replacement_cost"]["reason"])

    def test_expensive_book_goes_to_appraisal_unpriced(self):
        result = price_book(self.book, FakeProvider({"DE": (420.0, "EUR")}), "DE", "EUR", RATES, 150)
        self.assertEqual(result["status"], "needs_appraisal")
        self.assertIsNone(result["replacement_cost"])

    def test_totals_add_up_and_exclude_unpriced(self):
        from pipeline.pricing import book_totals
        lines = [{"status": "identified", "replacement_cost": {"amount": 10.0}, "used_value": {"amount": None}},
                 {"status": "identified", "replacement_cost": {"amount": 2.5, "converted": True}, "used_value": {"amount": None}},
                 {"status": "identified", "replacement_cost": {"amount": None}, "used_value": {"amount": None}},
                 {"status": "needs_appraisal", "replacement_cost": None, "used_value": {"amount": None}}]
        totals = book_totals(lines)
        self.assertEqual(totals["books_replacement_cost"], 12.5)
        self.assertEqual(totals["excluded_from_totals"], 2)
        self.assertEqual(totals["books_converted"], 1)

    def test_antiquarian_goes_to_appraisal(self):
        old = {**self.book, "first_publish_year": 1851}
        self.assertEqual(price_book(old, FakeProvider({"DE": (9.0, "EUR")}), "DE", "EUR", RATES, 150)["status"], "needs_appraisal")




def listing(title, value, currency="EUR", condition="Gut"):
    return {"title": title, "price": {"value": str(value), "currency": currency}, "condition": condition,
            "itemWebUrl": f"https://www.ebay.de/itm/{abs(hash(title)) % 10**6}"}


class FakeEbay(EbayUsedPrices):
    """The real filtering and median, with the HTTP call replaced: country -> listings."""

    def __init__(self, listings):
        super().__init__("id", "secret")
        self.listings, self.asked = listings, []

    def _search(self, params, marketplace):
        self.asked.append(marketplace)
        key = ("new", marketplace) if "conditionIds:{1000}" in params["filter"] else marketplace
        return {"itemSummaries": self.listings.get(key, [])}


class QuotaProvider(FakeProvider):
    def offers(self, title, author, country):
        from pipeline.pricing import GoogleBooksQuotaError
        raise GoogleBooksQuotaError("Google Books quota exhausted; pricing paused")


class TraceableEbayPriceTest(unittest.TestCase):
    def test_reported_price_is_a_real_listing_and_the_link_goes_to_it(self):
        # Even count: the plain median (9.62) matched no listing; the lower middle (7.99) is one.
        prices = [3.12, 6.0, 7.99, 11.24, 15.0, 20.0]
        ebay = FakeEbay({"EBAY_US": [listing(f"The Lean Startup Eric Ries copy {i}", p, "USD") for i, p in enumerate(prices)]})
        used = price_book({"title": "The Lean Startup", "author": "Eric Ries", "first_publish_year": 2011},
                          FakeProvider({}), "US", "USD", RATES, 150, ebay)["used_value"]
        self.assertEqual(used["amount"], 7.99)
        linked = next(l for l in used["listings"] if l["url"] == used["url"])
        self.assertEqual(linked["amount"], 7.99)
        self.assertEqual([l["amount"] for l in used["listings"]], prices)


class ReplacementFallbackTest(unittest.TestCase):
    book = {"title": "The Manager's Path", "author": "Camille Fournier", "first_publish_year": 2017}
    used = [listing("The Manager's Path Camille Fournier", 18.0)]
    new = [listing("The Manager's Path - Camille Fournier - neu", 30.0, condition="Neu"),
           listing("Camille Fournier The Manager's Path O'Reilly", 32.0, condition="Neu")]

    def test_google_quota_falls_back_to_new_copies_on_ebay_and_still_prices_used(self):
        ebay = FakeEbay({"EBAY_DE": self.used, ("new", "EBAY_DE"): self.new})
        result = price_book(self.book, QuotaProvider({}), "DE", "EUR", RATES, 150, ebay)
        self.assertEqual((result["replacement_cost"]["amount"], result["replacement_cost"]["source"]),
                         (30.0, "eBay Browse API (current new listings)"))
        self.assertNotIn("retry_replacement", result)
        self.assertEqual(result["used_value"]["amount"], 18.0)

    def test_quota_without_new_listings_is_retried_later_and_used_still_priced(self):
        result = price_book(self.book, QuotaProvider({}), "DE", "EUR", RATES, 150, FakeEbay({"EBAY_DE": self.used}))
        self.assertIsNone(result["replacement_cost"]["amount"])
        self.assertTrue(result["retry_replacement"])
        self.assertEqual(result["used_value"]["amount"], 18.0)

    def test_new_copies_on_ebay_come_before_the_ebook_price(self):
        result = price_book(self.book, FakeProvider({"DE": (12.99, "EUR")}), "DE", "EUR", RATES, 150,
                            FakeEbay({("new", "EBAY_DE"): self.new}))
        self.assertEqual(result["replacement_cost"]["amount"], 30.0)

    def test_without_new_listings_the_ebook_price_is_used_and_says_why(self):
        result = price_book(self.book, FakeProvider({"DE": (12.99, "EUR")}), "DE", "EUR", RATES, 150, FakeEbay({}))
        self.assertEqual(result["replacement_cost"]["amount"], 12.99)
        self.assertIn("no new listing", result["replacement_cost"]["instead_of"])


class UsedValueTest(unittest.TestCase):
    book = {"title": "Zero to One", "author": "Peter Thiel", "first_publish_year": 2014}

    def price(self, ebay, home="DE", currency="EUR"):
        return price_book(self.book, FakeProvider({home: (12.99, currency)}), home, currency, RATES, 150, ebay)

    def test_median_of_matching_used_listings(self):
        ebay = FakeEbay({"EBAY_DE": [listing("Zero to One - Peter Thiel - Taschenbuch", 6.0),
                                     listing("Peter Thiel Zero to One Notes on Startups", 8.0, condition="Sehr gut"),
                                     listing("Zero to One Peter Thiel wie neu", 9.5),
                                     listing("Zero to One Summary - Peter Thiel", 2.0),        # a summary, not the book
                                     listing("Zero to One Blake Masters", 5.0),                # surname missing
                                     listing("Konvolut 5 Bücher Peter Thiel Zero to One", 20)]})  # a lot of several books
        used = self.price(ebay)["used_value"]
        self.assertEqual((used["amount"], used["currency"], used["listings_considered"]), (8.0, "EUR", 3))
        self.assertFalse(used["converted"])
        self.assertIn("not sold prices", used["condition_assumed"])
        self.assertTrue(used["url"].startswith("https://www.ebay.de/"))

    def test_no_local_listing_is_converted_from_another_market(self):
        ebay = FakeEbay({"EBAY_US": [listing("Zero to One by Peter Thiel", 11.172, "USD")]})
        used = self.price(ebay)["used_value"]
        self.assertTrue(used["converted"])
        self.assertAlmostEqual(used["amount"], 10.0, places=2)
        self.assertEqual(ebay.asked[0], "EBAY_DE")                       # home market first

    def test_no_listing_anywhere_leaves_used_value_empty(self):
        used = self.price(FakeEbay({}))["used_value"]
        self.assertIsNone(used["amount"])
        self.assertIn("no used listing", used["reason"])

    def test_ebay_failure_leaves_only_used_value_empty(self):
        class Down(FakeEbay):
            def _search(self, params, marketplace):
                raise OSError("eBay unreachable")
        result = self.price(Down({}))
        self.assertEqual(result["replacement_cost"]["amount"], 12.99)
        self.assertIsNone(result["used_value"]["amount"])

    def test_appraisal_books_get_no_used_value(self):
        ebay = FakeEbay({"EBAY_DE": [listing("Zero to One Peter Thiel", 900.0)]})
        result = price_book(self.book, FakeProvider({"DE": (420.0, "EUR")}), "DE", "EUR", RATES, 150, ebay)
        self.assertEqual(result["status"], "needs_appraisal")
        self.assertIsNone(result["used_value"]["amount"])


class GoogleBooksQuotaTest(unittest.TestCase):
    def setUp(self):
        from pipeline import pricing
        self.pricing = pricing
        pricing._google_books_paused_until = 0.0
        self.addCleanup(setattr, pricing, "_google_books_paused_until", 0.0)

    def http_error(self, code, message):
        import io
        import urllib.error
        return urllib.error.HTTPError("https://www.googleapis.com/books/v1/volumes", code, "error", {},
                                      io.BytesIO(json.dumps({"error": {"message": message}}).encode()))

    def test_daily_quota_fails_fast_and_pauses_for_an_hour(self):
        from unittest import mock
        calls = []

        def fake_get(*args, **kwargs):
            calls.append(1)
            raise self.http_error(429, "Quota exceeded for quota metric 'Queries' and limit 'Queries per day'")
        provider = self.pricing.GoogleBooksPrices("k")
        with mock.patch("pipeline.pricing.get_cached", side_effect=fake_get), mock.patch("time.sleep") as slept:
            with self.assertRaises(self.pricing.GoogleBooksQuotaError):
                provider.offers("Zero to One", "Peter Thiel", "DE")
            with self.assertRaises(self.pricing.GoogleBooksQuotaError):
                provider.offers("Zero to One", "Peter Thiel", "AT")     # paused: no request at all
        self.assertEqual(len(calls), 1)
        self.assertFalse(any(c.args and c.args[0] >= 1 for c in slept.call_args_list))   # no retry waits
        self.assertGreater(self.pricing._google_books_paused_until - time.time(), 3000)

    def test_other_errors_do_not_pause(self):
        self.assertFalse(self.pricing.note_google_books_error(self.http_error(400, "Invalid country"), "Invalid country"))
        self.assertFalse(self.pricing.google_books_paused())

    def test_search_skips_google_books_while_paused(self):
        from unittest import mock
        from pipeline import catalog
        self.pricing._google_books_paused_until = time.time() + 60
        with mock.patch.object(catalog, "_google_books") as google, \
                mock.patch.object(catalog, "_open_library", return_value=([], "now")) as open_library:
            catalog._search("zero one peter thiel", 10)
        google.assert_not_called()
        open_library.assert_called_once()


if __name__ == "__main__":
    unittest.main()
