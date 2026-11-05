import json
import time
import unittest

from pipeline.catalog import query_text, score_match
from pipeline.fx import parse_ecb
from pipeline.pricing import EbayPrices, fallback_countries, local_threshold, price_book
from pipeline.spine_ocr import group_strips, keep_best_readings, point_to_original, to_original

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


def _line(text, poly, turn, conf=0.99):
    return {"text": text, "box": [0, 0, 1, 1], "confidence": conf, "poly": poly, "turn": turn,
            "word_confidences": [(w, conf) for w in text.split()]}


SPINE = [[0.40, 0.10], [0.46, 0.10], [0.46, 0.80], [0.40, 0.80]]
AUTHOR_END = [[0.40, 0.55], [0.46, 0.55], [0.46, 0.80], [0.40, 0.80]]


class KeepBestReadingsTest(unittest.TestCase):
    def test_points_map_back_like_boxes(self):
        for orientation in ("top_to_bottom", "bottom_to_top", "upside_down"):
            x0, y0, x1, y1 = to_original((0.1, 0.2, 0.3, 0.4), orientation)
            corners = [point_to_original(x, y, orientation) for x, y in ((0.1, 0.2), (0.3, 0.4))]
            self.assertAlmostEqual(min(c[0] for c in corners), x0)
            self.assertAlmostEqual(max(c[1] for c in corners), y1)

    def test_reading_seen_upright_wins_over_turned_one(self):
        kept = keep_best_readings({"top_to_bottom": [_line("Building a Second Brain", SPINE, 0)],
                                   "bottom_to_top": [_line("Brain Second a Building", SPINE, 180)]})
        self.assertEqual([l["text"] for ls in kept.values() for l in ls], ["Building a Second Brain"])

    def test_other_reading_kept_when_it_adds_two_words(self):
        kept = keep_best_readings({"top_to_bottom": [_line("Building a Second", SPINE, 0)],
                                   "horizontal": [_line("Brain TIAGO FORTE", SPINE, 90)]})
        self.assertEqual(sum(len(ls) for ls in kept.values()), 2)

    def test_one_new_word_or_unsure_words_are_not_enough(self):
        kept = keep_best_readings({"top_to_bottom": [_line("Invent Wander", SPINE, 0)],
                                   "horizontal": [_line("Wander Bezos", SPINE, 90)],
                                   "upside_down": [_line("Jeff Bezos Writings", SPINE, 180, conf=0.6)]})
        self.assertEqual([l["text"] for ls in kept.values() for l in ls], ["Invent Wander"])

    def test_texts_in_different_places_are_both_kept(self):
        apart = [[0.70, 0.10], [0.76, 0.10], [0.76, 0.80], [0.70, 0.80]]
        kept = keep_best_readings({"top_to_bottom": [_line("Zero to One", SPINE, 0), _line("Zero to One", apart, 0)]})
        self.assertEqual(len(kept["top_to_bottom"]), 2)

    def test_slanted_lines_group_along_their_slant(self):
        def slanted(text, start, length, offset):
            # A line leaning 60 degrees, `offset` across from the first spine, starting `start` along it.
            u, n = (0.5, -0.866), (0.866, 0.5)
            x, y = 0.3 + n[0] * offset + u[0] * start, 0.9 + n[1] * offset + u[1] * start
            ex, ey, nx, ny = u[0] * length, u[1] * length, n[0] * 0.03, n[1] * 0.03
            poly = [[x, y], [x + ex, y + ey], [x + ex + nx, y + ey + ny], [x + nx, y + ny]]
            xs, ys = [p[0] for p in poly], [p[1] for p in poly]
            return {"text": text, "poly": poly, "box": (min(xs), min(ys), max(xs), max(ys))}

        lines = [slanted("Building a Second", 0.0, 0.3, 0.0), slanted("TIAGO FORTE", 0.35, 0.2, 0.0),
                 slanted("Designing Machine", 0.1, 0.3, 0.06)]
        self.assertEqual(sorted(sorted(l["text"] for l in s) for s in group_strips(lines)),
                         [["Building a Second", "TIAGO FORTE"], ["Designing Machine"]])

    def test_lines_without_outline_pass_through(self):
        old = {"horizontal": [{"text": "THE LEAN STARTUP", "box": [0, 0, 1, 1], "confidence": 0.9}]}
        self.assertEqual(keep_best_readings(old), old)


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

    def test_query_drops_apostrophe_letters(self):
        self.assertEqual(query_text("The Staff Engineer's Path O'REILLY"), "staff engineers path oreilly")

    def test_publisher_on_the_spine_is_explained_by_the_result(self):
        doc = {"title": "Designing Machine Learning Systems", "author_name": ["Chip Huyen"], "publisher": ["O'Reilly Media"]}
        reading = "Designing Machine Learning Systems Huyen O'REILLY"
        self.assertEqual(score_match(reading, doc)["explained_fraction"], 1.0)
        self.assertLess(score_match(reading, {**doc, "publisher": []})["explained_fraction"], 1.0)

    def test_repeated_title_word_is_not_a_distinctive_title(self):
        self.assertFalse(score_match("OUR SUCCESS", {"title": "Your Success is Our Success", "author_name": []})["accepted"])

    def test_joined_words_must_be_read_in_full(self):
        # Live sweep: "Second Brain" matched "30-Second Brain" (Anil Seth) through the pair "30second".
        self.assertFalse(score_match("Second Brain", {"title": "30-Second Brain", "author_name": ["Anil Seth"]})["accepted"])
        self.assertEqual(score_match("LEON STARTUP ERIC RIES", LEAN)["title_coverage"], 1.0)      # "leonstartup" ~ "leanstartup"
        self.assertTrue(score_match("PAUL THEROUX ADEADHAND", {"title": "A Dead Hand", "author_name": ["Paul Theroux"]})["accepted"])

    def test_one_word_cannot_be_both_title_and_author(self):
        # Live sweep: a garbled spine "Lessings istisemme" was accepted as "Lessing" by G. E. Lessing.
        lessing = {"title": "Lessing", "author_name": ["Gotthold Ephraim Lessing"]}
        self.assertFalse(score_match("Lessings istisemme", lessing)["author_hit"])
        self.assertFalse(score_match("Lessings istisemme", lessing)["accepted"])
        nathan = {"title": "Nathan der Weise", "author_name": ["Gotthold Ephraim Lessing"]}
        self.assertTrue(score_match("LESSING NATHAN DER WEISE", nathan)["accepted"])

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
                        mock.patch("pipeline.pricing.load_books_key", lambda: "test-key")):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_fallback_is_capped(self):
        from pipeline.catalog import MAX_FALLBACK_QUERIES, search
        search("designing machine learning systems chip huyen", limit=10)
        self.assertEqual(len(self.calls), 1 + MAX_FALLBACK_QUERIES + 1)                # plus one Open Library query
        self.assertEqual(self.calls[1], "systems chip huyen")                       # last spine of a merged reading
        self.assertEqual(self.calls[2], "designing machine learning systems chip")   # then the last word dropped

    def test_publisher_word_at_the_end_is_retried_without(self):
        from pipeline.catalog import search
        search("staff engineers path oreilly", limit=10)
        self.assertEqual(self.calls[1], "staff engineers path")

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

    def test_ampersand_and_bracketed_notes_do_not_split_a_book(self):
        from pipeline.catalog import same_work
        self.assertTrue(same_work("How to Win Friends and Influence People", "Dale Carnegie",
                                  "How to Win Friends & Influence People", "Dale Carnegie"))
        self.assertTrue(same_work("The Mom Test", "Rob Fitzpatrick", "The MOM Test", "Rob Fitzpatrick (Emprenedor)"))
        self.assertTrue(same_work("Krieg & Frieden", "Leo Tolstoi", "Krieg und Frieden", "Leo Tolstoi"))   # any language
        self.assertFalse(same_work("Pride & Prejudice", "Jane Austen", "Pride and Zombies", "Jane Austen"))

    def test_sequels_and_other_authors_stay_apart(self):
        from pipeline.catalog import same_work
        self.assertFalse(same_work("Dune", "Frank Herbert", "Dune Messiah", "Frank Herbert"))
        self.assertFalse(same_work("Alone", "Richard D. Logan", "How to Not Die Alone", "Logan Ury"))
        self.assertFalse(same_work("Zero to One", "Peter Thiel", "Zero to One Hundred", "Peter Smith"))


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


def listing(title, value, currency="EUR", condition="Gut"):
    return {"title": title, "price": {"value": str(value), "currency": currency}, "condition": condition,
            "itemWebUrl": f"https://www.ebay.de/itm/{abs(hash((title, value))) % 10**6}"}


class FakeEbay(EbayPrices):
    """The real filtering and median, with the HTTP call replaced: (condition, marketplace) -> listings."""

    def __init__(self, listings):
        super().__init__("id", "secret")
        self.listings, self.asked = listings, []

    def _search(self, params, marketplace):
        self.asked.append(marketplace)
        key = ("new", marketplace) if "conditionIds:{1000}" in params["filter"] else marketplace
        return {"itemSummaries": self.listings.get(key, [])}


def new_copies(market, price, currency="EUR", title="The Lean Startup Eric Ries"):
    return FakeEbay({("new", market): [listing(f"{title} neu", price, currency, "Neu")]})


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
        cost = price_book(self.book, "DE", "EUR", RATES, 150, new_copies("EBAY_DE", 12.99))["replacement_cost"]
        self.assertEqual((cost["amount"], cost["source"]), (12.99, "eBay Browse API (current new listings)"))
        self.assertFalse(cost["converted"])

    def test_same_currency_neighbour_is_used_before_conversion(self):
        ebay = FakeEbay({("new", "EBAY_AT"): [listing("The Lean Startup Eric Ries", 11.0, condition="Neu")],
                         ("new", "EBAY_US"): [listing("The Lean Startup Eric Ries", 9.0, "USD", "New")]})
        cost = price_book(self.book, "DE", "EUR", RATES, 150, ebay)["replacement_cost"]
        self.assertEqual(cost["country"], "AT")
        self.assertFalse(cost["converted"])

    def test_foreign_price_is_converted_with_ecb_rate(self):
        cost = price_book(self.book, "DE", "EUR", RATES, 150, new_copies("EBAY_US", 11.17, "USD"))["replacement_cost"]
        self.assertTrue(cost["converted"])
        self.assertAlmostEqual(cost["amount"], 10.0, places=2)
        self.assertEqual(cost["original"], {"amount": 11.17, "currency": "USD"})
        self.assertEqual(cost["fx"]["date"], "2026-10-30")

    def test_no_new_listing_anywhere_leaves_replacement_empty(self):
        result = price_book(self.book, "DE", "EUR", RATES, 150, FakeEbay({}))
        self.assertEqual(result["status"], "identified")
        self.assertIsNone(result["replacement_cost"]["amount"])
        self.assertIn("no new listing", result["replacement_cost"]["reason"])

    def test_without_ebay_credentials_nothing_is_priced(self):
        result = price_book(self.book, "DE", "EUR", RATES, 150, None)
        self.assertIsNone(result["replacement_cost"]["amount"])
        self.assertIn("EBAY_CLIENT_ID", result["replacement_cost"]["reason"])

    def test_expensive_book_goes_to_appraisal_unpriced(self):
        result = price_book(self.book, "DE", "EUR", RATES, 150, new_copies("EBAY_DE", 420.0))
        self.assertEqual(result["status"], "needs_appraisal")
        self.assertIsNone(result["replacement_cost"])
        self.assertIsNone(result["used_value"]["amount"])

    def test_threshold_is_in_dollars_and_converted(self):
        # 150 USD is about 14,240 INR at the sample rate: a 1,200 INR book is priced, a 20,000 INR one is not.
        # India has no eBay site: US listings are converted.
        cheap = price_book(self.book, "IN", "INR", RATES, 150, new_copies("EBAY_US", 1200 / 106.084 * 1.1172, "USD"))
        dear = price_book(self.book, "IN", "INR", RATES, 150, new_copies("EBAY_US", 20000 / 106.084 * 1.1172, "USD"))
        self.assertEqual(cheap["status"], "identified")
        self.assertEqual(dear["status"], "needs_appraisal")
        self.assertAlmostEqual(local_threshold(150, "INR", RATES), 150 / 1.1172 * 106.084, delta=0.01)
        self.assertIn(f"{local_threshold(150, 'INR', RATES)} INR", dear["appraisal_reason"])

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
        self.assertEqual(price_book(old, "DE", "EUR", RATES, 150, new_copies("EBAY_DE", 9.0))["status"], "needs_appraisal")


class TraceableEbayPriceTest(unittest.TestCase):
    def test_reported_price_is_a_real_listing_and_the_link_goes_to_it(self):
        # Even count: the plain median (9.62) matched no listing; the lower middle (7.99) is one.
        prices = [3.12, 6.0, 7.99, 11.24, 15.0, 20.0]
        ebay = FakeEbay({"EBAY_US": [listing(f"The Lean Startup Eric Ries copy {i}", p, "USD") for i, p in enumerate(prices)]})
        used = price_book({"title": "The Lean Startup", "author": "Eric Ries", "first_publish_year": 2011},
                          "US", "USD", RATES, 150, ebay)["used_value"]
        self.assertEqual(used["amount"], 7.99)
        linked = next(l for l in used["listings"] if l["url"] == used["url"])
        self.assertEqual(linked["amount"], 7.99)
        self.assertEqual([l["amount"] for l in used["listings"]], prices)


class UsedValueTest(unittest.TestCase):
    book = {"title": "Zero to One", "author": "Peter Thiel", "first_publish_year": 2014}

    def price(self, ebay, home="DE", currency="EUR"):
        return price_book(self.book, home, currency, RATES, 150, ebay)

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

    def test_ebay_failure_is_marked_for_retry(self):
        class Down(FakeEbay):
            def _search(self, params, marketplace):
                raise OSError("eBay unreachable")
        result = self.price(Down({}))
        self.assertIsNone(result["replacement_cost"]["amount"])
        self.assertIsNone(result["used_value"]["amount"])
        self.assertTrue(result["retry"])

    def test_appraisal_books_get_no_used_value(self):
        ebay = FakeEbay({"EBAY_DE": [listing("Zero to One Peter Thiel", 900.0)],
                         ("new", "EBAY_DE"): [listing("Zero to One Peter Thiel", 420.0, condition="Neu")]})
        result = self.price(ebay)
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

    def test_daily_quota_pauses_search_for_an_hour(self):
        message = "Quota exceeded for quota metric 'Queries' and limit 'Queries per day'"
        self.assertTrue(self.pricing.note_google_books_error(self.http_error(429, message), message))
        self.assertTrue(self.pricing.google_books_paused())
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


class EbookReferenceTest(unittest.TestCase):
    book = {"title": "How to Win Friends and Influence People", "author": "Dale Carnegie", "first_publish_year": 1936}

    class Ebooks:
        source = "Google Books API (Google Play eBook)"

        def __init__(self, offers):
            self.offers_by_country, self.asked = offers, []

        def offers(self, title, author, country):
            self.asked.append(country)
            return self.offers_by_country.get(country, [])

    def ebook(self, amount):
        return {"amount": amount, "currency": "INR", "edition_title": "How to Win Friends", "url": "https://play.google.com/x",
                "retrieved_at": "2026-11-04"}

    PRINTED = "How to Win Friends and Influence People Dale Carnegie"

    def test_country_with_ebay_uses_ebay_and_never_asks_for_an_ebook(self):
        ebooks = self.Ebooks({"DE": [self.ebook(9.99)]})
        result = price_book(self.book, "DE", "EUR", RATES, 150, new_copies("EBAY_DE", 12.0, "EUR", self.PRINTED), ebooks)
        self.assertEqual(result["replacement_cost"]["amount"], 12.0)
        self.assertNotEqual(result["replacement_cost"].get("edition"), "eBook")
        self.assertEqual(ebooks.asked, [])

    def test_country_without_ebay_uses_its_ebook_price_labelled_as_ebook(self):
        from pipeline.pricing import book_totals
        ebooks = self.Ebooks({"IN": [self.ebook(250.0)]})
        result = price_book(self.book, "IN", "INR", RATES, 150, new_copies("EBAY_US", 14.99, "USD", self.PRINTED), ebooks)
        cost = result["replacement_cost"]
        self.assertEqual((cost["amount"], cost["edition"], cost["country"], cost["converted"]), (250.0, "eBook", "IN", False))
        self.assertEqual(ebooks.asked, ["IN"])
        self.assertEqual(book_totals([result])["books_replacement_cost"], 250.0)

    def test_no_ebook_falls_back_to_other_ebay_markets_and_says_why(self):
        result = price_book(self.book, "IN", "INR", RATES, 150, new_copies("EBAY_US", 14.99, "USD", self.PRINTED), self.Ebooks({}))
        cost = result["replacement_cost"]
        self.assertEqual((cost["country"], cost["converted"]), ("US", True))
        self.assertIn("no eBook for sale on Google Play in IN", cost["instead_of"])
        self.assertIn("no eBook", result["ebook_lookup"])

    def test_google_books_unavailable_falls_back_too(self):
        down = self.Ebooks({})
        down.offers = lambda title, author, country: None
        result = price_book(self.book, "IN", "INR", RATES, 150, new_copies("EBAY_US", 14.99, "USD", self.PRINTED), down)
        self.assertEqual(result["replacement_cost"]["country"], "US")
        self.assertIn("not looked up", result["ebook_lookup"])

    def test_summaries_and_other_authors_are_not_the_book(self):
        from unittest import mock
        from pipeline import pricing
        items = [{"volumeInfo": {"title": t, "authors": a, "language": "en"},
                  "saleInfo": {"saleability": "FOR_SALE", "retailPrice": {"amount": p, "currencyCode": "INR"}, "buyLink": "u"}}
                 for t, a, p in [("The Lean Startup", ["Eric Ries"], 499.0),
                                 ("The Lean Startup: Summary", ["Eric Ries"], 99.0),
                                 ("Lessons Inspired by The Lean Startup", ["Someone Else"], 115.0)]]
        with mock.patch.object(pricing, "get_cached", return_value=({"items": items}, "2026-11-04")), \
                mock.patch.object(pricing, "pace_google_books", lambda: None):
            found = pricing.GoogleBooksEbooks("k").offers("The Lean Startup", "Eric Ries", "IN")
        self.assertEqual([o["amount"] for o in found], [499.0])
