import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from pipeline import live_books


def textured(seed: int) -> np.ndarray:
    """A distinct scene per seed; identical seeds give near-duplicate frames."""
    rng = np.random.default_rng(seed)
    return cv2.GaussianBlur((rng.random((240, 320, 3)) * 255).astype(np.uint8), (0, 0), 2)


def accepted(work_key, title, author):
    return {"work_key": work_key, "title": title, "subtitle": "", "author": author, "first_publish_year": 2011,
            "reading": title.upper(), "frame_refs": [], "evidence": title, "title_coverage": 1.0, "author_hit": True, "matched_by": "fake",
            "catalogue": {"source": "Open Library"}}


LEAN = accepted("/works/1", "The Lean Startup", "Eric Ries")
ZERO = accepted("/works/2", "Zero to One", "Peter Thiel")


async def fake_run(agent, frames, country, currency):
    """Stands in for books_agent: frame f00000 shows one book, f00001 that book and another."""
    refs = [f["ref"] for f in frames]
    found = [{**LEAN, "frame_refs": refs}] + ([{**ZERO, "frame_refs": ["f00001"]}] if "f00001" in refs else [])
    priced = [{"work_key": b["work_key"], **fake_price(b, None, country, currency, None, None)} for b in found] if currency else []
    return {"ocr_lines": {}, "matches": {"by": "fake"}, "verified": {"accepted": found, "suggested": [], "unread": []}, "priced": priced}


def fake_price(book, provider, home, currency, rates, threshold, used_provider=None):
    return {"status": "identified", "replacement_cost": {"amount": 10.0, "currency": currency, "converted": False},
            "used_value": {"amount": None}}


class BookJobTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        for seed, name in enumerate(("f00000", "f00001")):
            cv2.imwrite(str(self.dir / f"{name}.jpg"), textured(seed))
        cv2.imwrite(str(self.dir / "f00002.jpg"), textured(0))          # camera held still: repeats f00000
        self.patches = [patch.object(live_books, "run_books_agent", fake_run),
                        patch.object(live_books, "create_books_agent", lambda reader, price, known=None: object()),
                        patch.object(live_books, "price_book", fake_price),
                        patch.object(live_books, "load_books_key", lambda: None),
                        patch.object(live_books, "ebay_used_prices", lambda: None)]
        for p in self.patches:
            p.start()
        self.job = live_books.BookJob(self.dir, lambda: None, lambda: None)

    async def asyncTearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    async def drain(self):
        while self.job.task and not self.job.task.done():
            await asyncio.sleep(0.01)

    async def test_books_merge_across_frames_and_are_priced(self):
        self.job.add("f00000", self.dir / "f00000.jpg", "DE", "EUR")
        await self.drain()
        status = self.job.status()
        self.assertEqual((status["frames_read"], status["totals"]["books_identified"], status["totals"]["books_replacement_cost"]), (1, 1, 10.0))
        self.job.add("f00001", self.dir / "f00001.jpg", "DE", "EUR")
        await self.drain()
        status = self.job.status()
        self.assertEqual([b["title"] for b in status["books"]], ["The Lean Startup", "Zero to One"])
        self.assertEqual(status["totals"]["books_replacement_cost"], 20.0)
        by_frame, _ = self.job.books_by_frame()
        self.assertEqual(by_frame["f00001"], {"/works/1", "/works/2"})          # Lean Startup seen on both frames, counted once
        self.assertTrue((self.dir / "books_priced.json").exists())

    async def test_near_duplicate_frame_is_not_sent_to_books_agent(self):
        sent = []

        async def counting_run(agent, frames, *args):
            sent.extend(f["ref"] for f in frames)
            return await fake_run(agent, frames, *args)

        with patch.object(live_books, "run_books_agent", counting_run):
            self.job.add("f00000", self.dir / "f00000.jpg", "DE", "EUR")
            self.job.add("f00002", self.dir / "f00002.jpg", "DE", "EUR")     # same view
            self.job.add("f00001", self.dir / "f00001.jpg", "DE", "EUR")     # new view
            await self.drain()
        self.assertEqual(sent, ["f00000", "f00001"])
        self.assertEqual(self.job.status()["frames_skipped_duplicate"], 1)
        by_frame, _ = self.job.books_by_frame()
        self.assertEqual(by_frame["f00002"], by_frame["f00000"])               # the skipped frame shares its books
        self.assertTrue(await self.job.wait_for(["f00002"], timeout=1))        # and never blocks a shelf analysis

    async def test_title_fragment_of_another_book_is_dropped(self):
        designing = {**accepted("/works/9", "Designing Machine Learning Systems", "Chip Huyen"), "frame_refs": ["f00000"]}
        fragment = {**accepted("/works/8", "Machine Learning Systems", "Jeff Smith"), "author_hit": False, "frame_refs": ["f00000"]}
        self.job._merge({"matches": {"by": "rules"}, "verified": {"accepted": [fragment], "suggested": [], "unread": []}})
        self.job._merge({"matches": {"by": "rules"}, "verified": {"accepted": [designing], "suggested": [], "unread": []}})
        self.assertEqual([b["title"] for b in self.job.identified], ["Designing Machine Learning Systems"])

    async def test_same_book_from_two_catalogues_is_counted_once(self):
        ol = {**accepted("/works/OL1W", "After the Affair", "Julia Cole"), "frame_refs": ["f00000"]}
        gb = {**accepted("gb:relate-after-the-affair|cole", "Relate - After The Affair", "Julia Cole"), "frame_refs": ["f00001"]}
        for line in (ol, gb):
            self.job._merge({"matches": {"by": "rules"}, "verified": {"accepted": [line], "suggested": [], "unread": []}})
        self.assertEqual([(b["title"], b["frame_refs"]) for b in self.job.identified], [("After the Affair", ["f00000", "f00001"])])

    async def test_frames_are_measured_after_their_ocr(self):
        import numpy as np
        from pipeline import measure

        class FakeMeasurer:
            depth_model = None

            def __init__(self):
                self.seen = []

            def measure_frame(self, ref, path, lines, overlay_dir=None):
                self.seen.append((ref, lines))
                mask = np.zeros((600, 400), dtype=bool)
                mask[100:520, 100:150] = True      # 420 px tall, 50 px thick
                spine = measure.Spine(ref, "THE LEAN STARTUP ERIC RIES", "top_to_bottom", 419, 49, (125, 310), 0,
                                      False, 0.95, None, sides=measure.side_vectors(mask))
                return measure.FrameResult(ref, [spine], {"px_per_cm": 20.0, "px_per_cm_xy": None}, None)

        async def run_with_text(agent, frames, country, currency):
            result = await fake_run(agent, frames, country, currency)
            return {**result, "ocr_lines": {f["ref"]: {"top_to_bottom": [{"text": "THE LEAN STARTUP"}]} for f in frames}}

        fake = FakeMeasurer()
        job = live_books.BookJob(self.dir, lambda: None, lambda: None, measurer=lambda: fake)
        with patch.object(live_books, "run_books_agent", run_with_text):
            job.add("f00000", self.dir / "f00000.jpg", "DE", "EUR")
            while (job.task and not job.task.done()) or (job.measure_task and not job.measure_task.done()):
                await asyncio.sleep(0.01)
        self.assertEqual(fake.seen, [("f00000", {"top_to_bottom": [{"text": "THE LEAN STARTUP"}]})])
        result = job.measurement()
        self.assertEqual((result["state"], result["frames_measured"]), ("done", 1))
        book = result["report"]["books"][0]
        self.assertEqual((book["title"], book["scale_method"]), ("The Lean Startup", "card"))
        self.assertAlmostEqual(book["spine_height_cm"], 21.0, delta=0.2)

    async def test_book_left_out_by_title_is_kept_on_record_but_not_in_totals(self):
        self.job.add("f00001", self.dir / "f00001.jpg", "DE", "EUR")     # Lean Startup and Zero to One, 10 EUR each
        await self.drain()
        result = self.job.exclude("zero to one", reason="borrowed from a friend")
        self.assertEqual((result["status"], result["title"]), ("excluded", "Zero to One"))
        status = self.job.status()
        self.assertEqual((status["totals"]["books_identified"], status["totals"]["books_replacement_cost"],
                          status["totals"]["books_excluded"]), (1, 10.0, 1))
        self.assertEqual({b["title"]: b["excluded"] for b in status["books"]}, {"The Lean Startup": False, "Zero to One": True})
        by_frame, _ = self.job.books_by_frame()
        self.assertEqual(by_frame["f00001"], {"/works/1"})                 # places and analyse no longer list it
        self.assertEqual(self.job.exclude("Zero to One", undo=True)["status"], "restored")
        self.assertEqual(self.job.status()["totals"]["books_identified"], 2)

    async def test_book_in_view_is_left_out_and_two_in_view_are_asked_about(self):
        self.job.add("f00000", self.dir / "f00000.jpg", "DE", "EUR")
        self.job.add("f00001", self.dir / "f00001.jpg", "DE", "EUR")
        await self.drain()
        self.assertEqual(self.job.exclude("", ["f00001"])["status"], "ambiguous")      # two books in view: ask
        result = self.job.exclude("", ["f00000"])                                       # one book in view
        self.assertEqual((result["status"], result["title"]), ("excluded", "The Lean Startup"))

    async def test_title_not_identified_yet_is_left_out_when_it_is(self):
        self.assertEqual(self.job.exclude("Zero to One")["status"], "pending")
        self.job.add("f00001", self.dir / "f00001.jpg", "DE", "EUR")
        await self.drain()
        self.assertEqual([b["title"] for b in self.job.status()["books"] if b["excluded"]], ["Zero to One"])
        self.assertEqual(self.job.pending_exclusions, [])

    async def test_title_with_and_without_the_article_is_one_book(self):
        self.job._merge({"matches": {"by": "rules"}, "verified": {"accepted": [
            {**accepted("gb:lean-startup|ries", "Lean Startup", "Eric Ries"), "frame_refs": ["f00000"]},
            {**accepted("/works/OL1W", "The Lean Startup", "Eric Ries"), "frame_refs": ["f00001"]}], "suggested": [], "unread": []}})
        self.assertEqual(len(self.job.identified), 1)

    async def test_edition_wording_does_not_make_a_second_book(self):
        self.job._merge({"matches": {"by": "rules"}, "verified": {"accepted": [
            {**accepted("/works/OL9W", "The Intelligent Investor", "Benjamin Graham"), "frame_refs": ["f00000"]},
            {**accepted("gb:intelligent-investor|graham", "The Intelligent Investor Third Edition", "Benjamin Graham"),
             "frame_refs": ["f00001"]}], "suggested": [], "unread": []}})
        self.assertEqual(len(self.job.identified), 1)

    async def test_confirmed_copies_multiply_count_and_total(self):
        self.job.add("f00001", self.dir / "f00001.jpg", "DE", "EUR")     # Lean Startup and Zero to One, 10 EUR each
        await self.drain()
        self.assertEqual(self.job.set_copies("zero to one", 3), {"status": "set", "title": "Zero to One", "copies": 3})
        totals = self.job.status()["totals"]
        self.assertEqual((totals["books_identified"], totals["titles_identified"], totals["books_replacement_cost"]), (4, 2, 40.0))

    async def test_title_on_two_spines_in_one_frame_is_flagged_until_confirmed(self):
        self.job.add("f00000", self.dir / "f00000.jpg", "DE", "EUR")     # The Lean Startup
        await self.drain()
        spine = lambda x0: {"text": "THE LEAN STARTUP ERIC RIES", "box": [0.1, x0, 0.9, x0 + 0.05], "confidence": 0.98}
        self.job.lines["f00000"] = {"horizontal": [spine(0.1), spine(0.5)], "top_to_bottom": [], "bottom_to_top": []}
        flagged = self.job.status()["possible_copies"]
        self.assertEqual([(c["title"], c["spines_in_one_frame"]) for c in flagged], [("The Lean Startup", 2)])
        self.assertEqual(self.job.status()["totals"]["books_identified"], 1)        # a guess changes no count
        self.job.set_copies("The Lean Startup", 2)
        self.assertEqual(self.job.status()["possible_copies"], [])
        self.assertEqual(self.job.status()["totals"]["books_identified"], 2)

    async def test_pricing_failure_leaves_book_unpriced_and_is_retried(self):
        calls = {"n": 0}

        def flaky(*args):
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError("HTTP Error 429: rate limited")
            return fake_price(*args)

        with patch.object(live_books, "price_book", flaky):
            self.job.add("f00000", self.dir / "f00000.jpg", "DE", "")       # identified without a currency
            await self.drain()
            await self.job.set_locale("DE", "EUR")                           # first price attempt fails
            self.assertEqual(self.job.status()["totals"]["books_priced"], 0)
            self.assertIn("pricing The Lean Startup", self.job.status()["errors"][-1])
            self.job.maybe_retry(every_s=0)
            await self.drain()
        self.assertEqual(self.job.status()["totals"]["books_priced"], 1)

    async def test_country_set_later_prices_what_is_already_identified(self):
        self.job.add("f00000", self.dir / "f00000.jpg")
        await self.drain()
        self.assertTrue(self.job.status()["waiting_for_country"])
        self.assertEqual(self.job.status()["totals"]["books_priced"], 0)
        await self.job.set_locale("DE", "EUR")
        self.assertEqual(self.job.status()["totals"]["books_priced"], 1)


if __name__ == "__main__":
    unittest.main()
