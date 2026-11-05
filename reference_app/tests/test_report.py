import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

APP = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(APP), str(APP / "live_demo")]

from fastapi.testclient import TestClient  # noqa: E402

import server as s  # noqa: E402
from packet import build_packet  # noqa: E402
from report import build_report  # noqa: E402

META = {"sweep_id": "live-test", "captured_at": "2026-11-03T10:00:00+0100", "device": "test browser", "duration_s": 171.5,
        "country": "Germany", "currency": "EUR"}
ITEMS = [{"id": "I001", "category": "furniture", "description": "grey sofa", "brand_model": "", "where": "living room", "notes": [],
          "frame_ref": "items/I001.jpg"},
         {"id": "I002", "category": "art", "description": "painting <b>by the door</b>", "brand_model": "", "where": "", "notes": ["a print"]},
         {"id": "I003", "category": "art", "description": "small framed portrait", "brand_model": "", "where": "", "notes": []}]


def price(amount, **extra):
    return {"amount": amount, "currency": "EUR", "source": "Google Books API", "url": "https://example.test/p",
            "retrieved_at": "2026-11-03T10:00:00+0100", "country": "DE", "condition_assumed": "new, eBook edition", **extra}


def line(n, title, author, cost, **extra):
    return {"id": f"book-{n:03d}", "work_key": f"/works/{n}", "title": title, "author": author, "frame_ref": f"f0000{n}",
            "frame_refs": [f"f0000{n}"], "id_confidence": 1.0, "status": "identified", "replacement_cost": cost,
            "used_value": {"amount": None, "reason": "no used-market price source configured"}, "excluded": None,
            "copies": 1, "copies_confirmed": False, **extra}


BOOKS = {
    "country": "DE", "currency": "EUR",
    "books": [line(1, "Zero to One", "Peter Thiel", price(9.99), copies=2, copies_confirmed=True),
              line(2, "Spark", "Bill Chambers", price(40.0)),
              line(3, "Ikigai", "Hector Garcia", price(12.0), excluded={"reason": "borrowed from my sister", "at": "2026-11-03T10:05:00+0100"}),
              line(4, "The Mom Test", "Rob Fitzpatrick", {"amount": None, "reason": "no price found in DE, US"}, id_confidence=0.7)],
    "shelf_by_work": {"/works/1": "bookshelf by the window, row 1"},
    "pending_exclusions": [], "suggested": [], "possible_copies": [],
    "unread_readings": [{"text": "HOW GOOGLE WORKS", "frame_refs": ["f00002"], "ocr_confidence": 0.97}],
    "measurement": {"state": "done", "frames_measured": 12, "report": {
        "card_frames": [{"frame_ref": "f00003"}], "shelf_run_m": 0.05,
        "books": [{"title": "Zero to One", "author": "Peter Thiel", "spine_height_cm": 21.2, "spine_thickness_cm": 1.6,
                   "scale_method": "card", "estimated": False}]}},
}

CONTRACT = {
    "sweep": {"id", "captured_at", "device", "duration_s", "country", "currency"},
    "room": {"length_m", "width_m", "height_m", "floor_area_m2", "wall_area_m2", "shelved_wall_area_m2", "scale_method", "confidence"},
    "book": {"id", "shelf", "position", "frame_ref", "status", "title", "author", "edition", "isbn", "spine_height_cm",
             "spine_thickness_cm", "id_confidence", "replacement_cost", "used_value"},
    "item": {"id", "category", "description", "brand_model", "frame_ref", "dimensions_cm", "status", "replacement_cost", "confidence"},
    "totals": {"book_count", "books_identified", "books_unidentified", "shelf_run_m", "books_replacement_cost", "books_used_value",
               "items_replacement_cost_low", "items_replacement_cost_high", "excluded_from_totals"},
}


class PacketTest(unittest.TestCase):
    def setUp(self):
        self.packet = build_packet(META, ITEMS, BOOKS)

    def test_every_contract_field_is_present(self):
        p = self.packet
        self.assertTrue(CONTRACT["sweep"] <= set(p["sweep"]))
        self.assertTrue(CONTRACT["room"] <= set(p["room"]))
        self.assertTrue(all(CONTRACT["book"] <= set(b) for b in p["books"]))
        self.assertTrue(all(CONTRACT["item"] <= set(i) for i in p["items"]))
        self.assertTrue(CONTRACT["totals"] <= set(p["totals"]))
        self.assertTrue({"amount", "source", "url", "retrieved_at", "converted"} <= set(p["books"][0]["replacement_cost"]))
        self.assertTrue({"amount", "source", "url", "retrieved_at", "condition_assumed"} <= set(p["books"][0]["used_value"]))
        self.assertTrue({"w", "h", "d"} == set(p["items"][0]["dimensions_cm"]))
        self.assertTrue({"low", "high", "source", "url", "retrieved_at"} <= set(p["items"][0]["replacement_cost"]))
        self.assertTrue(all({"ref_id", "reason"} == set(r) for r in p["review_queue"]))
        json.dumps(p)       # serialisable

    def test_totals_add_up_from_the_lines(self):
        t = self.packet["totals"]
        self.assertEqual(t["book_count"], 4)                      # Zero to One x2, Spark, Mom Test; Ikigai left out
        self.assertEqual(t["books_replacement_cost"], 59.98)      # 9.99 x 2 + 40.00; the unpriced book adds nothing
        self.assertEqual(t["excluded_from_totals"], 1 + 3)        # the unpriced book and three unpriced items
        self.assertIsNone(t["books_used_value"])                  # no used price anywhere: empty, not 0
        self.assertIsNone(t["items_replacement_cost_low"])

    def test_unknown_fields_are_empty_not_zero(self):
        self.assertTrue(all(v is None for v in self.packet["room"].values()))
        spark = next(b for b in self.packet["books"] if b["title"] == "Spark")
        self.assertIsNone(spark["spine_height_cm"])
        self.assertIsNone(spark["position"])

    def test_frames_and_shelves_are_traceable(self):
        zero = self.packet["books"][0]
        self.assertEqual((zero["frame_ref"], zero["shelf"]), ("frames/f00001.jpg", "bookshelf by the window, row 1"))
        self.assertEqual(self.packet["items"][0]["frame_ref"], "items/I001.jpg")

    def test_review_queue_says_why(self):
        reasons = {(r["ref_id"], r["reason"].split(":")[0]) for r in self.packet["review_queue"]}
        self.assertIn(("book-004", "no price"), reasons)
        self.assertIn(("book-004", "identification confidence 0.7"), reasons)
        self.assertIn(("I003", "art"), reasons)                    # portrait not said to be a print
        self.assertIn(("room", "room length and width not reported"), reasons)
        status = {i["id"]: i["status"] for i in self.packet["items"]}
        self.assertEqual((status["I002"], status["I003"]), (None, "needs_appraisal"))   # I002 is a print

    ROOM = {"length_m": 4.52, "width_m": 4.06, "corner_offset_m": 0.5, "method": "Depth Pro depth scaled by an A4 sheet",
            "walls": [{"wall": "1", "distance_m": 3.56, "measured": 4, "spread_m": 0.08, "reasons": []},
                      {"wall": "2", "distance_m": 4.02, "measured": 3, "spread_m": 0.2, "reasons": []}],
            "frames_used": ["room_frames/wall1_0.jpg", "room_frames/wall2_0.jpg"], "flags": []}

    def test_measured_room_fills_length_width_and_area(self):
        packet = build_packet(META, ITEMS, {**BOOKS, "room": {"state": "done", "room": self.ROOM}})
        room = packet["room"]
        self.assertEqual((room["length_m"], room["width_m"], room["floor_area_m2"]), (4.52, 4.06, 18.35))
        self.assertIsNone(room["height_m"])
        self.assertEqual(room["frames_used"], ["room_frames/wall1_0.jpg", "room_frames/wall2_0.jpg"])
        self.assertTrue(CONTRACT["room"] <= set(room))

    def test_one_wall_or_disagreeing_frames_leave_the_room_empty(self):
        one = {**self.ROOM, "length_m": None, "width_m": None,
               "walls": [self.ROOM["walls"][0], {"wall": "2", "distance_m": None, "measured": 0, "reasons": ["sheet not found"]}]}
        packet = build_packet(META, ITEMS, {**BOOKS, "room": {"state": "done", "room": one}})
        self.assertIsNone(packet["room"]["length_m"])
        self.assertTrue(any("1 of 2 walls measured (sheet not found)" in r["reason"] for r in packet["review_queue"]))
        noisy = {**self.ROOM, "walls": [self.ROOM["walls"][0], {**self.ROOM["walls"][1], "spread_m": 1.2}]}
        packet = build_packet(META, ITEMS, {**BOOKS, "room": {"state": "done", "room": noisy}})
        self.assertIsNone(packet["room"]["length_m"])
        self.assertTrue(any("frames disagree" in r["reason"] for r in packet["review_queue"]))

    def test_room_still_being_measured_is_said_not_called_skipped(self):
        packet = build_packet(META, ITEMS, {**BOOKS, "room": {"state": "measuring", "room": None}})
        self.assertIsNone(packet["room"]["length_m"])
        self.assertIn("measuring the room", packet["pending"])
        self.assertTrue(any("still running" in r["reason"] for r in packet["review_queue"] if r["ref_id"] == "room"))

    def test_skipped_room_step_is_said(self):
        reasons = [r["reason"] for r in build_packet(META, ITEMS, BOOKS)["review_queue"] if r["ref_id"] == "room"]
        self.assertEqual(reasons, ["room length and width not reported: the room step was skipped (no wall measured)"])

    def test_detected_objects_join_the_packet_unless_already_logged(self):
        detected = {"sofa": {"count": 1, "score": 0.62, "frame_ref": "items/detected-sofa.jpg"},
                    "chair": {"count": 3, "score": 0.55, "frame_ref": "items/detected-chair.jpg"},
                    "framed_art": {"count": 2, "score": 0.4, "frame_ref": "items/detected-framed_art.jpg"}}
        logged = [{"id": "I001", "category": "furniture", "description": "grey three-seat sofa", "notes": []}]
        packet = build_packet(META, logged, BOOKS, detected)
        found = {i["category"]: i for i in packet["items"] if i["id"].startswith("D")}
        self.assertEqual(set(found), {"chair", "framed_art"})                  # the sofa was already logged
        self.assertEqual((found["chair"]["count"], found["chair"]["frame_ref"]), (3, "items/detected-chair.jpg"))
        self.assertEqual(found["framed_art"]["status"], "needs_appraisal")
        self.assertTrue(any(r["ref_id"] == found["chair"]["id"] and "not confirmed" in r["reason"] for r in packet["review_queue"]))
        page = build_report(packet)
        self.assertIn('<td>Chair</td><td class="num">3</td>', page)
        self.assertIn("detected, not confirmed", page)

    def test_used_value_from_few_listings_is_flagged(self):
        used = {"amount": 24.3, "currency": "EUR", "source": "eBay Browse API (current used listings)", "url": "https://www.ebay.de/itm/1",
                "retrieved_at": "2026-11-03", "condition_assumed": "used", "listings_considered": 1}
        books = {**BOOKS, "books": [line(1, "Zero to One", "Peter Thiel", price(9.99), used_value=used)]}
        packet = build_packet(META, ITEMS, books)
        self.assertEqual(packet["totals"]["books_used_value"], 24.3)
        self.assertIn("used value from only 1 eBay listing(s)", [r["reason"] for r in packet["review_queue"]])

    def test_book_confirmed_by_the_policyholder_is_counted_and_flagged(self):
        line = {**BOOKS["books"][0], "confirmed_by_policyholder": True}
        packet = build_packet(META, ITEMS, {**BOOKS, "books": [line] + BOOKS["books"][1:]})
        self.assertTrue(packet["books"][0]["confirmed_by_policyholder"])
        self.assertTrue(any(r["ref_id"] == packet["books"][0]["id"] and "confirmed by the policyholder" in r["reason"]
                            for r in packet["review_queue"]))

    def test_left_out_book_is_kept_apart(self):
        self.assertNotIn("Ikigai", [b["title"] for b in self.packet["books"]])
        self.assertEqual(self.packet["excluded_books"][0]["excluded"]["reason"], "borrowed from my sister")


class ReportTest(unittest.TestCase):
    def report(self, books=BOOKS, usage=None):
        return build_report(build_packet(META, ITEMS, books), usage)

    def test_report_is_the_policy_inventory(self):
        page = self.report()
        self.assertIn("<h1>Policy inventory report</h1>", page)
        self.assertIn("Adding items to the policy", page)
        self.assertNotIn("claim report", page.lower())

    def test_room_section_shows_length_width_and_frames(self):
        page = self.report({**BOOKS, "room": {"state": "done", "room": PacketTest.ROOM}})
        self.assertIn("<strong>4.52 × 4.06 m</strong>", page)
        self.assertIn('href="room_frames/wall1_0.jpg"', page)

    def test_ebook_replacement_cost_is_labelled_and_reviewed(self):
        line = {**BOOKS["books"][0], "replacement_cost": {"amount": 250.0, "currency": "EUR", "edition": "eBook", "country": "IN",
                                                          "converted": False, "source": "Google Books API (Google Play eBook)",
                                                          "url": "https://play.google.com/x", "retrieved_at": "2026-11-04"}}
        packet = build_packet(META, ITEMS, {**BOOKS, "books": [line] + BOOKS["books"][1:]})
        self.assertEqual(packet["books"][0]["replacement_cost"]["edition"], "eBook")
        self.assertTrue(any("eBook edition's price" in r["reason"] for r in packet["review_queue"]))
        self.assertIn("EBOOK PRICE", build_report(packet, None))

    def test_objects_by_type_books_and_totals(self):
        page = self.report()
        self.assertIn('<td>Art</td><td class="num">2</td>', page)
        self.assertIn('<td>Books</td><td class="num">4</td>', page)
        self.assertIn("59.98 EUR", page)
        self.assertIn("21.2 × 1.6 cm", page)
        self.assertIn("2 copies", page)
        self.assertIn('href="frames/f00001.jpg"', page)

    def test_left_out_book_is_listed_apart_with_its_reason(self):
        books_section, left_out_section = self.report().split("Left out at the policyholder")
        self.assertNotIn("Ikigai", books_section)
        self.assertIn("borrowed from my sister", left_out_section)

    def test_nothing_priced_is_not_shown_as_zero(self):
        unpriced = {**BOOKS, "books": [line(1, "Zero to One", "Peter Thiel", {"amount": None, "reason": "quota"})]}
        self.assertIn('Books replacement cost</div><div class="value">none priced', self.report(unpriced))

    def test_cost_and_latency_section(self):
        usage = {"stages": {"SearchMarket": {"count": 4, "total_s": 40.0, "mean_s": 10.0, "p50_s": 9.0, "p95_s": 18.0, "max_s": 18.0}},
                 "calls": {"gemini_matcher": {"requests": 4, "errors": 1, "input_tokens": 12000, "output_tokens": 800}},
                 "cost": {"usd_by_service": {}, "usd_total": None, "unpriced_services": ["gemini_matcher"], "rates_file": "usage_rates.json"}}
        page = self.report(usage=usage)
        self.assertIn("<td>SearchMarket</td>", page)
        self.assertIn("Not priced (no rate set): Gemini (book matching)", page)

    def test_report_says_when_it_is_still_being_worked_on(self):
        page = self.report({**BOOKS, "still_reading": True})
        self.assertIn('<meta name="report-state" content="in-progress">', page)
        self.assertIn("Still working on: reading book spines.", page)
        self.assertIn('<meta name="report-state" content="complete">', self.report(BOOKS))
        self.assertIn("Room length and width not measured in this sweep", page)

    def test_text_is_escaped(self):
        page = self.report()
        self.assertNotIn("<b>by the door</b>", page)
        self.assertIn("&lt;b&gt;by the door&lt;/b&gt;", page)

    def test_without_the_book_worker_objects_still_report(self):
        page = build_report(build_packet(META, ITEMS, None))
        self.assertIn("grey sofa", page)
        self.assertIn("The book worker did not answer", page)

    def test_session_usage_adds_app_and_worker_requests(self):
        session = s.IntakeSession("x")
        session.usage.add_call("gemini_live", input_tokens=500, output_tokens=200)
        worker = {"stages": {}, "calls": {"google_books": {"requests": 7, "errors": 0, "input_tokens": 0, "output_tokens": 0}}}
        usage = s.session_usage(session, worker)
        self.assertEqual((usage["calls"]["google_books"]["requests"], usage["calls"]["gemini_live"]["input_tokens"]), (7, 500))


class ReportEndpointTest(unittest.TestCase):
    def setUp(self):
        s.sessions.clear()
        self.tmp = tempfile.TemporaryDirectory()
        self.patches = [patch.object(s, "VISION_URL", ""), patch.object(s, "SWEEPS_DIR", Path(self.tmp.name))]
        for p in self.patches:
            p.start()
        self.client = TestClient(s.app, base_url="http://127.0.0.1:4188")
        self.headers = {"Origin": "http://127.0.0.1:4188"}

    def tearDown(self):
        self.client.close()
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()
        s.sessions.clear()

    def new_session(self):
        sid = self.client.post("/api/sessions", headers=self.headers).json()["session_id"]
        session = s.sessions[sid]
        session.sweep_id = "live-test"
        return sid, session

    def test_report_and_packet_are_saved_in_the_sweep_folder(self):
        sid, session = self.new_session()
        session.last_frame = b"\xff\xd8 fake jpeg"
        result = s.log_item(session, {"category": "furniture", "description": "oak dining table"})
        s.save_item_frame(session, result["item_id"])
        page = self.client.get(f"/api/sessions/{sid}/report", headers=self.headers)
        self.assertEqual(page.status_code, 200)
        self.assertIn("oak dining table", page.text)
        folder = Path(self.tmp.name) / "live-test"
        self.assertEqual((folder / "report.html").read_text(), page.text)
        packet = self.client.get(f"/api/sessions/{sid}/claim_packet.json", headers=self.headers).json()
        self.assertEqual(packet["items"][0]["frame_ref"], "items/I001.jpg")
        self.assertEqual(json.loads((folder / "claim_packet.json").read_text())["items"][0]["description"], "oak dining table")
        frame = self.client.get(f"/api/sessions/{sid}/items/I001.jpg", headers=self.headers)
        self.assertEqual(frame.content, b"\xff\xd8 fake jpeg")

    def test_finish_builds_the_packet_without_the_agent(self):
        sid, session = self.new_session()
        s.log_item(session, {"category": "lamp", "description": "brass floor lamp"})
        result = self.client.post(f"/api/sessions/{sid}/finish", headers=self.headers).json()
        self.assertNotIn("room", result)                            # room measurement is not part of this version
        packet = json.loads((Path(self.tmp.name) / "live-test" / "claim_packet.json").read_text())
        self.assertEqual(packet["items"][0]["description"], "brass floor lamp")

    def test_files_outside_the_frame_folders_are_refused(self):
        sid, _ = self.new_session()
        self.assertEqual(self.client.get(f"/api/sessions/{sid}/measure/f00001.jpg", headers=self.headers).status_code, 404)
        self.assertEqual(self.client.get(f"/api/sessions/{sid}/frames/..%2Fclaim_packet.json", headers=self.headers).status_code, 404)

    def test_report_needs_the_session_owner(self):
        sid = self.client.post("/api/sessions", headers=self.headers).json()["session_id"]
        self.client.cookies.clear()
        self.assertIn(self.client.get(f"/api/sessions/{sid}/report").status_code, {403, 404})


if __name__ == "__main__":
    unittest.main()
