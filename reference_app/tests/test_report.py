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
        "country": "Germany", "currency": "EUR", "purpose": "claim", "purpose_details": "water damage"}
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
        self.assertIn(("room", "room not measured"), reasons)
        status = {i["id"]: i["status"] for i in self.packet["items"]}
        self.assertEqual((status["I002"], status["I003"]), (None, "needs_appraisal"))   # I002 is a print

    GOOD_ROOM = {"length_m": 5.1, "width_m": 4.3, "height_m": 2.6, "floor_area_m2": 21.93, "floor_area_ft2": 236.1,
                 "wall_area_m2": 49.4, "wall_area_ft2": 531.7, "shelved_wall_area_m2": 1.6, "shape": "rectangle",
                 "scale_method": "bank card", "scale_estimated": False, "confidence": 0.7, "walls_found": 3,
                 "floor_plane_found": True, "ceiling_plane_found": True, "frames_used": ["r00001", "f00012"]}

    def test_reliable_room_fills_the_contract(self):
        packet = build_packet(META, ITEMS, {**BOOKS, "room": {"state": "done", "room": self.GOOD_ROOM}})
        self.assertEqual((packet["room"]["floor_area_m2"], packet["room"]["height_m"], packet["room"]["shelved_wall_area_m2"]),
                         (21.93, 2.6, 1.6))
        self.assertEqual(packet["room"]["frames_used"], ["room_frames/r00001.jpg", "frames/f00012.jpg"])
        self.assertFalse([r for r in packet["review_queue"] if r["ref_id"] == "room"])
        self.assertIn("21.93 m²", build_report(packet))

    def test_unreliable_room_is_left_empty_with_the_reasons(self):
        weak = {**self.GOOD_ROOM, "walls_found": 0, "confidence": 0.22, "scale_estimated": True}
        packet = build_packet(META, ITEMS, {**BOOKS, "room": {"state": "done", "room": weak}})
        self.assertTrue(all(v is None for v in packet["room"].values()))
        reason = next(r["reason"] for r in packet["review_queue"] if r["ref_id"] == "room")
        for part in ("left empty", "0 wall(s) found", "confidence 0.22", "estimated by Depth Pro"):
            self.assertIn(part, reason)
        self.assertNotIn("21.93", build_report(packet))

    def test_room_without_a_ceiling_leaves_height_and_walls_empty(self):
        room = {**self.GOOD_ROOM, "ceiling_plane_found": False}
        packet = build_packet(META, ITEMS, {**BOOKS, "room": {"state": "done", "room": room}})
        self.assertEqual((packet["room"]["height_m"], packet["room"]["wall_area_m2"], packet["room"]["floor_area_m2"]), (None, None, 21.93))
        self.assertIn("Ceiling height", build_report(packet))

    def test_skipped_room_turn_is_not_an_error(self):
        packet = build_packet(META, ITEMS, {**BOOKS, "room": {"state": "failed",
                                                             "error": "ValueError: only 3 room frames saved; at least 8 are needed"}})
        self.assertIn("room not measured: the room turn was skipped or too short", [r["reason"] for r in packet["review_queue"]])

    def test_room_still_running_is_said(self):
        packet = build_packet(META, ITEMS, {**BOOKS, "room": {"state": "running"}})
        self.assertIsNone(packet["room"]["floor_area_m2"])
        self.assertIn("room measurement still running; reload the report", [r["reason"] for r in packet["review_queue"]])

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

    def test_policyholder_count_adds_unidentified_books(self):
        books = {**BOOKS, "place_counts": [{"place": "bookshelf by the window, row 1", "stated": 12, "identified": 2},
                                           {"place": "coffee table", "stated": None, "identified": 1}]}
        packet = build_packet(META, ITEMS, books)
        t = packet["totals"]
        self.assertEqual((t["books_identified"], t["books_unidentified"], t["book_count"]), (4, 10, 14))
        self.assertTrue(any(r["ref_id"] == "bookshelf by the window, row 1" and "10 counted but not identified" in r["reason"]
                            for r in packet["review_queue"]))

    def test_used_value_from_few_listings_is_flagged(self):
        used = {"amount": 24.3, "currency": "EUR", "source": "eBay Browse API (current used listings)", "url": "https://www.ebay.de/itm/1",
                "retrieved_at": "2026-11-03", "condition_assumed": "used", "listings_considered": 1}
        books = {**BOOKS, "books": [line(1, "Zero to One", "Peter Thiel", price(9.99), used_value=used)]}
        packet = build_packet(META, ITEMS, books)
        self.assertEqual(packet["totals"]["books_used_value"], 24.3)
        self.assertIn("used value from only 1 eBay listing(s)", [r["reason"] for r in packet["review_queue"]])

    def test_left_out_book_is_kept_apart(self):
        self.assertNotIn("Ikigai", [b["title"] for b in self.packet["books"]])
        self.assertEqual(self.packet["excluded_books"][0]["excluded"]["reason"], "borrowed from my sister")


class ReportTest(unittest.TestCase):
    def report(self, books=BOOKS, usage=None):
        return build_report(build_packet(META, ITEMS, books), usage)

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
        busy = {**BOOKS, "still_reading": True, "room": {"state": "running"}}
        page = self.report(busy)
        self.assertIn('<meta name="report-state" content="in-progress">', page)
        self.assertIn("Still working on: reading book spines, measuring the room", page)
        self.assertIn('<meta name="report-state" content="complete">', self.report({**BOOKS, "room": {"state": "done", "room": None}}))

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
        self.assertEqual(result["room"], "off")                     # no worker in this test
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
