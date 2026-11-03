import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "live_demo"))

from narration import SceneTracker, book_events, label, needs_reply, notice_text, ready_to_speak  # noqa: E402


def det(category, box, score=0.6):
    return {"category": category, "box": box, "score": score}


def frame(*dets, sharpness=300.0, width=288, height=512):
    return {"width": width, "height": height, "sharpness": sharpness, "detections": list(dets)}


SOFA = det("sofa", [150, 300, 285, 420])
CHAIR_L, CHAIR_R = det("chair", [0, 300, 60, 400]), det("chair", [70, 300, 130, 400])
SHELF = det("bookshelf", [200, 50, 285, 500])


class SceneTrackerTest(unittest.TestCase):
    def test_object_reported_once_after_two_sightings(self):
        t = SceneTracker()
        self.assertEqual(t.update(frame(SOFA), 0), [])                       # one frame is not enough
        events = t.update(frame(SOFA, CHAIR_L, CHAIR_R), 1)
        self.assertEqual(events, [{"kind": "new_objects", "objects": [
            {"category": "sofa", "count": 1, "label": "sofa", "where": "right"}]}])
        events = t.update(frame(SOFA, CHAIR_L, CHAIR_R), 2)                   # chairs now confirmed, sofa not repeated
        self.assertEqual([o["label"] for o in events[0]["objects"]], ["2 chairs"])
        self.assertEqual(t.update(frame(SOFA, CHAIR_L, CHAIR_R), 3), [])

    def test_higher_count_later_is_not_announced_again(self):
        t = SceneTracker()
        t.update(frame(CHAIR_L), 0)
        t.update(frame(CHAIR_L), 1)                                            # "chair" announced
        third = det("chair", [140, 300, 200, 400])
        t.update(frame(CHAIR_L, CHAIR_R, third), 2)
        self.assertEqual(t.update(frame(CHAIR_L, CHAIR_R, third), 3), [])      # same chairs from another angle
        self.assertEqual(t.summary()["objects_logged"], ["3 chairs"])         # progress still reflects it

    def test_single_frame_false_positive_is_ignored(self):
        t = SceneTracker()
        t.update(frame(det("lamp", [0, 0, 20, 40])), 0)
        for i in range(1, 5):
            self.assertEqual(t.update(frame(), i), [])

    def test_bookshelf_event_with_position(self):
        t = SceneTracker()
        t.update(frame(SHELF), 0)
        self.assertEqual(t.update(frame(SHELF), 1), [{"kind": "bookshelf", "count": 1, "where": "right", "first": True}])
        self.assertEqual(t.bookshelves, 1)

    def test_small_spines_ask_to_step_closer(self):
        t = SceneTracker()
        books = [det("book", [10 * i, 200, 10 * i + 8, 240]) for i in range(5)]   # 40 px tall in a 512 px frame
        self.assertEqual(t.update(frame(*books), 0), [{"kind": "spines_small", "books": 5}])
        self.assertEqual(t.update(frame(*books), 1), [])                          # not repeated straight away

    def test_blur_after_three_blurry_frames(self):
        t = SceneTracker()
        for i in range(6):
            t.update(frame(sharpness=300), i)
        kinds = [[e["kind"] for e in t.update(frame(sharpness=40), 6 + i)] for i in range(3)]
        self.assertEqual(kinds[-1], ["blurry"])

    def test_progress_summary(self):
        t = SceneTracker(progress_every_s=10)
        for i in range(4):
            t.update(frame(SOFA), i)
        progress = [e for e in t.update(frame(SOFA), 11) if e["kind"] == "progress"]
        self.assertEqual(progress[0]["objects_logged"], ["sofa"])


class SpineFrameTest(unittest.TestCase):
    books = [det("book", [10 * i, 100, 10 * i + 8, 400]) for i in range(4)]

    def test_frames_with_a_book_go_to_books_agent(self):
        t = SceneTracker()
        self.assertTrue(t.worth_reading(frame(*self.books)))
        self.assertTrue(t.worth_reading(frame(det("book", [50, 50, 250, 450]))))  # one book held up is enough
        self.assertFalse(t.worth_reading(frame(SOFA, CHAIR_L)))                    # no book detected
        self.assertFalse(t.worth_reading(frame(SHELF)))                            # empty bookshelf

    def test_blurry_spine_frame_is_not_kept(self):
        t = SceneTracker()
        for i in range(6):
            t.update(frame(sharpness=300), i)
        self.assertFalse(t.worth_reading(frame(*self.books, sharpness=60)))


class BookEventsTest(unittest.TestCase):
    def status(self, books, priced=0, total=0.0, currency="EUR", frames=1):
        return {"frames_read": frames, "queued": 0, "currency": currency, "waiting_for_country": not currency,
                "totals": {"books_identified": len(books), "books_priced": priced, "books_replacement_cost": total},
                "books": [{"work_key": k, "title": k.title(), "author": "A"} for k in books]}

    def test_new_titles_are_spoken_and_progress_is_silent(self):
        # New titles go to the agent to say once; progress is context only and carries no prices.
        first = book_events(None, self.status(["lean startup"], 1, 9.41))
        self.assertEqual([e["kind"] for e in first], ["books_identified", "books_progress"])
        self.assertTrue(needs_reply(first))
        later = book_events(self.status(["lean startup"], 1, 9.41), self.status(["lean startup"], 1, 9.41, frames=2))
        self.assertEqual([e["kind"] for e in later], ["books_progress"])
        self.assertFalse(needs_reply(later))
        self.assertEqual(book_events(self.status(["x"]), self.status(["x"])), [])

    def test_title_that_drops_out_and_returns_is_announced_once(self):
        announced = set()
        book_events(None, self.status(["a", "w"]), announced)
        book_events(self.status(["a", "w"]), self.status(["a"], frames=2), announced)       # "w" drops out
        events = book_events(self.status(["a"], frames=2), self.status(["a", "w"], frames=3), announced)
        self.assertNotIn("books_identified", [e["kind"] for e in events])

    def test_progress_says_no_money(self):
        text = notice_text(book_events(None, self.status(["lean startup"], 1, 9.41)))
        self.assertNotIn("9.41", text)
        self.assertNotIn("replacement", text)


class DeliveryTest(unittest.TestCase):
    base = dict(agent_speaking=False, user_voice_at=0.0, agent_done_at=0.0, notice_at=0.0)

    def test_waits_for_a_pause(self):
        self.assertTrue(ready_to_speak(10, urgent=False, **self.base))
        self.assertFalse(ready_to_speak(10, urgent=False, **{**self.base, "agent_speaking": True}))
        self.assertFalse(ready_to_speak(10, urgent=False, **{**self.base, "user_voice_at": 9.0}))

    def test_urgent_notices_may_come_sooner(self):
        recent = {**self.base, "notice_at": 6.0}
        self.assertFalse(ready_to_speak(10, urgent=False, **recent))
        self.assertTrue(ready_to_speak(10, urgent=True, **recent))

    def test_progress_alone_is_silent_context(self):
        progress = [{"kind": "progress", "frames_analysed": 5, "bookshelves_spotted": 0, "frames_with_spines": 0, "objects_logged": []}]
        self.assertFalse(needs_reply(progress))
        self.assertIn("no need to reply", notice_text(progress))
        self.assertIn("Briefly tell", notice_text([{"kind": "blurry"}]))

    def test_labels(self):
        self.assertEqual(label("bookshelf", 2), "2 bookshelves")
        self.assertEqual(label("framed_art"), "painting or framed picture")




class CopiesQuestionTest(unittest.TestCase):
    def test_possible_copies_are_asked_about_once(self):
        from narration import book_events, describe, needs_reply
        status = {"books": [], "totals": {}, "possible_copies": [
            {"work_key": "/works/1", "title": "Zero to One", "author": "Peter Thiel", "spines_in_one_frame": 2}]}
        announced = set()
        first = book_events(None, status, announced)
        asked = [e for e in first if e["kind"] == "copies_question"]
        self.assertEqual(len(asked), 1)
        self.assertTrue(needs_reply(asked))
        self.assertIn("set_book_copies", describe(asked))
        self.assertEqual([e for e in book_events(status, status, announced) if e["kind"] == "copies_question"], [])


class ContentsTest(unittest.TestCase):
    def test_confirmed_objects_are_kept_with_count_and_score(self):
        tracker = SceneTracker()
        frame = {"width": 1000, "height": 600, "detections": [
            {"category": "chair", "score": 0.6, "box": [0, 0, 100, 100]}, {"category": "chair", "score": 0.5, "box": [200, 0, 300, 100]},
            {"category": "window", "score": 0.9, "box": [400, 0, 500, 100]}, {"category": "book", "score": 0.8, "box": [600, 0, 650, 100]}]}
        tracker.update(frame, 0.0)
        tracker.update(frame, 1.0)
        self.assertEqual(tracker.contents(), {"chair": {"count": 2, "score": 0.6}})


if __name__ == "__main__":
    unittest.main()
