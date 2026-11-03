import unittest

import numpy as np

from pipeline import measure
from pipeline.measure import FrameResult, Spine


def rect_mask(x0, y0, x1, y1, shape=(1080, 1920)):
    mask = np.zeros(shape, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def spine(text, mask, orientation="top_to_bottom", book_id=None, depth=None, ref="f1"):
    length, thickness, centre, angle = measure.fit_rectangle(mask)
    return Spine(ref, text, orientation, length, thickness, centre, angle, measure.touches_edge(mask), 0.95, mask,
                 depth_m=depth, book_id=book_id, sides=measure.side_vectors(mask))


class GeometryTest(unittest.TestCase):
    def test_rectangle_fit_gives_long_and_short_side(self):
        length, thickness, _, _ = measure.fit_rectangle(rect_mask(100, 100, 160, 500))     # upright spine
        self.assertAlmostEqual(length, 400, delta=2)
        self.assertAlmostEqual(thickness, 60, delta=2)
        length, thickness, _, _ = measure.fit_rectangle(rect_mask(100, 100, 500, 160))     # the same book lying flat
        self.assertAlmostEqual(length, 400, delta=2)

    def test_card_scale_from_its_two_sides(self):
        self.assertAlmostEqual(measure.card_scale(171.2, 107.96), 20.0, places=2)         # 20 px per cm

    def test_per_axis_scale_undoes_vertical_foreshortening(self):
        # Upright card seen from above: 20 px/cm across, 18.8 px/cm vertically (6% shorter).
        sx, sy = measure.axis_scales((0.0, 8.56 * 18.8), (5.398 * 20.0, 0.0))
        self.assertAlmostEqual(sx, 20.0, places=3)
        self.assertAlmostEqual(sy, 18.8, places=3)
        upright = spine("x", rect_mask(100, 100, 160, 100 + round(21 * 18.8)))      # 21 cm tall, 3 cm thick
        result = measure.sizes(upright, (sx, sy))
        self.assertAlmostEqual(result["height_cm"], 21.0, delta=0.1)
        self.assertAlmostEqual(result["thickness_cm"], 3.0, delta=0.1)

    def test_card_turned_45_degrees_falls_back_to_one_scale(self):
        self.assertIsNone(measure.axis_scales((100.0, 100.0), (-60.0, 60.0)))

    def test_shape_that_is_not_a_card_is_rejected(self):
        self.assertIsNone(measure.card_scale(400, 60))                                      # a spine
        self.assertIsNone(measure.card_scale(100, 100))                                     # a square

    def test_model_scale_is_the_pinhole_model(self):
        # 2000 px focal length, 1 m away: 20 px per cm.
        self.assertAlmostEqual(measure.model_scale(1.0, 2000), 20.0)

    def test_spine_behind_the_card_gets_no_height(self):
        card = rect_mask(100, 400, 280, 510)                  # card held across the bottom of the spine
        hidden = spine("x", rect_mask(100, 100, 160, 400))     # outline stops where the card starts
        clear = spine("y", rect_mask(400, 100, 460, 500))
        beside = spine("z", rect_mask(290, 100, 340, 505))     # card stands next to it, not past its end
        measure.mark_behind_card([hidden, clear, beside], card)
        self.assertFalse(beside.behind_card)
        self.assertIsNone(measure.sizes(hidden, 20)["height_cm"])
        self.assertIsNotNone(measure.sizes(hidden, 20)["thickness_cm"])
        self.assertAlmostEqual(measure.sizes(clear, 20)["height_cm"], 20.0, delta=0.1)

    def test_cut_off_spine_gets_no_height(self):
        cut = spine("x", rect_mask(0, 100, 60, 500))
        self.assertIsNone(measure.sizes(cut, 20)["height_cm"])
        self.assertAlmostEqual(measure.sizes(cut, 20)["thickness_cm"], 3.0, delta=0.1)


class LinkTest(unittest.TestCase):
    def test_title_read_on_a_spine_links_that_spine(self):
        books = [{"id": "book-1", "title": "Spark: The Definitive Guide"}, {"id": "book-2", "title": "Zero to One"}]
        spark = spine("Spark: The Definitive Guide a", rect_mask(10, 10, 60, 400))
        junk = spine("PUSSSOUD", rect_mask(100, 10, 150, 400))
        measure.link_books([spark, junk], books)
        self.assertEqual((spark.book_id, junk.book_id), ("book-1", None))


class ScaleMethodTest(unittest.TestCase):
    def test_book_measured_with_card_is_a_ruler_in_another_frame(self):
        # Frame 1: card at 20 px/cm measures book-1 as 400 px / 20 = 20 cm.
        b1 = spine("ruler", rect_mask(100, 100, 160, 500), book_id="book-1", ref="f1")
        f1 = FrameResult("f1", [b1], {"px_per_cm": 20.0}, None,
                         [{"spine": b1, "method": "card", **measure.sizes(b1, 20.0)}])
        # Frame 2 (closer, no card): book-1 is 600 px tall, so 30 px/cm; the other spine is 450 px.
        r = spine("ruler", rect_mask(100, 100, 190, 700), book_id="book-1", ref="f2")
        other = spine("other", rect_mask(300, 100, 360, 550), ref="f2")
        f2 = FrameResult("f2", [r, other], None, None, [])
        measure._add_ruler_measurements([f1, f2])
        ruled = [m for m in f2.measurements if m["spine"] is other and m["method"] == "ruler"]
        self.assertAlmostEqual(ruled[0]["height_cm"], 15.0, delta=0.2)

    def test_card_is_preferred_and_model_kept_for_comparison(self):
        books = [{"id": "book-1", "title": "Zero to One"}]
        s = spine("Zero to One", rect_mask(100, 100, 150, 520), book_id="book-1")
        result = FrameResult("f1", [s], {"px_per_cm": 20.0}, 2000.0, [
            {"spine": s, "method": "card", **measure.sizes(s, 20.0)},
            {"spine": s, "method": "model", **measure.sizes(s, 17.0)}])
        line = measure._combine([result], books, use_model=True)["books"][0]
        self.assertEqual(line["scale_method"], "card")
        self.assertFalse(line["estimated"])
        self.assertAlmostEqual(line["spine_height_cm"], 21.0, delta=0.2)
        self.assertIn("model", line["by_method"])

    def test_model_only_is_labelled_estimated(self):
        books = [{"id": "book-1", "title": "Zero to One"}]
        s = spine("Zero to One", rect_mask(100, 100, 150, 520), book_id="book-1")
        result = FrameResult("f1", [s], None, 2000.0, [{"spine": s, "method": "model", **measure.sizes(s, 20.0)}])
        line = measure._combine([result], books, use_model=True)["books"][0]
        self.assertEqual((line["scale_method"], line["estimated"]), ("model", True))

    def test_size_outside_book_range_is_not_reported(self):
        books = [{"id": "book-1", "title": "Zero to One"}]
        s = spine("Zero to One", rect_mask(100, 100, 400, 520), book_id="book-1")     # 300 px thick: 15 cm at 20 px/cm
        result = FrameResult("f1", [s], {"px_per_cm": 20.0}, None, [{"spine": s, "method": "card", **measure.sizes(s, 20.0)}])
        line = measure._combine([result], books, use_model=False)["books"][0]
        self.assertIsNone(line["spine_thickness_cm"])

    def test_shelf_run_adds_identified_upright_books_only(self):
        books = [{"id": "book-1", "title": "Zero to One"}, {"id": "book-2", "title": "Working Backwards"}]
        up = spine("Zero to One", rect_mask(100, 100, 150, 520), book_id="book-1")                       # 2.5 cm
        flat = spine("Working Backwards", rect_mask(100, 600, 560, 660), "horizontal", book_id="book-2")
        unread = spine("PUSSSOUD", rect_mask(300, 100, 340, 520))
        ms = [{"spine": x, "method": "card", **measure.sizes(x, 20.0)} for x in (up, flat, unread)]
        report = measure._combine([FrameResult("f1", [up, flat, unread], {"px_per_cm": 20.0}, None, ms)], books, use_model=False)
        self.assertEqual(report["shelf_run_m"], 0.03)       # 2.5 cm rounded; the flat book and the unread spine excluded
        self.assertEqual(len(report["unidentified_spines"]), 1)




class ClosestThicknessTest(unittest.TestCase):
    def test_close_up_views_decide_thickness(self):
        far = spine("x", rect_mask(100, 100, 160, 500))      # 60 px: seen at ~20 px/cm, outline edges inflate it
        near = spine("x", rect_mask(100, 100, 220, 900))     # 120 px: a close-up at ~60 px/cm
        values = [{"spine": far, "thickness_cm": 3.0}, {"spine": far, "thickness_cm": 3.1},
                  {"spine": near, "thickness_cm": 2.0}]
        self.assertEqual(measure.closest_thickness(values), 2.0)


if __name__ == "__main__":
    unittest.main()
