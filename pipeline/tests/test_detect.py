import unittest

from pipeline.detect import _canonical, _touches_edge, resolve_overlaps


def det(category, box, score=0.5):
    return {"category": category, "box": box, "score": score}


class ResolveOverlapsTest(unittest.TestCase):
    def test_same_box_two_labels_keeps_higher_score(self):
        kept = resolve_overlaps([det("sofa", [0, 0, 100, 50], 0.6), det("chair", [2, 1, 101, 52], 0.4)])
        self.assertEqual([d["category"] for d in kept], ["sofa"])

    def test_shelf_board_inside_bookshelf_is_dropped(self):
        kept = resolve_overlaps([det("bookshelf", [0, 0, 100, 200], 0.5), det("table", [5, 90, 95, 110], 0.6)])
        self.assertEqual([d["category"] for d in kept], ["bookshelf"])

    def test_books_inside_bookshelf_are_kept(self):
        kept = resolve_overlaps([det("bookshelf", [0, 0, 100, 200]), det("book", [0, 0, 100, 195], 0.7)])
        self.assertEqual(sorted(d["category"] for d in kept), ["book", "bookshelf"])

    def test_separate_objects_are_kept(self):
        kept = resolve_overlaps([det("table", [0, 100, 100, 150]), det("chair", [120, 90, 160, 160])])
        self.assertEqual(len(kept), 2)


class HelpersTest(unittest.TestCase):
    def test_truncated_box(self):
        self.assertTrue(_touches_edge((0, 10, 50, 60), 480, 854))
        self.assertFalse(_touches_edge((20, 10, 50, 60), 480, 854))

    def test_merged_phrase_maps_to_category(self):
        self.assertEqual(_canonical("book spine"), "book")
        self.assertEqual(_canonical("picture frame painting"), "framed_art")


if __name__ == "__main__":
    unittest.main()
