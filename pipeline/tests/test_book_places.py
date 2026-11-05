import unittest

from pipeline.book_places import BookPlaces, place_key

TITLES = {"a": "Book A", "b": "Book B", "c": "Book C", "d": "Book D"}


class BookPlacesTest(unittest.TestCase):
    def setUp(self):
        self.m = BookPlaces()

    def test_new_repeat_extended(self):
        books = {"f1": {"a", "b"}, "f2": {"b", "c"}, "f3": {"a"}, "f4": {"d"}}
        first = self.m.record("books on the coffee table", ["f1", "f2"], books, TITLES)
        self.assertEqual((first["status"], first["new_books"]), ("new", ["Book A", "Book B", "Book C"]))
        self.assertEqual(self.m.record("Books on the coffee table", ["f3"], books, TITLES)["status"], "repeat")
        extended = self.m.record("books on the coffee table", ["f4"], books, TITLES)
        self.assertEqual((extended["status"], extended["new_books"]), ("extended", ["Book D"]))

    def test_rows_are_separate_places(self):
        self.assertNotEqual(place_key("window shelf, row 1"), place_key("window shelf, row 2"))
        self.assertEqual(place_key("Window shelf,  row 1"), place_key("window shelf row 1"))

    def test_same_books_at_another_place_are_flagged_not_double_counted(self):
        books = {"f1": {"a", "b", "c"}, "f9": {"a", "b", "c"}}
        self.m.record("window shelf, row 1", ["f1"], books, TITLES)
        again = self.m.record("book in my hand", ["f9"], books, TITLES)
        self.assertEqual(again["status"], "possible_repeat_of_other_place")
        self.assertEqual(again["looks_like"]["place"], "window shelf, row 1")
        self.assertEqual(again["books_here"], [])
        self.assertEqual(self.m.summary(TITLES)["books_at_places"], 3)

    def test_book_seen_from_two_places_is_counted_once(self):
        books = {"f1": {"a", "b"}, "f2": {"b", "c", "d"}}
        self.m.record("window shelf, row 1", ["f1"], books, TITLES)
        row2 = self.m.record("window shelf, row 2", ["f2"], books, TITLES)
        self.assertEqual(row2["books_here"], ["Book C", "Book D"])
        self.assertEqual(row2["already_counted_elsewhere"], ["Book B (window shelf, row 1)"])

    def test_unclaimed_frames_join_the_last_place(self):
        books = {"f1": {"a"}, "f2": {"b"}}
        self.m.record("book in my hand", ["f1"], books, TITLES)
        result = self.m.continue_last(["f2"], books, TITLES)
        self.assertEqual((result["place"], result["new_books"]), ("book in my hand", ["Book B"]))

    def test_unclaimed_frames_with_no_place_yet(self):
        result = self.m.continue_last(["f1"], {"f1": {"a"}}, TITLES)
        self.assertEqual(result["place"], "books not given a place")


if __name__ == "__main__":
    unittest.main()
