"""Book memory per place: which books were seen where during the session.

The voice agent calls analyse_books after books have been shown somewhere, with its own short
description of the place ("bookshelf by the window, row 2", "books on the coffee table",
"book in my hand"). The frames kept since the previous call come with it. The memory answers
what the voice agent cannot tell from low-resolution video:

1. Which books were found at this place, and which of them are new to the session.
2. Is this place new, shown again with nothing new (repeat), or shown again with more books
   (extended)?
3. Do these books look like a place already recorded (most of the titles in common)? Then the
   voice agent asks whether it was filmed again or is a second copy, instead of counting twice.

A book stays at the place where it was first recorded; seeing it again from another place
(frames overlapping two shelf rows, or a re-film) never counts it twice.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

SAME_PLACE_OVERLAP = 0.6     # share of titles two places must have in common to be "the same books"


def place_key(label: str) -> str:
    """Places are matched on their normalised wording: 'Row 1' and 'row 1 ' are one place,
    'row 1' and 'row 2' are not (fuzzy matching would merge them)."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", label.lower())).strip()


def overlap(a: set, b: set) -> float:
    return len(a & b) / min(len(a), len(b)) if a and b else 0.0


@dataclass
class Place:
    key: str
    label: str
    frame_refs: set = field(default_factory=set)
    work_keys: set = field(default_factory=set)
    stated_count: int | None = None      # how many books the policyholder says are here


class BookPlaces:
    def __init__(self):
        self.places: dict[str, Place] = {}
        self.last: Place | None = None

    def record(self, label: str, frame_refs: list[str], books_by_frame: dict[str, set], titles: dict[str, str]) -> dict:
        label = " ".join(label.split()) or "books not given a place"
        key = place_key(label)
        keys = set().union(*(books_by_frame.get(f, set()) for f in frame_refs)) if frame_refs else set()
        recorded_anywhere = {k for p in self.places.values() for k in p.work_keys}
        place = self.places.get(key)
        if place is None:
            place = self.places[key] = Place(key=key, label=label)
            status = "new"
        else:
            status = "repeat" if not (keys - place.work_keys) else "extended"

        looks_like = None
        for other in self.places.values():
            if other is not place and len(keys & other.work_keys) >= 2 and overlap(keys, other.work_keys) >= SAME_PLACE_OVERLAP:
                looks_like = {"place": other.label, "shared_titles": sorted(titles.get(k, k) for k in keys & other.work_keys)}
        if looks_like and status == "new":
            status = "possible_repeat_of_other_place"

        elsewhere = {k: other.label for other in self.places.values() if other is not place for k in other.work_keys if k in keys}
        new_here = keys - set(elsewhere)
        place.frame_refs.update(frame_refs)
        place.work_keys.update(new_here)
        self.last = place
        return {"place": place.label, "status": status,
                "books_here": sorted(titles.get(k, k) for k in place.work_keys),
                "new_books": sorted(titles.get(k, k) for k in new_here - recorded_anywhere),
                "already_counted_elsewhere": sorted(f"{titles.get(k, k)} ({where})" for k, where in elsewhere.items()),
                "looks_like": looks_like, "frames_used": len(frame_refs)}

    def continue_last(self, frame_refs: list[str], books_by_frame: dict[str, set], titles: dict[str, str]) -> dict:
        """Frames shown after the last analysis (for example at the end) belong to the last place."""
        return self.record(self.last.label if self.last else "books not given a place", frame_refs, books_by_frame, titles)

    def set_count(self, label: str, count: int) -> dict:
        """The policyholder's own count of the books at a place (a rough count is fine). Books the team
        could not identify are the difference, so the claim's book count covers unreadable spines too."""
        label = " ".join(label.split()) or (self.last.label if self.last else "books not given a place")
        key = place_key(label)
        place = self.places.get(key) or self.places.setdefault(key, Place(key=key, label=label))
        place.stated_count = count
        return {"place": place.label, "stated_count": count, "identified_here": len(place.work_keys)}

    def summary(self, titles: dict[str, str]) -> dict:
        return {"places": [{"place": p.label, "books": sorted(titles.get(k, k) for k in p.work_keys), "stated_count": p.stated_count}
                           for p in self.places.values()],
                "books_at_places": sum(len(p.work_keys) for p in self.places.values())}
