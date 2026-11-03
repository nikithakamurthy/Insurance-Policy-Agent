"""Turn per-frame detections into events the voice agent should hear about.

The vision worker sees about one camera frame per second. This tracker keeps a short memory
of the scene and reports only what matters to the conversation:

- new_objects: a category seen in at least 2 of the last 4 frames that was not reported yet
  (a single-frame detection is often a false positive), with how many and where in view;
- bookshelf: a bookshelf first comes into view (so the agent can direct the shelf capture);
- spines_small: spines are in view but too small to read, so the user should step closer;
- blurry: several frames in a row are much blurrier than usual;
- progress: a periodic summary of what has been logged.

Counts are the most seen in any single frame, a lower bound with no cross-frame tracking.
"""

from __future__ import annotations

import statistics
from collections import Counter, deque
from dataclasses import dataclass, field

STRUCTURAL = {"window", "door"}
LABELS = {"framed_art": "painting or framed picture", "electronics": "electronic device", "appliance": "appliance",
          "decor": "decor item", "cabinet": "cabinet", "bookshelf": "bookshelf"}


def label(category: str, count: int = 1) -> str:
    name = LABELS.get(category, category.replace("_", " "))
    if count == 1:
        return name
    plural = name[:-1] + "ves" if name.endswith("shelf") else name + ("es" if name.endswith(("ch", "sh", "s")) else "s")
    return f"{count} {plural}"


def position(box: list[float], width: int) -> str:
    centre = (box[0] + box[2]) / 2 / max(1, width)
    return "left" if centre < 0.33 else "right" if centre > 0.67 else "centre"


@dataclass
class SceneTracker:
    confirm_frames: int = 2
    window: int = 4
    small_spine_height: float = 0.12   # spine boxes shorter than this share of the frame are unreadable
    progress_every_s: float = 25.0
    recent: deque = field(default_factory=lambda: deque(maxlen=4))
    reported: dict = field(default_factory=dict)        # category -> highest count reported
    max_seen: Counter = field(default_factory=Counter)  # category -> most seen in one frame
    best_score: dict = field(default_factory=dict)      # confirmed category -> highest detector score
    sharpness: deque = field(default_factory=lambda: deque(maxlen=30))
    frames: int = 0
    bookshelves: int = 0
    spine_frames: int = 0
    last_progress: float = 0.0
    last_small_spines: float = -1e9
    last_blur: float = -1e9

    def update(self, result: dict, now: float) -> list[dict]:
        """Feed one vision-worker result; return the events worth telling the agent."""
        self.frames += 1
        width, height = result.get("width", 1), result.get("height", 1)
        dets = [d for d in result.get("detections", []) if d.get("score", 0) >= 0.35]
        counts = Counter(d["category"] for d in dets)
        self.recent.append(counts)
        for category, n in counts.items():
            self.max_seen[category] = max(self.max_seen[category], n)
        events: list[dict] = []

        confirmed = {c for c in counts if sum(1 for f in self.recent if f.get(c)) >= self.confirm_frames}
        new = []
        for category in confirmed:
            score = max(d["score"] for d in dets if d["category"] == category)
            self.best_score[category] = max(self.best_score.get(category, 0.0), score)
        for category in sorted(confirmed - {"book"} - STRUCTURAL):
            n = max(f.get(category, 0) for f in self.recent)
            first = category not in self.reported
            if first or n > self.reported[category]:
                # Only a first sighting is announced. A higher count later is usually the same
                # objects seen from another angle, so it updates the progress summary silently,
                # except for bookshelves, where a second unit changes the capture plan.
                box = max((d for d in dets if d["category"] == category), key=lambda d: d["score"])["box"]
                if category == "bookshelf":
                    self.bookshelves = max(self.bookshelves, n)
                    events.append({"kind": "bookshelf", "count": n, "where": position(box, width), "first": first})
                elif first:
                    new.append({"category": category, "count": n, "label": label(category, n), "where": position(box, width)})
                self.reported[category] = n
        if new:
            events.append({"kind": "new_objects", "objects": new})

        books = [d for d in dets if d["category"] == "book"]
        if len(books) >= 3:
            self.spine_frames += 1
            tallest = max((d["box"][3] - d["box"][1]) / height for d in books)
            if tallest < self.small_spine_height and now - self.last_small_spines > 20:
                self.last_small_spines = now
                events.append({"kind": "spines_small", "books": len(books)})

        value = result.get("sharpness")
        if value is not None:
            self.sharpness.append(float(value))
            if len(self.sharpness) >= 8:
                typical = statistics.median(self.sharpness)
                if all(v < 0.4 * typical for v in list(self.sharpness)[-3:]) and now - self.last_blur > 15:
                    self.last_blur = now
                    events.append({"kind": "blurry"})

        if now - self.last_progress >= self.progress_every_s and self.frames >= 5:
            self.last_progress = now
            events.append({"kind": "progress", **self.summary()})
        return events

    def contents(self) -> dict[str, dict]:
        """Everything the detector confirmed (seen in at least two of a few frames), books and
        windows or doors left out: the most seen in one frame, and the best score."""
        return {c: {"count": n, "score": round(self.best_score.get(c, 0.0), 2)}
                for c, n in self.reported.items() if c != "book" and c not in STRUCTURAL}

    def worth_reading(self, result: dict) -> bool:
        """Send this frame to books_agent: Grounding DINO found a book in it, and it is not blurry.

        One book is enough (held up, on a table, in a box or on a shelf). Blurry frames cannot
        be read, so they are skipped.
        """
        if not any(d["category"] == "book" and d.get("score", 0) >= 0.3 for d in result.get("detections", [])):
            return False
        value = result.get("sharpness")
        if value is None or len(self.sharpness) < 5:
            return True
        return float(value) >= 0.5 * statistics.median(self.sharpness)

    def summary(self) -> dict:
        logged = [label(c, n) for c, n in sorted(self.reported.items()) if c != "bookshelf"]
        return {"frames_analysed": self.frames, "bookshelves_spotted": self.bookshelves,
                "frames_with_spines": self.spine_frames, "objects_logged": logged}


PROMPT_KINDS = {"bookshelf", "spines_small", "blurry"}   # instructions the user should hear soon
# Context for the agent, no reply needed. Titles are read out from analyse_shelf results, so
# the background book notices only keep the agent informed.
SILENT_KINDS = {"progress", "books_progress"}     # newly identified books are spoken (once each)


def needs_reply(events: list[dict]) -> bool:
    """Progress alone is context for the agent; anything else asks it to speak."""
    return any(e["kind"] not in SILENT_KINDS for e in events)


def book_events(previous: dict | None, status: dict, announced: set | None = None) -> list[dict]:
    """Events from two book-agent status snapshots: newly identified titles, and running totals.

    Pass the same `announced` set on every call: identification is recomputed as frames
    arrive, so a title can drop out of one snapshot and return in the next; it is announced once.
    """
    events = []
    before = announced if announced is not None else {b["work_key"] for b in (previous or {}).get("books", [])}
    new = [b for b in status.get("books", []) if b["work_key"] not in before]
    if announced is not None:
        announced.update(b["work_key"] for b in new)
    if new:
        events.append({"kind": "books_identified", "titles": [f"{b['title']} by {b['author']}" if b["author"] else b["title"] for b in new]})
    for copy in status.get("possible_copies", []) if announced is not None else []:
        key = "copies:" + copy["work_key"]
        if key not in announced:          # asked once per title
            announced.add(key)
            events.append({"kind": "copies_question", "title": copy["title"], "author": copy["author"],
                           "spines": copy["spines_in_one_frame"]})
    keys = ("frames_read", "queued", "currency")
    totals_changed = (previous or {}).get("totals") != status.get("totals")
    if previous is None or totals_changed or any((previous or {}).get(k) != status.get(k) for k in keys):
        t = status.get("totals", {})
        # Counts only: prices and sizes are worked out in the background and go to the report.
        events.append({"kind": "books_progress", "frames_read": status.get("frames_read", 0), "queued": status.get("queued", 0),
                       "identified": t.get("books_identified", 0)})
    return events


def ready_to_speak(now: float, *, agent_speaking: bool, user_voice_at: float, agent_done_at: float,
                   notice_at: float, urgent: bool) -> bool:
    """Deliver a spoken notice only in a pause: nobody talking, and not too soon after the last one."""
    gap = 3.0 if urgent else 7.0
    return (not agent_speaking and now - user_voice_at >= 1.5 and now - agent_done_at >= 1.0
            and now - notice_at >= gap)


def notice_text(events: list[dict]) -> str:
    """One APP VISION notice for a batch of events, phrased for the voice agent."""
    ask = (" Briefly tell the policyholder what you are logging or what to do next, and ask a short clarifying "
           "question only if it helps. Do not repeat things you already said." if needs_reply(events) else
           " For your awareness only; no need to reply now.")
    return f"APP VISION (from the team's object detector, not the policyholder): {describe(events)}{ask} This notice is app state."


def describe(events: list[dict]) -> str:
    """Plain description of a batch of events, also used as the activity-feed headline."""
    lines = []
    for e in events:
        if e["kind"] == "new_objects":
            lines.append("Newly in view and logged by the team: " + ", ".join(f"{o['label']} ({o['where']})" for o in e["objects"]) + ".")
        elif e["kind"] == "bookshelf":
            what = "A bookshelf is" if e["count"] == 1 else f"{e['count']} bookshelves are"
            lines.append(f"{what} in view ({e['where']} of the frame)" + (", the first one spotted." if e["first"] else "."))
        elif e["kind"] == "spines_small":
            lines.append(f"{e['books']} book spines are in view but too small to read; the policyholder should step closer.")
        elif e["kind"] == "blurry":
            lines.append("The last few frames are blurry; ask them to slow down and hold steady.")
        elif e["kind"] == "books_identified":
            lines.append("Book team identified (new, not mentioned before): " + "; ".join(e["titles"]) + ".")
        elif e["kind"] == "books_progress":
            lines.append(f"Book team progress: {e['frames_read']} spine frames read, {e['queued']} waiting; "
                         f"{e['identified']} books identified so far.")
        elif e["kind"] == "copies_question":
            lines.append(f"The book team read {e['title']} on {e['spines']} separate spines in one frame; it is counted once. "
                         "Ask how many copies they have, then call set_book_copies with their answer.")
        elif e["kind"] == "progress":
            logged = ", ".join(e["objects_logged"]) or "nothing yet"
            lines.append(f"Progress: {e['frames_analysed']} frames analysed; logged so far: {logged}; "
                         f"bookshelves spotted: {e['bookshelves_spotted']}; frames showing spines: {e['frames_with_spines']}.")
    return " ".join(lines)
