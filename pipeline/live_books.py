"""The books of one live sweep: runs books_agent on each frame and keeps the sweep-wide list.

The live app keeps each sharp frame in which Grounding DINO detected a book and hands it to
the worker. One BookJob per sweep runs books_agent (pipeline/books_agent.py, an ADK
SequentialAgent: OCR -> catalogue search -> rule match -> model match + verify -> price) on each frame in
arrival order, and merges the results:

- accepted books: one line per work, merged across frames (frame_refs), counted and priced;
- suggested books: the model's pick was not supported by the words read; review only;
- unread text: confident readings that matched nothing; review only.

Prices are cached per work and redone only if the country changes. Results are written to
the sweep folder (spine_text.json, books_identified.json, books_priced.json).
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Callable

import cv2
from PIL import Image

from .books_agent import create_books_agent, run_books_agent
from .catalog import same_work, significant, tokens
from .common import write_json
from .frames import _thumb, is_duplicate
from .pricing import book_totals, ebay_prices, ebooks_provider, price_book
from .telemetry import Usage, current as current_usage, stage

logger = logging.getLogger(__name__)
BATCH = 6    # frames per books_agent run


class BookJob:
    def __init__(self, sweep_dir: Path, reader: Callable[[], Any], rates: Callable[[], Any], *,
                 threshold: float = 150.0, measurer: Callable[[], Any] | None = None):
        self.sweep_dir = Path(sweep_dir)
        self._reader, self._rates = reader, rates
        # Spine measurement (pipeline/measure.py) runs on each frame right after its OCR, in its own
        # queue, so reading and identification never wait for it.
        self._measurer = measurer
        self.measure_queue: asyncio.Queue = asyncio.Queue()
        self.measure_task: asyncio.Task | None = None
        self.frame_measures: dict[str, Any] = {}       # frame_ref -> measure.FrameResult
        self.measuring = False
        self.threshold = threshold
        self.queue: asyncio.Queue = asyncio.Queue()
        self.lines: dict[str, dict] = {}
        self.identified: list[dict] = []
        self.unidentified = 0
        self.last_result: dict = {"identified": [], "unidentified_readings": []}
        self.last_price_attempt = 0.0
        self.done_frames: set = set()           # frames read and identified
        self.aliases: dict[str, str] = {}       # skipped near-duplicate frame -> the frame it repeats
        self.frames_skipped = 0
        self._last_sent: tuple[str, object] | None = None   # (frame_ref, thumbnail) last sent to books_agent
        self.prices: dict[str, dict] = {}      # work_key -> price_book result
        self.country = self.currency = ""
        self.frames_received = self.frames_read = 0
        self.errors: list[str] = []
        self.busy = False
        self.updated_at = time.time()
        self.task: asyncio.Task | None = None
        self._agent = None
        self.registry: dict[str, dict] = {}      # work_key -> accepted book, merged across frames
        self.suggested: dict[str, dict] = {}     # work_key -> pick not supported by the OCR text
        self.declined: set[str] = set()          # suggestions the policyholder said are not theirs
        self.unread: dict[str, dict] = {}        # reading text -> {text, frame_refs, ocr_confidence}
        self.matched_by: dict[str, int] = {}     # which matcher decided (the model, or the rules fallback)
        # Books the policyholder asked to leave out (not theirs, borrowed, not to be covered): kept on
        # record, marked, and out of every total. A title not identified yet waits in pending_exclusions.
        self.usage = Usage()                     # latency per stage and outside requests, this sweep
        # Copies the policyholder confirmed (work_key -> count). A title read on two spines in one
        # frame is only flagged (possible_copies) until they say how many they have.
        self.copies: dict[str, int] = {}
        self.excluded: dict[str, dict] = {}      # work_key -> {title, author, reason, at}
        self.pending_exclusions: list[dict] = []  # {title, reason, at}

    def add(self, frame_ref: str, path: Path, country: str = "", currency: str = "") -> None:
        """Queue a frame for books_agent, unless it shows almost the same view as the last one sent.

        A near-duplicate (camera held still) would cost a full OCR pass and a model call for
        nothing new. It becomes an alias of the frame it repeats: it counts as done when that
        frame is done, and its books are that frame's books. Comparing with the last frame sent,
        not the previous frame, keeps a slow pan from being skipped step by step.
        """
        self.frames_received += 1
        thumb = self._thumb(Path(path))
        if thumb is not None and self._last_sent is not None and is_duplicate(self._last_sent[1], thumb):
            self.aliases[frame_ref] = self._last_sent[0]
            self.frames_skipped += 1
            return
        if thumb is not None:
            self._last_sent = (frame_ref, thumb)
        self.queue.put_nowait((frame_ref, Path(path), country, currency))
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self.run())

    @staticmethod
    def _thumb(path: Path):
        image = cv2.imread(str(path))
        return _thumb(image) if image is not None else None

    def _original(self, frame_ref: str) -> str:
        return self.aliases.get(frame_ref, frame_ref)

    async def set_locale(self, country: str, currency: str) -> None:
        """Country set or changed mid-sweep: price (or re-price) everything identified so far."""
        if (country, currency) == (self.country, self.currency):
            return
        self.country, self.currency, self.prices = country, currency, {}
        if self.currency and self.identified:
            token = current_usage.set(self.usage)
            try:
                await asyncio.to_thread(self._price_new)
            finally:
                current_usage.reset(token)
            self._write(self.last_result)
        self.updated_at = time.time()

    async def run(self) -> None:
        """Process everything queued, in batches: books_agent takes all frames waiting (up to
        BATCH), so search and the model call run once per batch and the job keeps up with the camera."""
        while not self.queue.empty():
            batch = []
            while not self.queue.empty() and len(batch) < BATCH:
                batch.append(self.queue.get_nowait())
            refs = [b[0] for b in batch]
            self.busy = True
            try:
                await self._process(batch)
            except Exception as exc:   # one bad batch must not stop the sweep
                logger.exception("Book job failed on %s", refs)
                self.errors = (self.errors + [f"{refs[0]}..{refs[-1]}: {type(exc).__name__}"])[-10:]
            finally:
                self.done_frames.update(refs)   # also on failure, so waiting callers are not held up
                self.busy = False
                self.updated_at = time.time()

    async def _process(self, batch: list[tuple]) -> None:
        current_usage.set(self.usage)        # this task's context: every step below counts toward this sweep
        with stage("books: whole batch"):
            await self._process_batch(batch)

    async def _process_batch(self, batch: list[tuple]) -> None:
        country, currency = batch[-1][2], batch[-1][3]
        if (country, currency) != (self.country, self.currency):
            self.country, self.currency, self.prices = country, currency, {}
        if self._agent is None:
            self._agent = create_books_agent(self._reader, self._price_one, known=self._known_docs)
        frames = [{"ref": ref, "path": str(path)} for ref, path, _, _ in batch]
        result = await run_books_agent(self._agent, frames, self.country, self.currency)
        self.lines.update(result.get("ocr_lines") or {})
        if self._measurer is not None:
            for frame in frames:
                if frame["ref"] in (result.get("ocr_lines") or {}):
                    self.measure_queue.put_nowait((frame["ref"], Path(frame["path"]), result["ocr_lines"][frame["ref"]]))
            if self.measure_task is None or self.measure_task.done():
                self.measure_task = asyncio.create_task(self._measure_queued())
        self.frames_read += len(batch)
        self._merge(result)
        if self.currency:
            await asyncio.to_thread(self._price_new)   # retries anything the agent could not price
        self._write(self.last_result)

    async def _measure_queued(self) -> None:
        """Measure each read frame: spine outlines, the card, depth. One frame failing loses only it."""
        current_usage.set(self.usage)
        try:
            measurer = await asyncio.to_thread(self._measurer)     # loads SAM and Depth Pro on first use
        except Exception as exc:
            logger.exception("Spine measurement unavailable")
            self.errors = (self.errors + [f"measurement: {type(exc).__name__}"])[-10:]
            return
        while not self.measure_queue.empty():
            ref, path, lines = self.measure_queue.get_nowait()
            self.measuring = True
            try:
                with stage("measure: whole frame"):
                    self.frame_measures[ref] = await asyncio.to_thread(measurer.measure_frame, ref, path, lines,
                                                                       self.sweep_dir / "measure")
            except Exception as exc:
                logger.exception("Measuring %s failed", ref)
                self.errors = (self.errors + [f"measuring {ref}: {type(exc).__name__}"])[-10:]
            finally:
                self.measuring = False

    def measurement(self) -> dict:
        """Sizes per book from the frames measured so far; written to books_measured.json."""
        from .measure import finish

        waiting = self.measure_queue.qsize() + int(self.measuring) + (self.queue.qsize() if self._measurer else 0)
        if self._measurer is None:
            return {"state": "off", "frames_measured": 0, "frames_waiting": 0}
        report = finish(list(self.frame_measures.values()), self.identified)
        if self.frame_measures:
            write_json(self.sweep_dir / "books_measured.json", report)
        return {"state": "running" if waiting or self.busy else "done", "frames_measured": len(self.frame_measures),
                "frames_waiting": waiting, "report": report}

    def _known_docs(self) -> list[dict]:
        """Books identified so far (left-out ones too), shaped like catalogue results, for books_agent
        to match new readings against before searching."""
        return [{"key": b["work_key"], "title": b["title"], "subtitle": b.get("subtitle", ""),
                 "author_name": [a.strip() for a in (b.get("author") or "").split(",") if a.strip()],
                 "first_publish_year": b.get("first_publish_year"), "published_year_seen": b.get("published_year_seen"),
                 "source": (b.get("catalogue") or {}).get("source"), "url": (b.get("catalogue") or {}).get("url"),
                 "retrieved_at": (b.get("catalogue") or {}).get("retrieved_at", "")}
                for b in list(self.registry.values())]

    def _price_one(self, book: dict, state: dict) -> dict | None:
        """PriceBooks step callback: price a newly accepted work once per sweep."""
        if not self.currency:
            return None
        if book["work_key"] not in self.prices or self.prices[book["work_key"]].get("retry"):
            try:
                with stage("pricing: one book"):
                    self.prices[book["work_key"]] = price_book(book, self.country, self.currency, self._rates(),
                                                               self.threshold, ebay_prices(), ebooks=ebooks_provider())
            except Exception as exc:
                self.errors = (self.errors + [f"pricing {book['title'][:40]}: {type(exc).__name__}"])[-10:]
                return None
        return self.prices[book["work_key"]]

    def _drop_title_fragments(self) -> None:
        """Drop a book whose title words are a strict subset of another identified book's, when its
        author was not read: "Machine Learning Systems" read off the spine of "Designing Machine
        Learning Systems" matches a different catalogue work of exactly that name."""
        words = {k: set(significant(tokens(b["title"]))) for k, b in self.registry.items()}
        for key, book in list(self.registry.items()):
            if not book.get("author_hit") and not book.get("confirmed_by_policyholder") and \
                    any(words[key] < words[other] for other in self.registry if other != key):
                del self.registry[key]
                self.prices.pop(key, None)

    def _merge(self, result: dict) -> None:
        """Add a batch's results to the session-wide lists; each line carries the frames it was seen in."""
        verified = result.get("verified") or {}
        by = (result.get("matches") or {}).get("by", "none")
        self.matched_by[by] = self.matched_by.get(by, 0) + 1
        for line in verified.get("accepted", []):
            # The same book under another catalogue entry (Open Library vs Google Books, an edition
            # or "by Author" suffix) joins the first entry.
            key = next((k for k, b in self.registry.items() if k == line["work_key"]
                        or same_work(b["title"], b["author"], line["title"], line["author"])), line["work_key"])
            book = self.registry.setdefault(key, {**line, "frame_ref": line["frame_refs"][0], "frame_refs": set(),
                                                  "id_confidence": line["title_coverage"]})
            book["frame_refs"].update(line["frame_refs"])
            self.suggested.pop(line["work_key"], None)
        self._drop_title_fragments()
        self._apply_pending_exclusions()
        for line in verified.get("suggested", []):
            if line["work_key"] not in self.registry and line["work_key"] not in self.declined and \
                    not any(same_work(b["title"], b["author"], line["title"], line["author"])
                                                                 for b in self.registry.values()):
                self.suggested.setdefault(line["work_key"], {**line, "frame_refs": set()})["frame_refs"].update(line["frame_refs"])
        for item in verified.get("unread", []):
            entry = self.unread.setdefault(item["text"], {"text": item["text"], "frame_refs": set(), "ocr_confidence": item["confidence"]})
            entry["frame_refs"].update(item["frame_refs"])
        self._refresh()

    def _refresh(self) -> None:
        self.identified = [{**b, "id": f"book-{i:03d}", "frame_refs": sorted(b["frame_refs"])}
                           for i, b in enumerate(self.registry.values(), 1)]
        unread = [{**u, "frame_refs": sorted(u["frame_refs"])} for u in self.unread.values()]
        self.unidentified = len(unread)
        self.last_result = {"identified": self.identified, "unidentified_readings": unread,
                            "suggested": [{**b, "frame_refs": sorted(b["frame_refs"])} for b in self.suggested.values()],
                            "matched_by": self.matched_by, "method": "books_agent (ADK SequentialAgent)"}

    def confirm(self, title: str, confirmed: bool) -> dict:
        """The policyholder's answer about a suggested book (the model's pick that the words read did
        not support). Yes: it joins the inventory, marked as confirmed by them. No: it is dropped and
        not suggested again."""
        scored = sorted(((self._title_match(title, b), b) for b in self.suggested.values()), key=lambda x: -x[0])
        matches = [b for score, b in scored if score >= 80]
        if len(matches) > 1 and scored[0][0] - scored[1][0] >= 10:
            matches = matches[:1]
        if len(matches) != 1:
            return {"status": "ambiguous" if matches else "not_found",
                    "candidates": [f"{b['title']} by {b['author']}" for b in (matches or list(self.suggested.values()))[:6]]}
        book = self.suggested.pop(matches[0]["work_key"])
        if not confirmed:
            self.declined.add(book["work_key"])
        else:
            self.registry[book["work_key"]] = {**book, "frame_ref": sorted(book["frame_refs"])[0], "frame_refs": set(book["frame_refs"]),
                                               "id_confidence": book.get("title_coverage"), "confirmed_by_policyholder": True}
        self._refresh()
        self.updated_at = time.time()
        return {"status": "added" if confirmed else "declined", "title": book["title"], "author": book["author"]}

    def to_confirm(self, frame_refs: list[str] | None = None) -> list[str]:
        """Suggested books to ask the policyholder about, optionally only those read on these frames."""
        wanted = {self._original(f) for f in frame_refs} if frame_refs is not None else None
        return [f"{b['title']} by {b['author']}".strip() for b in self.suggested.values()
                if wanted is None or {self._original(f) for f in b["frame_refs"]} & wanted][:5]

    def _price_new(self) -> None:
        """Price every identified work without a price yet. A failure leaves that one book
        unpriced for a later retry; it never stops the rest of the sweep."""
        self.last_price_attempt = time.time()
        rates = self._rates()
        for book in self.identified:
            if book["work_key"] in self.prices and not self.prices[book["work_key"]].get("retry"):
                continue
            try:
                self.prices[book["work_key"]] = price_book(book, self.country, self.currency, rates, self.threshold, ebay_prices(),
                                                           ebooks=ebooks_provider())
            except Exception as exc:
                self.errors = (self.errors + [f"pricing {book['title'][:40]}: {type(exc).__name__}"])[-10:]

    def maybe_retry(self, every_s: float = 20.0) -> None:
        """When idle, retry books whose price lookup failed (eBay unreachable)."""
        missing = self.currency and any(b["work_key"] not in self.prices or self.prices[b["work_key"]].get("retry")
                                        for b in self.identified)
        idle = (self.task is None or self.task.done()) and not self.busy
        if missing and idle and time.time() - self.last_price_attempt >= every_s:
            self.task = asyncio.create_task(self._retry())

    async def _retry(self) -> None:
        current_usage.set(self.usage)
        await asyncio.to_thread(self._price_new)
        self._write(self.last_result)
        self.updated_at = time.time()

    # ---- leaving books out ------------------------------------------------------------------
    @staticmethod
    def _title_match(said: str, book: dict) -> int:
        from rapidfuzz import fuzz
        return max(fuzz.token_set_ratio(said.lower(), book["title"].lower()),
                   fuzz.token_set_ratio(said.lower(), f"{book['title']} {book['author']}".lower()))

    def exclude(self, title: str = "", frame_refs: list[str] | None = None, *, undo: bool = False, reason: str = "") -> dict:
        """Leave a book out of the report and totals (or put it back), by the title said, else the
        book on the given frames. Never guesses: two candidates come back for the agent to ask."""
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        pool = [b for b in self.registry.values() if (b["work_key"] in self.excluded) == undo]
        if title:
            scored = sorted(((self._title_match(title, b), b) for b in pool), key=lambda x: -x[0])
            matches = [b for score, b in scored if score >= 80]
            if len(matches) > 1 and scored[0][0] - scored[1][0] >= 10:
                matches = matches[:1]
        else:
            by_frame, _ = self.books_by_frame(include_excluded=True)
            keys = set().union(*(by_frame.get(self._original(f), set()) for f in frame_refs or []))
            matches = [b for b in pool if b["work_key"] in keys]
        if len(matches) > 1:
            return {"status": "ambiguous", "candidates": [f"{b['title']} by {b['author']}" for b in matches[:6]],
                    "ask": "Ask which of these they mean, then call again with its title."}
        if not matches:
            if undo:
                before = len(self.pending_exclusions)
                self.pending_exclusions = [p for p in self.pending_exclusions if self._title_match(title, {"title": p["title"], "author": ""}) < 80]
                if title and len(self.pending_exclusions) < before:
                    return {"status": "restored", "title": title}
                return {"status": "not_found", "excluded_books": [b["title"] for b in self.excluded.values()]}
            if title:      # not identified (yet): left out if it is identified later
                self.pending_exclusions.append({"title": title, "reason": reason, "at": stamp})
                return {"status": "pending", "title": title,
                        "note": "Not among the books identified so far; it will be left out if it is identified later."}
            return {"status": "not_found", "note": "No identified book on the frames in view. Ask for the title."}
        book = matches[0]
        if undo:
            self.excluded.pop(book["work_key"], None)
        else:
            self.excluded[book["work_key"]] = {"title": book["title"], "author": book["author"], "reason": reason, "at": stamp}
        self.updated_at = time.time()
        self._write(self.last_result)
        return {"status": "restored" if undo else "excluded", "title": book["title"], "author": book["author"],
                "books_excluded": len(self.excluded)}

    def set_copies(self, title: str, copies: int) -> dict:
        """The policyholder's count of copies of one identified book."""
        if not 1 <= copies <= 50:
            return {"status": "invalid", "note": "Copies must be between 1 and 50."}
        scored = sorted(((self._title_match(title, b), b) for b in self.registry.values()), key=lambda x: -x[0])
        matches = [b for score, b in scored if score >= 80]
        if len(matches) > 1 and scored[0][0] - scored[1][0] >= 10:
            matches = matches[:1]
        if len(matches) != 1:
            return {"status": "ambiguous" if matches else "not_found",
                    "candidates": [f"{b['title']} by {b['author']}" for b in matches[:6]]}
        book = matches[0]
        self.copies[book["work_key"]] = copies
        self.updated_at = time.time()
        self._write(self.last_result)
        return {"status": "set", "title": book["title"], "copies": copies}

    def possible_copies(self) -> list[dict]:
        """Books whose title was read on two or more separate spines in one frame, not yet confirmed.
        Each OCR strip is one spine, so two strips carrying the title are likely two copies."""
        from .catalog import _found, _with_joins
        from .spine_ocr import group_strips

        out = []
        for book in self.identified:
            key = book["work_key"]
            if key in self.copies or key in self.excluded:
                continue
            words = significant(tokens(book["title"].split(":")[0]))
            if len(words) < 2:
                continue
            most = 0
            for ref in book["frame_refs"]:
                spines = 0
                for orientation, lines in (self.lines.get(ref) or {}).items():
                    for strip in group_strips([{**l, "box": tuple(l["box"])} for l in lines if l.get("confidence", 1) >= 0.5]):
                        pool = _with_joins(tokens(" ".join(l["text"] for l in strip)))
                        spines += sum(_found(w, pool) for w in words) / len(words) >= 0.75
                most = max(most, spines)
            if most >= 2:
                out.append({"work_key": key, "title": book["title"], "author": book["author"], "spines_in_one_frame": most})
        return out

    def _apply_pending_exclusions(self) -> None:
        for pending in self.pending_exclusions[:]:
            for book in self.registry.values():
                if book["work_key"] not in self.excluded and self._title_match(pending["title"], book) >= 80:
                    self.excluded[book["work_key"]] = {"title": book["title"], "author": book["author"],
                                                       "reason": pending["reason"], "at": pending["at"]}
                    self.pending_exclusions.remove(pending)
                    break

    async def wait_for(self, frame_refs: list[str], timeout: float = 25.0) -> bool:
        """Wait until these frames are read and identified; False if the timeout passed first."""
        deadline = time.time() + timeout
        while not {self._original(f) for f in frame_refs} <= self.done_frames:
            if time.time() > deadline:
                return False
            await asyncio.sleep(0.25)
        return True

    def books_by_frame(self, include_excluded: bool = False) -> tuple[dict[str, set], dict[str, str]]:
        """Work keys seen on each frame (skipped near-duplicates share their original's), and titles.
        Books the policyholder left out are not listed unless include_excluded."""
        by_frame: dict[str, set] = {}
        titles = {}
        for book in self.identified:
            if book["work_key"] in self.excluded and not include_excluded:
                continue
            titles[book["work_key"]] = f"{book['title']} by {book['author']}".strip() if book["author"] else book["title"]
            for ref in book["frame_refs"]:
                by_frame.setdefault(ref, set()).add(book["work_key"])
        for alias, original in self.aliases.items():
            if original in by_frame:
                by_frame[alias] = by_frame[original]
        return by_frame, titles

    def unread_on(self, frame_refs: list[str], min_confidence: float = 0.95) -> list[str]:
        """Confidently read spine text on these frames that matched no catalogue entry.

        Readings below min_confidence are mostly mirrored text from reading a frame in the wrong
        rotation; reporting them would make the agent ask for needless re-films.
        """
        refs = {self._original(f) for f in frame_refs}
        return [u["text"] for u in self.last_result.get("unidentified_readings", [])
                if refs & set(u["frame_refs"]) and u.get("ocr_confidence", 0) >= min_confidence]

    def lines_out(self) -> list[dict]:
        out = []
        for book in self.identified:
            price = self.prices.get(book["work_key"])
            out.append({"id": book["id"], "work_key": book["work_key"], "title": book["title"], "author": book["author"],
                        "frame_ref": book["frame_ref"], "frame_refs": book["frame_refs"], "id_confidence": book["id_confidence"],
                        **(price or {"status": "identified", "replacement_cost": None, "used_value": None}),
                        "excluded": self.excluded.get(book["work_key"]),
                        "copies": self.copies.get(book["work_key"], 1), "copies_confirmed": book["work_key"] in self.copies,
                        "confirmed_by_policyholder": bool(book.get("confirmed_by_policyholder"))})
        return out

    def totals(self, books: list[dict] | None = None) -> dict:
        """Totals of the books still in the inventory; left-out books are counted separately."""
        books = self.lines_out() if books is None else books
        kept = [b for b in books if not b["excluded"]]
        totals = book_totals(kept) if kept else {"books_identified": 0, "books_priced": 0, "books_replacement_cost": 0.0}
        return {**totals, "books_excluded": len(books) - len(kept)}

    def _write(self, identified: dict) -> None:
        write_json(self.sweep_dir / "spine_text.json", {"model": "docTR, rotated boxes, four orientations (live)", "lines": self.lines})
        write_json(self.sweep_dir / "books_identified.json", identified)
        books = self.lines_out()
        write_json(self.sweep_dir / "books_priced.json", {"country": self.country, "currency": self.currency,
                                                          "totals": self.totals(books), "books": books})

    def status(self) -> dict:
        books = self.lines_out()
        return {
            "frames_received": self.frames_received, "frames_read": self.frames_read, "queued": self.queue.qsize(),
            "frames_skipped_duplicate": self.frames_skipped,
            "busy": self.busy, "country": self.country, "currency": self.currency,
            "waiting_for_country": not self.currency, "unidentified_readings": self.unidentified,
            "totals": self.totals(books),
            "books": [{"work_key": b["work_key"], "title": b["title"], "author": b["author"], "status": b["status"],
                       "excluded": bool(b["excluded"]), "copies": b["copies"],
                       "amount": (b.get("replacement_cost") or {}).get("amount"), "converted": bool((b.get("replacement_cost") or {}).get("converted")),
                       "used_amount": (b.get("used_value") or {}).get("amount"),
                       "frame_ref": b["frame_ref"]} for b in books],
            "suggested": [{"title": b["title"], "author": b["author"], "reading": b["reading"]} for b in self.suggested.values()],
            "possible_copies": self.possible_copies(),
            "matched_by": self.matched_by, "errors": self.errors, "updated_at": self.updated_at,
            "frames_measured": len(self.frame_measures), "frames_waiting_measurement": self.measure_queue.qsize() + int(self.measuring),
        }
