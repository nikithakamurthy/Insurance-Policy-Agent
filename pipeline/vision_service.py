"""Live worker for the voice agent: object detection per frame, and the book agent per sweep.

    .venv/bin/python -m uvicorn pipeline.vision_service:app --host 127.0.0.1 --port 4180

POST /detect                    one JPEG frame (raw body) -> objects in view, positions, sharpness.
                                Detection runs on a copy at most 640 px wide; boxes are returned
                                in the original frame's pixels.
POST /books/{sweep}/frames      queue a saved frame with a book in it for books_agent.
POST /books/{sweep}/analyse     the books in the frames just shown, recorded under a place.
GET  /books/{sweep}/summary     everything found so far, per place (final=true waits for reading).
GET  /books/{sweep}/status      progress and results.
GET  /books/{sweep}/measure     spine sizes so far; each frame is measured right after its OCR.
POST /books/{sweep}/exclude     leave a book out of the report and totals (by title or the book in view), or undo.
POST /books/{sweep}/copies      how many copies of a book the policyholder has.
POST /books/{sweep}/count       how many books the policyholder says are at a place.
GET  /books/{sweep}/report      books, prices, exclusions, totals and measurements for the session report.
GET  /books/{sweep}/usage       latency per stage and outside requests and tokens for the sweep.
POST /room/{sweep}              measure the room (VGGT, bank card scale) from the room frames, in the background.
GET  /room/{sweep}              room measurement state and result.

Detection uses the GPU and answers in about 0.2 s; the book agent's OCR runs on the CPU in the
background, so live narration is not held up by reading spines.
"""

from __future__ import annotations

import asyncio
import io
import re
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from PIL import Image
from pydantic import BaseModel

from .common import pick_device
from .detect import Detector
from .frames import sharpness
from .live_books import BookJob

app = FastAPI(title="Sweep live worker")
MAX_BYTES = 6 * 1024 * 1024
DETECT_WIDTH = 640
SWEEPS = (Path(__file__).resolve().parent.parent / "sweeps").resolve()
_state: dict = {"jobs": {}}


def _detector() -> Detector:
    if "detector" not in _state:
        _state["detector"] = Detector(pick_device())
        _state["lock"] = asyncio.Lock()
    return _state["detector"]


def _reader():
    if "reader" not in _state:
        from .spine_ocr import SpineReader
        _state["reader"] = SpineReader()
    return _state["reader"]


def _rates():
    # Reference rates change daily; reload at most every 6 hours.
    if "rates" not in _state or time.time() - _state["rates_at"] > 6 * 3600:
        from .fx import load_rates
        _state["rates"], _state["rates_at"] = load_rates(), time.time()
    return _state["rates"]


def _gemini_key() -> None:
    """books_agent's matching step calls Gemini through ADK, which reads GOOGLE_API_KEY."""
    import os

    from .common import load_api_key
    try:
        os.environ.setdefault("GOOGLE_API_KEY", load_api_key())
        os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "False")
    except RuntimeError:
        pass   # no key: books_agent falls back to rule-based matching


@app.on_event("startup")
async def warm_up() -> None:
    _gemini_key()
    # Loading takes several seconds each; do it before the first frame arrives.
    await asyncio.to_thread(_detector)
    asyncio.get_running_loop().run_in_executor(None, _reader)


@app.get("/health")
def health() -> dict:
    return {"ok": "detector" in _state, "model": _state["detector"].model_id if "detector" in _state else None,
            "ocr_ready": "reader" in _state}


@app.post("/detect")
async def detect(request: Request) -> dict:
    data = await request.body()
    if not data.startswith(b"\xff\xd8\xff") or len(data) > MAX_BYTES:
        raise HTTPException(422, "Send one JPEG frame as the request body")
    detector = _detector()
    started = time.perf_counter()
    image = Image.open(io.BytesIO(data)).convert("RGB")
    scale = min(1.0, DETECT_WIDTH / image.width)
    small = image if scale == 1.0 else image.resize((DETECT_WIDTH, round(image.height * scale)))
    bgr = cv2.cvtColor(np.asarray(small), cv2.COLOR_RGB2BGR)
    async with _state["lock"]:   # one frame at a time on the GPU
        detections = await asyncio.to_thread(detector.detect, small)
    for d in detections:
        d["box"] = [round(v / scale, 1) for v in d["box"]]
    return {"width": image.width, "height": image.height, "sharpness": round(sharpness(bgr, width=small.width), 1),
            "detections": detections, "latency_ms": int((time.perf_counter() - started) * 1000), "model": detector.model_id}


class BookFrame(BaseModel):
    frame_ref: str
    path: str
    country: str = ""
    currency: str = ""


def _job(sweep: str) -> BookJob:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", sweep):
        raise HTTPException(422, "Invalid sweep id")
    if sweep not in _state["jobs"]:
        _state["jobs"][sweep] = BookJob(SWEEPS / sweep, _reader, _rates, measurer=_measurer)
    return _state["jobs"][sweep]


@app.post("/books/{sweep}/frames")
async def add_book_frame(sweep: str, frame: BookFrame) -> dict:
    job = _job(sweep)
    path = Path(frame.path).resolve()
    if SWEEPS not in path.parents or not path.is_file():
        raise HTTPException(422, "Frame must be a saved file inside the sweeps folder")
    job.add(frame.frame_ref, path, frame.country, frame.currency)
    return job.status()


class BooksFrame(BaseModel):
    frame_ref: str
    path: str


class AnalyseRequest(BaseModel):
    place: str = ""       # the voice agent's words: "bookshelf by the window, row 2", "book in my hand"
    context: str = ""
    frames: list[BooksFrame] = []
    continue_last: bool = False


def _places(sweep: str):
    from .book_places import BookPlaces
    _job(sweep)
    return _state.setdefault("places", {}).setdefault(sweep, BookPlaces())


@app.post("/books/{sweep}/analyse")
async def analyse_books(sweep: str, request: AnalyseRequest) -> dict:
    """The books found in the frames just shown, recorded under the place the voice agent named."""
    job, places = _job(sweep), _places(sweep)
    refs = [f.frame_ref for f in request.frames]
    if any(SWEEPS not in Path(f.path).resolve().parents for f in request.frames):
        raise HTTPException(422, "Frames must be inside the sweeps folder")
    complete = await job.wait_for(refs)
    by_frame, titles = job.books_by_frame()
    result = (places.continue_last(refs, by_frame, titles) if request.continue_last
              else places.record(request.place, refs, by_frame, titles))
    result["unreadable_spine_text"] = job.unread_on(refs)[:5]
    result["still_reading"] = not complete
    # No running totals here: they go stale and the agent repeated them as final figures.
    # Counts and money come from GET /books/{sweep}/summary (the get_books tool) only.
    return result


@app.get("/books/{sweep}/summary")
async def books_summary(sweep: str, final: bool = False) -> dict:
    """Everything recorded so far, per place. final=true waits for reading to finish."""
    job, places = _job(sweep), _places(sweep)
    if final:
        deadline = time.time() + 60
        while (job.busy or job.queue.qsize()) and time.time() < deadline:
            await asyncio.sleep(0.5)
    _, titles = job.books_by_frame()
    status = job.status()
    return {**places.summary(titles), "books": status["books"], "totals": status["totals"], "currency": job.currency,
            "suggested": status.get("suggested", []), "unidentified_readings": status["unidentified_readings"],
            "still_reading": bool(job.busy or job.queue.qsize())}


def _measurer():
    """Spine measurement models (SAM, Depth Pro), loaded on first use; shares the live detector."""
    if "measurer" not in _state:
        from .measure import FrameMeasurer
        _state["measurer"] = FrameMeasurer(pick_device(), detector=_detector())
    return _state["measurer"]


@app.get("/books/{sweep}/measure")
async def measurement(sweep: str) -> dict:
    """Spine sizes from the frames measured so far; state is "running" until every read frame is measured."""
    job = _job(sweep)
    return await asyncio.to_thread(job.measurement)


class Exclusion(BaseModel):
    title: str = ""               # the title as the policyholder said it; empty means the book in view
    frame_refs: list[str] = []    # the frames in view, used when no title is given
    undo: bool = False
    reason: str = ""


@app.post("/books/{sweep}/exclude")
async def exclude_book(sweep: str, request: Exclusion) -> dict:
    """Leave a book out of the report and totals at the policyholder's request (or put it back)."""
    job = _job(sweep)
    if not request.title and request.frame_refs:
        await job.wait_for(request.frame_refs, timeout=20)      # the book in view may still be being read
    return job.exclude(request.title[:200], request.frame_refs[-6:], undo=request.undo, reason=request.reason[:300])


class BookCount(BaseModel):
    place: str = ""          # empty: the place shown last
    count: int


@app.post("/books/{sweep}/count")
async def set_book_count(sweep: str, request: BookCount) -> dict:
    """The policyholder's count of the books at a place; unidentified books are the difference."""
    if not 0 <= request.count <= 5000:
        raise HTTPException(422, "Count must be between 0 and 5000")
    _job(sweep)
    return _places(sweep).set_count(request.place[:120], request.count)


class Copies(BaseModel):
    title: str
    copies: int


@app.post("/books/{sweep}/copies")
async def set_copies(sweep: str, request: Copies) -> dict:
    """The policyholder's count of copies of one book; totals multiply by it."""
    return _job(sweep).set_copies(request.title[:200], request.copies)


@app.get("/books/{sweep}/report")
async def books_report(sweep: str) -> dict:
    """Everything the session report needs about books: each book with its price and sources,
    the books left out, totals, and the spine measurements so far."""
    job, places = _job(sweep), _places(sweep)
    books = job.lines_out()
    measurement = await asyncio.to_thread(job.measurement)
    # Each book's place as the voice agent named it ("bookshelf by the window, row 2"); a book stays
    # at the first place it was recorded. Books identified after analyse_books returned (reading can
    # lag the camera by minutes) are placed by frame: the place whose shown frames they were read on.
    shelf = {}
    for place in places.places.values():
        for key in place.work_keys:
            shelf.setdefault(key, place.label)
    for book in books:
        if book["work_key"] not in shelf:
            seen = {job._original(f) for f in book["frame_refs"]}
            place = next((pl for pl in places.places.values() if seen & {job._original(f) for f in pl.frame_refs}), None)
            if place:
                shelf[book["work_key"]] = place.label
    unread = [{**u, "frame_refs": sorted(u["frame_refs"])} for u in job.last_result.get("unidentified_readings", [])]
    place_counts = [{"place": pl.label, "stated": pl.stated_count,
                     "identified": sum(b["copies"] for b in books if not b["excluded"] and shelf.get(b["work_key"]) == pl.label)}
                    for pl in places.places.values()]
    return {"country": job.country, "currency": job.currency, "books": books, "totals": job.totals(books),
            "shelf_by_work": shelf, "unread_readings": unread, "possible_copies": job.possible_copies(),
            "place_counts": place_counts,
            "pending_exclusions": job.pending_exclusions, "suggested": job.status().get("suggested", []),
            "unidentified_readings": job.unidentified, "frames_read": job.frames_read,
            "still_reading": bool(job.busy or job.queue.qsize()), "measurement": measurement,
            "room": (_state.get("rooms", {}).get(sweep).status() if _state.get("rooms", {}).get(sweep) else {"state": "not_started"}),
            "usage": job.usage.summary()}


@app.get("/books/{sweep}/usage")
async def books_usage(sweep: str) -> dict:
    """Latency per stage and outside requests (with cost where rates are configured) for this sweep."""
    return _job(sweep).usage.summary()


async def _run_room(sweep: str) -> None:
    from .common import pick_device
    from .telemetry import current as current_usage

    job, room = _job(sweep), _state["rooms"][sweep]
    current_usage.set(job.usage)
    deadline = time.time() + 300
    while (job.busy or job.queue.qsize() or job.measure_queue.qsize() or job.measuring) and time.time() < deadline:
        await asyncio.sleep(1)       # card frames come from spine measurement; one heavy GPU job at a time
    card_frames = [ref for ref, result in sorted(job.frame_measures.items()) if result.card]
    await asyncio.to_thread(room.run, card_frames, _measurer, _detector(), pick_device())


@app.post("/room/{sweep}")
async def start_room(sweep: str) -> dict:
    """Measure the room from the sweep's room frames, in the background (after the sweep ends)."""
    from .live_room import RoomJob

    _job(sweep)      # validates the sweep id
    rooms = _state.setdefault("rooms", {})
    if sweep not in rooms or rooms[sweep].state in {"failed", "not_started"}:
        rooms[sweep] = RoomJob(SWEEPS / sweep)
        rooms[sweep].state = "queued"
        _state.setdefault("room_tasks", {})[sweep] = asyncio.create_task(_run_room(sweep))
    return rooms[sweep].status()


@app.get("/room/{sweep}")
async def room_status(sweep: str) -> dict:
    _job(sweep)
    room = _state.get("rooms", {}).get(sweep)
    return room.status() if room else {"state": "not_started", "room": None, "error": None, "seconds": None}


class Locale(BaseModel):
    country: str = ""
    currency: str = ""


@app.post("/books/{sweep}/locale")
async def set_book_locale(sweep: str, locale: Locale) -> dict:
    job = _job(sweep)
    await job.set_locale(locale.country, locale.currency)
    return job.status()


@app.get("/books/{sweep}/status")
async def book_status(sweep: str) -> dict:
    job = _job(sweep)
    job.maybe_retry()
    return job.status()
