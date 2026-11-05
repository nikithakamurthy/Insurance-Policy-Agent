"""Live worker for the voice agent: object detection per frame, and the book agent per sweep.

    .venv/bin/python -m uvicorn pipeline.vision_service:app --host 127.0.0.1 --port 4180

POST /detect                    one JPEG frame (raw body) -> objects in view, positions, sharpness.
                                Detection runs on a copy at most 640 px wide; boxes are returned
                                in the original frame's pixels.
POST /books/{sweep}/frames      queue a saved frame with a book in it for books_agent.
POST /books/{sweep}/analyse     the books in the frames just shown, recorded under a place.
GET  /books/{sweep}/summary     everything found so far, per place (final=true waits for reading).
GET  /books/{sweep}/status      progress and results.
POST /books/{sweep}/exclude     leave a book out of the report and totals (by title or the book in view), or undo.
POST /books/{sweep}/copies      how many copies of a book the policyholder has.
GET  /books/{sweep}/report      books, prices, exclusions, totals and measurements for the session report.
POST /room/{sweep}/wall         one wall of the room: frames held on its paper sheet, measured now (Depth Pro + sheet).

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
    result["to_confirm"] = job.to_confirm(refs)
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
            "suggested": status.get("suggested", []), "to_confirm": job.to_confirm(),
            "unidentified_readings": status["unidentified_readings"], "still_reading": bool(job.busy or job.queue.qsize())}


def _measurer():
    """Spine measurement models (SAM, Depth Pro), loaded on first use; shares the live detector."""
    if "measurer" not in _state:
        from .measure import FrameMeasurer
        _state["measurer"] = FrameMeasurer(pick_device(), detector=_detector())
    return _state["measurer"]


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


class Copies(BaseModel):
    title: str
    copies: int


@app.post("/books/{sweep}/copies")
async def set_copies(sweep: str, request: Copies) -> dict:
    """The policyholder's count of copies of one book; totals multiply by it."""
    return _job(sweep).set_copies(request.title[:200], request.copies)


class Confirmation(BaseModel):
    title: str
    confirmed: bool


@app.post("/books/{sweep}/confirm")
async def confirm_book(sweep: str, request: Confirmation) -> dict:
    """The policyholder's yes or no about a book the team suggested but could not read clearly enough."""
    job = _job(sweep)
    result = job.confirm(request.title[:200], request.confirmed)
    if result["status"] == "added" and job.currency:
        await asyncio.to_thread(job._price_new)
    job._write(job.last_result)
    return result


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
    return {"country": job.country, "currency": job.currency, "books": books, "totals": job.totals(books),
            "shelf_by_work": shelf, "unread_readings": unread, "possible_copies": job.possible_copies(),
            "pending_exclusions": job.pending_exclusions, "suggested": job.status().get("suggested", []),
            "unidentified_readings": job.unidentified, "frames_read": job.frames_read,
            "room": _room(sweep).status(),
            "still_reading": bool(job.busy or job.queue.qsize()), "measurement": measurement,
            "usage": job.usage.summary()}





class RoomWall(BaseModel):
    wall: str                 # "1" or "2": which of the two walls the policyholder is holding the phone on
    paper: str = "A4"         # A4 or Letter
    frames: list[BooksFrame]  # saved frames from the hold


def _room(sweep: str):
    from .room_corner import CornerRoomJob

    _job(sweep)      # validates the sweep id
    return _state.setdefault("rooms", {}).setdefault(sweep, CornerRoomJob())


@app.post("/room/{sweep}/wall")
async def add_room_wall(sweep: str, request: RoomWall) -> dict:
    """Measure one wall from the frames held on its sheet; answers when done (a few seconds per frame)."""
    from .telemetry import current as current_usage
    from .telemetry import stage

    room = _room(sweep)
    frames = []
    for f in request.frames[:12]:
        path = Path(f.path).resolve()
        if SWEEPS not in path.parents or not path.is_file():
            raise HTTPException(422, "Frames must be saved files inside the sweeps folder")
        frames.append((f.frame_ref, path))
    measurer = _measurer()
    current_usage.set(_job(sweep).usage)
    with stage("room: one wall (sheet + Depth Pro)"):
        return await asyncio.to_thread(room.add_wall, request.wall[:8], request.paper, frames,
                                       measurer.detector, measurer.segmenter, measurer.depth_model)





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
