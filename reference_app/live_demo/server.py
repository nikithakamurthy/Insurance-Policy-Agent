"""FastAPI backend for the add-to-policy live agent.

The browser transport lives here: the Gemini Live voice agent and its tools, camera frames to the
vision worker (pipeline/vision_service.py), and the session report and packet (claim_packet.json, the brief's format).
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
import os
import re
import sys
import time
import uuid
import secrets
import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

APP_DIR = Path(__file__).resolve().parents[1]
DEMO_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(APP_DIR.parent) not in sys.path:      # the repo root, for pipeline.telemetry (standard library only)
    sys.path.append(str(APP_DIR.parent))


def _load_dotenv() -> None:
    env_path = APP_DIR / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _cors_origins() -> list[str]:
    raw = os.getenv("FNOL_CORS_ORIGINS", "")
    if raw.strip():
        return [origin.strip() for origin in raw.split(",") if origin.strip()]
    return ["http://127.0.0.1:4177", "http://localhost:4177"]


_load_dotenv()

if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from locales import country_code, resolve_country  # noqa: E402
from narration import PROMPT_KINDS, SceneTracker, book_events, describe, needs_reply, notice_text, ready_to_speak  # noqa: E402
from packet import build_packet  # noqa: E402
from report import build_report  # noqa: E402
from pipeline.telemetry import Usage, cost, stage  # noqa: E402

# Live worker (pipeline/vision_service.py): detection per frame and the book agent per sweep.
# An empty URL disables live narration and live book identification.
VISION_URL = os.getenv("FNOL_VISION_URL", "http://127.0.0.1:4180/detect")
WORKER_URL = VISION_URL.rsplit("/detect", 1)[0]
VISION_INTERVAL_S = 1.0
SWEEPS_DIR = Path(os.getenv("FNOL_SWEEPS_DIR", Path(__file__).resolve().parents[2] / "sweeps"))
MAX_FRAME_BYTES = 6 * 1024 * 1024
ROOM_HOLD_S = 4.0         # measure_room: frames taken while the policyholder holds still on one wall's sheet
ROOM_HOLD_MAX_FRAMES = 8
ROOM_STEP_MAX_S = 300     # a room step left open this long ends by itself, so object notices are never held for good
ROOM_HELD_KINDS = {"new_objects", "bookshelf", "books_identified"}
KEEP_EVERY_S = 1.5        # at most one kept spine frame per 1.5 s; a paused camera repeats itself
BOOK_POLL_S = 4.0
from live_tools import (  # noqa: E402
    LIVE_MODEL_ID,
    TOOL_NAMES,
    build_live_config,
    camera_mode_instruction,
    locale_instruction,
    scheduling_for,
    tool_headline,
)

GENAI_CLIENT = None
logger = logging.getLogger(__name__)


def audio_message(blob):
    """The voice agent's audio for the browser; anything else in the stream is ignored."""
    if not isinstance(blob.data, bytes) or not (blob.mime_type or "").startswith("audio/pcm"):
        return None
    return {"type": "audio", "data": base64.b64encode(blob.data).decode("ascii"), "mime_type": blob.mime_type}


class LocaleRequest(BaseModel):
    country: str = Field(default="", max_length=60)
    currency: str = Field(default="", max_length=3)


class SessionResponse(BaseModel):
    session_id: str
    model: str
    has_api_key: bool
    state: dict[str, Any]


@dataclass
class IntakeSession:
    session_id: str
    transcript: list[dict[str, str]] = field(default_factory=list)
    live_model: str | None = None
    tool_activity: list[dict[str, Any]] = field(default_factory=list)
    last_frame: bytes | None = None              # the agent's latest camera frame, kept as a logged item's evidence
    camera_enabled: bool = False
    country: str = ""
    currency: str = ""
    items: list = field(default_factory=list)   # objects the voice agent logged (books are separate)
    owner: str = ""
    updated_at: float = field(default_factory=time.monotonic)
    revision: int = 0
    deleted: bool = False
    live_socket: Any = None
    tasks: set = field(default_factory=set)
    # live sweep: high-resolution frames arrive over HTTP and are handled by the live socket
    on_frame: Any = None
    analyse_tool: Any = None
    books_tool: Any = None
    exclude_tool: Any = None
    copies_tool: Any = None
    confirm_tool: Any = None
    room_tool: Any = None
    room_step_tool: Any = None
    usage: Any = field(default_factory=Usage)    # latency per stage and Gemini Live tokens, this session
    sweep_started: str = ""                      # ISO time the camera session started
    detected: dict = field(default_factory=dict)  # detector-confirmed contents: category -> count, score, frame_ref
    report_live: bool = False                    # the live report panel has been opened on the page
    device: str = ""                             # the browser's user agent, for the packet
    live_tokens_seen: int = 0                    # highest running token count Gemini Live reported
    sweep_id: str = ""


sessions: dict[str, IntakeSession] = {}

app = FastAPI(title="Add-to-policy live agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_methods=["*"],
    allow_headers=["*"],
)


def _has_api_key() -> bool:
    return bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))


def _client():
    global GENAI_CLIENT
    if not _has_api_key():
        raise HTTPException(
            status_code=503,
            detail=(
                "Missing GOOGLE_API_KEY. Add it to "
                f"{APP_DIR / '.env'} and restart the live intake backend."
            ),
        )
    if os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
    try:
        from google import genai
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail="Missing google-genai package. Run pip install -r requirements.txt.",
        ) from exc
    if GENAI_CLIENT is None:
        GENAI_CLIENT = genai.Client()
    return GENAI_CLIENT


def append_turn(session: IntakeSession, speaker: str, text: str, turn_id: str | None = None):
    text = text.strip()[:8000]
    turn_id = turn_id or uuid.uuid4().hex
    if any(t.get("id") == turn_id for t in session.transcript):
        return turn_id
    if len(session.transcript) >= 300 or sum(len(t["text"]) for t in session.transcript) + len(text) > 64000:
        raise ValueError("This intake reached its conversation limit. Download the packet and start a new intake.")
    session.transcript.append({"id": turn_id, "speaker": speaker, "text": text})
    if speaker == "Claimant":
        session.revision += 1
    session.updated_at = time.monotonic()
    return turn_id


def _ui_state(session: IntakeSession) -> dict[str, Any]:
    """What the page shows: the conversation, the team's activity, the locale and what was logged."""
    return {"session_id": session.session_id, "revision": session.revision, "transcript": session.transcript,
            "tool_activity": session.tool_activity[-12:], "live_model": session.live_model,
            "locale": {"country": session.country, "currency": session.currency},
            "items": session.items}


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "has_api_key": _has_api_key(), "live_model": LIVE_MODEL_ID, "tools": TOOL_NAMES}


SESSION_TTL = 30 * 60
MAX_SESSIONS = 32
MAX_MESSAGE_BYTES = 800000


def allowed_origin(origin: str | None, host: str, scheme="http") -> bool:
    if not origin:
        return False
    return origin in set(_cors_origins()) | {f"{scheme}://{host}"}


def local_host(host: str) -> bool:
    return host.split(":")[0] in {"localhost", "127.0.0.1"}


@app.middleware("http")
async def local_access(request: Request, call_next):
    # Local-only demo: public serving requires a separate authenticated deployment design.
    if not local_host(request.headers.get("host", "")) or request.client.host not in {"127.0.0.1", "::1", "testclient"}:
        return Response("This demo accepts local connections only.", status_code=403)
    origin = request.headers.get("origin")
    if (origin and not allowed_origin(origin, request.headers.get("host", ""), request.url.scheme)) or (request.method not in {"GET", "HEAD", "OPTIONS"} and not origin):
        return Response("Origin not allowed", status_code=403)
    try:
        length = int(request.headers.get("content-length", 0) or 0)
    except ValueError:
        return Response("Invalid Content-Length", status_code=400)
    limit = MAX_FRAME_BYTES if request.url.path.endswith("/frames") else 16000
    if request.url.path.startswith("/api") and length > limit:
        return Response("Request too large", status_code=413)
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    return response


async def discard_session(session: IntakeSession):
    session.deleted = True
    sessions.pop(session.session_id, None)
    if session.live_socket:
        with contextlib.suppress(Exception):
            await session.live_socket.close(code=1000)
    for task in list(session.tasks):
        task.cancel()
    if session.tasks:
        await asyncio.gather(*list(session.tasks), return_exceptions=True)
    session.last_frame = None


async def cleanup_sessions():
    for session in list(sessions.values()):
        if time.monotonic() - session.updated_at > SESSION_TTL:
            await discard_session(session)


def owned_session(session_id: str, owner: str | None):
    session = sessions.get(session_id)
    if not session or session.deleted or not owner or not secrets.compare_digest(session.owner, owner):
        raise HTTPException(404, "Intake not found or expired. Start a new intake.")
    if time.monotonic() - session.updated_at > SESSION_TTL:
        raise HTTPException(410, "Intake expired. Start a new intake.")
    session.updated_at = time.monotonic()
    return session


@app.post("/api/sessions", response_model=SessionResponse)
async def create_session(request: Request, response: Response) -> SessionResponse:
    await cleanup_sessions()
    owner = request.cookies.get("intake_owner") or secrets.token_urlsafe(32)
    if len(sessions) >= MAX_SESSIONS or sum(s.owner == owner for s in sessions.values()) >= 4:
        raise HTTPException(429, "Too many intakes. Close or reset an existing intake first.")
    session = IntakeSession(session_id=uuid.uuid4().hex, owner=owner)
    append_turn(session, "Agent", "Hi, I'm your home-contents insurance assistant. I'll help you add the things you own to your policy. Which country do you live in?")
    sessions[session.session_id] = session
    response.set_cookie("intake_owner", owner, httponly=True, samesite="strict", max_age=SESSION_TTL, secure=request.url.scheme == "https")
    return SessionResponse(session_id=session.session_id, model=LIVE_MODEL_ID, has_api_key=_has_api_key(), state=_ui_state(session))


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str, request: Request):
    session = owned_session(session_id, request.cookies.get("intake_owner"))
    return {"session_id": session.session_id, "state": _ui_state(session), "has_api_key": _has_api_key()}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str, request: Request):
    session = owned_session(session_id, request.cookies.get("intake_owner"))
    await discard_session(session)
    return {"deleted": True}


async def write_report(session: IntakeSession) -> tuple[str, Path | None]:
    """Build the session report from the logged items and the book worker's results, and save it
    as report.html in the sweep folder. Without a sweep or worker, objects only."""
    books = None
    if session.sweep_id and VISION_URL:
        try:
            async with __import__("httpx").AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{WORKER_URL}/books/{session.sweep_id}/report")
                response.raise_for_status()
                books = response.json()
        except Exception:
            logger.warning("Book worker did not answer for the report of %s", session.sweep_id, exc_info=True)
    usage = session_usage(session, (books or {}).get("usage"))
    sweep_dir = SWEEPS_DIR / session.sweep_id if session.sweep_id else None
    frames = json.loads((sweep_dir / "frames.json").read_text())["frames"] if sweep_dir and (sweep_dir / "frames.json").exists() else []
    packet = build_packet({"sweep_id": session.sweep_id, "captured_at": session.sweep_started or None, "device": session.device,
                           "duration_s": max((f["t"] for f in frames), default=None), "country": session.country,
                           "currency": session.currency},
                          session.items, books, session.detected)
    page = build_report(packet, usage)
    path = None
    if sweep_dir:
        sweep_dir.mkdir(parents=True, exist_ok=True)
        path = sweep_dir / "report.html"
        path.write_text(page)
        (sweep_dir / "claim_packet.json").write_text(json.dumps(packet, indent=2, ensure_ascii=False))
        (sweep_dir / "usage.json").write_text(json.dumps(usage, indent=2))
    return page, path


def session_usage(session: IntakeSession, worker: dict | None) -> dict:
    """This session's latency per stage and outside usage: the app's (voice model, detection round
    trips, tools) and the book worker's (OCR, search, matching, pricing, measurement)."""
    mine = session.usage.summary()
    calls = {name: dict(entry) for name, entry in (worker or {}).get("calls", {}).items()}
    for name, entry in mine["calls"].items():
        total = calls.setdefault(name, {"requests": 0, "errors": 0, "input_tokens": 0, "output_tokens": 0})
        for key in total:
            total[key] += entry[key]
    return {"stages": {**(worker or {}).get("stages", {}), **mine["stages"]}, "calls": calls, "cost": cost(calls),
            "gemini_live_running_total_tokens": session.live_tokens_seen,
            "book_worker_reached": worker is not None}


@app.get("/api/sessions/{session_id}/claim_packet.json")
async def session_packet(session_id: str, request: Request):
    """claim_packet.json for this sweep, rebuilt now (the report page is saved alongside)."""
    session = owned_session(session_id, request.cookies.get("intake_owner"))
    await write_report(session)
    if not session.sweep_id:
        raise HTTPException(404, "No sweep yet")
    return FileResponse(SWEEPS_DIR / session.sweep_id / "claim_packet.json", media_type="application/json",
                        headers={"Content-Disposition": 'attachment; filename="claim_packet.json"', "Cache-Control": "no-store"})


@app.get("/api/sessions/{session_id}/{folder}/{name}")
def session_frame(session_id: str, folder: str, name: str, request: Request):
    """A saved frame the report or packet points to (frames/f00012.jpg, items/I003.jpg, room_frames/wall1_0.jpg)."""
    session = owned_session(session_id, request.cookies.get("intake_owner"))
    if folder not in {"frames", "items", "room_frames"} or not re.fullmatch(r"[A-Za-z0-9_-]{1,40}\.jpg", name) or not session.sweep_id:
        raise HTTPException(404, "Not found")
    path = SWEEPS_DIR / session.sweep_id / folder / name
    if not path.is_file():
        raise HTTPException(404, "Not found")
    return FileResponse(path, media_type="image/jpeg")


@app.post("/api/sessions/{session_id}/finish")
async def finish_sweep(session_id: str, request: Request):
    """Build the report and packet now."""
    session = owned_session(session_id, request.cookies.get("intake_owner"))
    _, path = await write_report(session)
    return {"report": str(path) if path else None}


@app.get("/api/sessions/{session_id}/report")
async def session_report(session_id: str, request: Request):
    """The report, rebuilt on each request so late measurements and prices show up; also saved."""
    session = owned_session(session_id, request.cookies.get("intake_owner"))
    page, _ = await write_report(session)
    return HTMLResponse(page, headers={"Cache-Control": "no-store"})


def set_camera_mode(session: IntakeSession, enabled: bool) -> bool:
    changed = session.camera_enabled != enabled
    session.camera_enabled = enabled
    if not enabled:
        session.last_frame = None      # an old frame must not become evidence for an item logged later
    return changed


def set_locale(session: IntakeSession, country: Any, currency: Any) -> bool:
    """Validate and store the pricing locale. Returns whether it changed."""
    if not isinstance(country, str) or not isinstance(currency, str):
        raise ValueError("Country and currency must be text")
    country, currency = " ".join(country.split()), currency.strip().upper()
    if len(country) > 60 or (country and not all(c.isalpha() or c in " .'-()" for c in country)):
        raise ValueError("Country must be a name of up to 60 letters")
    if currency and not (len(currency) == 3 and currency.isascii() and currency.isalpha()):
        raise ValueError("Currency must be a 3-letter ISO code such as USD or INR")
    changed = (session.country, session.currency) != (country, currency)
    session.country, session.currency = country, currency
    if changed:
        session.revision += 1
    return changed


MAX_ITEMS = 300


def log_item(session: IntakeSession, args: dict[str, Any]) -> dict[str, Any]:
    """Log an object the voice agent saw, or update one already logged (by item_id)."""
    text = lambda k, n: " ".join(str(args.get(k) or "").split())[:n]
    item_id = text("item_id", 10).upper()
    if item_id:
        item = next((i for i in session.items if i["id"] == item_id), None)
        if item is None:
            raise ValueError(f"No item {item_id} has been logged.")
        for key, n in (("brand_model", 120), ("where", 120), ("description", 160)):
            if text(key, n):
                item[key] = text(key, n)
        if text("note", 300):
            item["notes"].append(text("note", 300))
        return {"item_id": item["id"], "description": item["description"], "updated": True, "items_logged": len(session.items)}
    description = text("description", 160)
    if not description:
        raise ValueError("Describe the item, for example 'grey three-seat sofa'.")
    same = next((i for i in session.items if i["description"].lower() == description.lower()), None)
    if same:
        return {"item_id": same["id"], "description": same["description"], "already_logged": True, "items_logged": len(session.items)}
    if len(session.items) >= MAX_ITEMS:
        raise ValueError("Item limit reached for this session.")
    item = {"id": f"I{len(session.items) + 1:03d}", "category": text("category", 40).lower() or "other",
            "description": description, "brand_model": text("brand_model", 120), "where": text("where", 120),
            "notes": [text("note", 300)] if text("note", 300) else [], "logged_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    session.items.append(item)
    session.revision += 1
    return {"item_id": item["id"], "description": description, "items_logged": len(session.items)}


def save_item_frame(session: IntakeSession, item_id: str) -> None:
    """Keep the camera frame in view when an object is logged, as its evidence (items/I003.jpg)."""
    item = next((i for i in session.items if i["id"] == item_id), None)
    if item is None or not session.last_frame or not session.sweep_id:
        return
    path = SWEEPS_DIR / session.sweep_id / "items" / f"{item_id}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(session.last_frame)
    item["frame_ref"] = f"items/{item_id}.jpg"


def set_country(session: IntakeSession, country: Any, currency: Any = "") -> dict[str, Any]:
    """Set the pricing locale from a country; the currency follows from the country.

    A currency the policyholder states is used only when the country is not in the table
    (a territory, or an unusual name). Raises ValueError when neither works.
    """
    if not isinstance(country, str) or not country.strip():
        raise ValueError("Say which country the policyholder lives in.")
    resolved = resolve_country(country)
    if resolved:
        name, code = resolved
        source = "derived from country"
    elif isinstance(currency, str) and currency.strip():
        name, code = " ".join(country.split()), currency.strip().upper()
        source = "stated by policyholder"
    else:
        raise ValueError(f"Unknown country '{country.strip()[:60]}'. Ask which currency they want, "
                         "then call set_country again with that currency.")
    set_locale(session, name, code)
    return {"country": session.country, "currency": session.currency, "currency_source": source}


@app.put("/api/sessions/{session_id}/locale")
def update_locale(session_id: str, body: LocaleRequest, request: Request):
    session = owned_session(session_id, request.cookies.get("intake_owner"))
    try:
        set_country(session, body.country, body.currency)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"locale": {"country": session.country, "currency": session.currency}}


@app.post("/api/sessions/{session_id}/frames")
async def upload_frame(session_id: str, request: Request):
    """One high-resolution camera frame (raw JPEG body) for live detection and the book agent."""
    session = owned_session(session_id, request.cookies.get("intake_owner"))
    if session.on_frame is None:
        raise HTTPException(409, "Start the live call before sending frames.")
    data = await request.body()
    if not data.startswith(b"\xff\xd8\xff") or len(data) > MAX_FRAME_BYTES:
        raise HTTPException(422, "Send one JPEG frame as the request body")
    return await session.on_frame(data)


@app.on_event("startup")
async def start_cleanup():
    async def sweep():
        while True:
            await asyncio.sleep(60)
            await cleanup_sessions()
    app.state.cleanup_task = asyncio.create_task(sweep())


@app.on_event("shutdown")
async def stop_cleanup():
    app.state.cleanup_task.cancel()
    await asyncio.gather(app.state.cleanup_task, return_exceptions=True)
    for session in list(sessions.values()):
        await discard_session(session)


@app.websocket("/ws/live")
async def live_voice(websocket: WebSocket) -> None:
    host = websocket.headers.get("host", "")
    if not local_host(host) or websocket.client.host not in {"127.0.0.1", "::1", "testclient"} or not allowed_origin(websocket.headers.get("origin"), host, "https" if websocket.url.scheme == "wss" else "http"):
        await websocket.close(code=1008)
        return
    try:
        session = owned_session(websocket.query_params.get("session_id", ""), websocket.cookies.get("intake_owner"))
    except HTTPException:
        await websocket.close(code=1008)
        return
    if session.live_socket is not None:
        await websocket.close(code=1008)
        return
    session.live_socket = websocket
    session.live_model = LIVE_MODEL_ID
    await websocket.accept()
    from google.genai import types
    send_lock = asyncio.Lock()
    tasks = set()
    tool_tasks = {}
    pending = {"Claimant": {"id": uuid.uuid4().hex, "text": ""}, "Agent": {"id": uuid.uuid4().hex, "text": ""}}

    async def send(payload):
        if session.deleted:
            return
        async with send_lock:
            await websocket.send_json({**payload, "session_id": session.session_id})

    def track(task):
        tasks.add(task)
        session.tasks.add(task)
        def finished(done):
            tasks.discard(done)
            session.tasks.discard(done)
            if not done.cancelled():
                error = done.exception()
                if error:
                    logger.error("Background intake task failed: %s", type(error).__name__)
        task.add_done_callback(finished)
        return task

    async def finalize(speaker):
        turn = pending[speaker]
        if not turn["text"].strip():
            return
        append_turn(session, speaker, turn["text"], turn["id"])
        await send({"type": "transcript", "speaker": speaker, "text": turn["text"], "id": turn["id"], "final": True})
        pending[speaker] = {"id": uuid.uuid4().hex, "text": ""}

    async def publish_tool(entry):
        session.tool_activity = [item for item in session.tool_activity if item["id"] != entry["id"]] + [dict(entry)]
        session.tool_activity = session.tool_activity[-30:]
        await send({"type": "tool", **entry})

    async def execute_tool(fc, live_session):
        started = time.monotonic()
        name, args, call_id = str(fc.name or ""), dict(fc.args or {}), str(fc.id or uuid.uuid4().hex)
        entry = {"id": call_id, "name": name, "args": args, "phase": "running", "headline": tool_headline(name, args, None), "model": LIVE_MODEL_ID}
        await publish_tool(entry)
        urgent = False
        try:
            if name in {"analyse_books", "get_books", "exclude_book", "set_book_copies", "confirm_book", "measure_room", "room_step"}:
                tool = {"analyse_books": session.analyse_tool, "get_books": session.books_tool,
                        "exclude_book": session.exclude_tool, "set_book_copies": session.copies_tool,
                        "confirm_book": session.confirm_tool,
                        "measure_room": session.room_tool,
                        "room_step": session.room_step_tool}[name]
                result = await tool(args) if tool else {"error": "The camera is not connected yet."}
            elif name == "log_item":
                try:
                    result = log_item(session, args)
                    if not result.get("updated") and not result.get("already_logged"):
                        save_item_frame(session, result["item_id"])
                except ValueError as exc:
                    result = {"error": str(exc)}
            elif name == "set_country":
                try:
                    result = set_country(session, args.get("country", ""), args.get("currency", ""))
                except ValueError as exc:
                    result = {"error": str(exc)}
            else:
                result = {"error": "Unknown tool"}
        except asyncio.CancelledError:
            entry.update(phase="cancelled", headline="Cancelled")
            with contextlib.suppress(Exception):
                await publish_tool(entry)
            raise
        except Exception:
            logger.exception("Tool %s failed", name)
            result = {"error": f"{name} failed. Continue the conversation and retry if needed."}
        session.usage.add_stage(f"tool: {name}", time.monotonic() - started)
        scheduling = scheduling_for(urgent=urgent)
        entry.update(phase="error" if "error" in result else "done", headline=result.get("error") or tool_headline(name, args, result), duration_ms=int((time.monotonic() - started) * 1000), result=result, scheduling=scheduling.value if scheduling else None)
        await publish_tool(entry)
        await send({"type": "state", "state": _ui_state(session)})
        await live_session.send_tool_response(function_responses=[types.FunctionResponse(id=call_id, name=name, response=result, scheduling=scheduling)])

    async def launch_tool(fc, live_session):
        if str(fc.id) in tool_tasks:
            return
        if len(tool_tasks) >= 4:
            await live_session.send_tool_response(function_responses=[types.FunctionResponse(id=fc.id, name=fc.name, response={"error": "The team is busy. Wait for current tools to finish."})])
            return
        task = track(asyncio.create_task(execute_tool(fc, live_session)))
        tool_tasks[str(fc.id)] = task
        task.add_done_callback(lambda done, key=str(fc.id): tool_tasks.pop(key, None))

    try:
        if not _has_api_key():
            await send({"type": "error", "message": "A Google API key is required in the server environment."})
            return
        history = [
            types.Content(
                role="user" if turn["speaker"] == "Claimant" else "model",
                parts=[types.Part(text=turn["text"])],
            )
            for turn in session.transcript
            if turn["speaker"] in {"Claimant", "Agent"}
        ]
        config = build_live_config(camera_enabled=session.camera_enabled, seed_history=bool(history),
                                   country=session.country, currency=session.currency)
        # Restore dialogue as context before accepting another turn on reconnect.
        async with _client().aio.live.connect(model=LIVE_MODEL_ID, config=config) as live_session:
            if history:
                # The SDK has awaited setup_complete. Close the initial-history
                # batch without treating it as a new request for speech.
                await live_session.send_client_content(turns=history, turn_complete=True)
            await send({"type": "session", "model": LIVE_MODEL_ID, "tools": TOOL_NAMES})
            await send({"type": "state", "state": _ui_state(session)})
            await send({"type": "ready"})

            # ---- live narration: frames -> vision worker -> scene tracker -> agent ----
            tracker = SceneTracker()
            queued_events: list[dict] = []
            voice = {"agent_speaking": False, "user_voice_at": 0.0, "agent_done_at": 0.0, "notice_at": 0.0}
            vision = {"busy": False, "last": 0.0, "down_until": 0.0, "warned": False}
            # The room step (measuring the room first): object notices are held and summarised when it ends.
            room_step = {"active": False, "started": 0.0, "held": [], "walls": set(), "ending": None}
            room_measuring: list = []          # background wall measurements (asyncio tasks)
            vision_http = __import__("httpx").AsyncClient(timeout=8.0) if VISION_URL else None
            session.sweep_id = session.sweep_id or f"live-{time.strftime('%Y%m%d-%H%M%S')}-{session.session_id[:6]}"
            session.sweep_started = session.sweep_started or time.strftime("%Y-%m-%dT%H:%M:%S%z")
            session.device = websocket.headers.get("user-agent", "")[:200]
            sweep_dir = SWEEPS_DIR / session.sweep_id
            # A reconnect reuses the sweep; continue its frame numbering instead of overwriting.
            previous_frames = json.loads((sweep_dir / "frames.json").read_text())["frames"] if (sweep_dir / "frames.json").exists() else []
            # since_call: frames kept since the last analyse_books call (the books just shown).
            kept = {"frames": previous_frames, "last_at": 0.0, "started": time.monotonic(), "since_call": []}

            async def worker_down(exc: Exception):
                vision["down_until"] = time.monotonic() + 30
                if not vision["warned"]:
                    vision["warned"] = True
                    logger.warning("Live worker unavailable at %s: %s", WORKER_URL, type(exc).__name__)
                    await send({"type": "error", "message": "Live object detection and book reading are not running; the agent can still see the camera."})

            async def feed(events: list[dict], result: dict | None = None):
                queued_events.extend(events)
                await publish_tool({"id": uuid.uuid4().hex, "name": "vision", "args": {}, "phase": "done",
                                    "headline": describe(events)[:160], "model": (result or {}).get("model", ""),
                                    "duration_ms": (result or {}).get("latency_ms")})

            def keep_frame(frame: bytes, result: dict) -> dict:
                """Save a frame worth reading into the sweep folder, in the offline pipeline's layout."""
                (sweep_dir / "frames").mkdir(parents=True, exist_ok=True)
                frame_ref = f"f{len(kept['frames']):05d}"
                path = sweep_dir / "frames" / f"{frame_ref}.jpg"
                path.write_bytes(frame)
                record = {"id": frame_ref, "t": round(time.monotonic() - kept["started"], 2), "path": f"frames/{frame_ref}.jpg",
                          "width": result.get("width"), "height": result.get("height"), "sharpness": result.get("sharpness"),
                          "blurry": False, "n_books": sum(d["category"] == "book" for d in result.get("detections", [])),
                          "bookshelf": any(d["category"] == "bookshelf" for d in result.get("detections", []))}
                kept["frames"].append(record)
                kept["since_call"].append({"frame_ref": frame_ref, "path": str(path)})
                (sweep_dir / "frames.json").write_text(json.dumps({"source": "live", "frames": kept["frames"]}, indent=1))
                return {**record, "abs_path": str(path)}

            async def show_report_once(events: list[dict]) -> None:
                """Open the live report on the page as soon as the team identifies something."""
                if not session.report_live and any(e["kind"] in {"new_objects", "bookshelf", "books_identified"} for e in events):
                    session.report_live = True
                    await send({"type": "report_live"})

            async def on_frame(frame: bytes) -> dict:
                now = time.monotonic()
                vision["last_frame"] = frame          # the shelf tool can always use what is in view now
                if not vision_http or vision["busy"] or now - vision["last"] < VISION_INTERVAL_S or now < vision["down_until"]:
                    return {"analysed": False}
                vision["busy"], vision["last"] = True, now
                try:
                    with stage("detect: round trip", session.usage):
                        # 15 s: detection shares the GPU with OCR and spine measurement and can wait behind them.
                        response = await vision_http.post(VISION_URL, content=frame, headers={"Content-Type": "image/jpeg"}, timeout=15.0)
                        response.raise_for_status()
                        result = response.json()
                    if result.get("latency_ms") is not None:
                        session.usage.add_stage("detect: model on the worker", result["latency_ms"] / 1000)
                except Exception as exc:
                    await worker_down(exc)
                    return {"analysed": False}
                finally:
                    vision["busy"] = False
                if vision["warned"]:                   # back after an outage: say so, and warn again next time
                    vision["warned"] = False
                    await send({"type": "error", "message": "Live object detection and book reading are running again."})
                vision["last_result"] = result
                events = tracker.update(result, time.monotonic())
                for category, seen in tracker.contents().items():
                    entry = session.detected.setdefault(category, {})
                    if "frame_ref" not in entry:       # the first sighting is the evidence frame
                        (sweep_dir / "items").mkdir(parents=True, exist_ok=True)
                        (sweep_dir / "items" / f"detected-{category}.jpg").write_bytes(frame)
                        entry["frame_ref"] = f"items/detected-{category}.jpg"
                    entry.update(seen)
                if events:
                    await feed(events, result)
                    await show_report_once(events)
                if not (tracker.worth_reading(result) and now - kept["last_at"] >= KEEP_EVERY_S):
                    return {"analysed": True, "kept": False}
                kept["last_at"] = now
                record = keep_frame(frame, result)
                await queue_for_books(record)
                return {"analysed": True, "kept": True, "frame_ref": record["id"]}

            async def queue_for_books(record: dict):
                try:
                    await vision_http.post(f"{WORKER_URL}/books/{session.sweep_id}/frames", json={
                        # Price lookups take ISO country codes ("DE"), not names ("Germany").
                        "frame_ref": record["id"], "path": record["abs_path"], "country": country_code(session.country),
                        "currency": session.currency if country_code(session.country) else ""})
                except Exception as exc:
                    await worker_down(exc)

            session.on_frame = on_frame

            async def analyse_tool(args: dict) -> dict:
                """analyse_books: the books in the frames shown since the last call, recorded under the
                place the voice agent names (a shelf row, a table, a pile, a book in someone's hand)."""
                if not vision_http:
                    return {"error": "The books agent is not running."}
                frames = kept["since_call"][:]
                kept["since_call"].clear()
                # Nothing with a book kept since the last call (for example the book is still in
                # view and unchanged): use what is in view now.
                if not frames and vision.get("last_frame"):
                    record = keep_frame(vision["last_frame"], vision.get("last_result") or {})
                    await queue_for_books(record)
                    frames = [{"frame_ref": record["id"], "path": record["abs_path"]}]
                    kept["since_call"].clear()     # keep_frame queued it for the next call too; it belongs to this one
                if not frames:
                    return {"error": "No camera frame yet. Ask them to point the camera at the books."}
                try:
                    response = await vision_http.post(f"{WORKER_URL}/books/{session.sweep_id}/analyse", timeout=45.0, json={
                        "place": str(args.get("place", ""))[:120], "context": str(args.get("context", ""))[:300],
                        "frames": frames[-12:]})
                    response.raise_for_status()
                    result = response.json()
                except Exception as exc:
                    await worker_down(exc)
                    return {"error": "The books agent did not answer. Carry on; the frames are saved."}
                if len(result.get("books_here", [])) > 12:
                    result["books_here"] = result["books_here"][:12] + [f"and {len(result['books_here']) - 12} more"]
                return result

            async def books_tool(args: dict) -> dict:
                """get_books: the book team's results so far, or the final summary once reading is done."""
                if not vision_http:
                    return {"error": "The book team is not running."}
                final = bool(args.get("final"))
                if final:
                    await send({"type": "report", "path": ""})      # show it now; it fills in as the team finishes
                if final and kept["since_call"]:
                    # Frames shown after the last analyse_books call would be counted in the total
                    # but at no place; hand them to the place shown last before summarising.
                    frames = kept["since_call"][:]
                    kept["since_call"].clear()
                    try:
                        await vision_http.post(f"{WORKER_URL}/books/{session.sweep_id}/analyse", timeout=45.0,
                                               json={"continue_last": True, "frames": frames})
                    except Exception as exc:
                        await worker_down(exc)
                try:
                    data = (await vision_http.get(f"{WORKER_URL}/books/{session.sweep_id}/summary",
                                                  params={"final": str(final).lower()}, timeout=75.0)).json()
                except Exception as exc:
                    await worker_down(exc)
                    return {"error": "The book team did not answer. The frames are saved and can be processed later."}
                report_note = None
                if final:
                    _, report_path = await write_report(session)
                    await send({"type": "report", "path": str(report_path or "")})
                    report_note = "shown on their screen and saved" + (f" as {report_path.name}" if report_path else "")
                totals = data.get("totals", {})
                places = data.get("places", [])
                return {
                    "report": report_note,
                    "objects_logged": [f"{i['description']}" for i in session.items][:30],
                    "objects_detected": {c: d.get("count", 1) for c, d in session.detected.items()},
                    "final": final and not data.get("still_reading"), "still_reading": data.get("still_reading"),
                    "places": [{"place": p["place"], "identified": len(p["books"])} for p in places],
                    "books_identified": totals.get("books_identified", 0),
                    "books_left_out": totals.get("books_excluded", 0),
                    "books_needing_appraisal": totals.get("books_needs_appraisal", 0),
                    "unreadable_spine_readings": data.get("unidentified_readings", 0),
                    "titles": [b["title"] for b in data.get("books", []) if not b.get("excluded")][:25],
                    "to_confirm": data.get("to_confirm", []),
                    "guardrail": "Counts only. Prices and book sizes are in the report on their screen; never state a "
                                 "price, size or money total.",
                }

            async def exclude_tool(args: dict) -> dict:
                """exclude_book: leave a book out of the report and totals, by title, or the book in view now."""
                if not vision_http:
                    return {"error": "The books agent is not running."}
                title, refs = str(args.get("title") or "").strip()[:200], []
                if not title:
                    # The book they mean is the one on camera: keep and queue the current frame so it is read.
                    if vision.get("last_frame"):
                        record = keep_frame(vision["last_frame"], vision.get("last_result") or {})
                        await queue_for_books(record)
                        refs = [record["id"]]
                    refs = [f["id"] for f in kept["frames"][-2:] if f["id"] not in refs] + refs
                try:
                    response = await vision_http.post(f"{WORKER_URL}/books/{session.sweep_id}/exclude", timeout=30.0, json={
                        "title": title, "frame_refs": refs, "undo": bool(args.get("undo")),
                        "reason": str(args.get("reason") or "")[:300]})
                    response.raise_for_status()
                    return response.json()
                except Exception as exc:
                    await worker_down(exc)
                    return {"error": "The books agent did not answer. Ask again in a moment."}

            async def copies_tool(args: dict) -> dict:
                """set_book_copies: how many copies of a book the policyholder has."""
                if not vision_http:
                    return {"error": "The books agent is not running."}
                try:
                    copies = int(args.get("copies") or 0)
                except (TypeError, ValueError):
                    return {"error": "Copies must be a whole number."}
                try:
                    response = await vision_http.post(f"{WORKER_URL}/books/{session.sweep_id}/copies", timeout=15.0,
                                                      json={"title": str(args.get("title") or "")[:200], "copies": copies})
                    response.raise_for_status()
                    return response.json()
                except Exception as exc:
                    await worker_down(exc)
                    return {"error": "The books agent did not answer. Ask again in a moment."}

            async def confirm_tool(args: dict) -> dict:
                """confirm_book: the policyholder's yes or no about a book the team suggested."""
                if not vision_http:
                    return {"error": "The books agent is not running."}
                try:
                    response = await vision_http.post(f"{WORKER_URL}/books/{session.sweep_id}/confirm", timeout=30.0, json={
                        "title": str(args.get("title") or "")[:200], "confirmed": bool(args.get("confirmed"))})
                    response.raise_for_status()
                    return response.json()
                except Exception as exc:
                    await worker_down(exc)
                    return {"error": "The books agent did not answer. Ask again in a moment."}

            async def room_tool(args: dict) -> dict:
                """measure_room: the frames of the next few seconds, held still on one wall's sheet, measured by the worker."""
                if not vision_http:
                    return {"error": "The measuring team is not running."}
                if args.get("ready") is not True:
                    return {"error": "Not measured: first ask them to point at the sheet, hold still and say ready; "
                                     "then call measure_room again with ready true."}
                if str(args.get("paper") or "").strip().lower() not in {"a4", "letter"}:
                    return {"error": "Not measured: ask whether the sheets are A4 or US Letter, then call again with that size."}
                wall = "2" if str(args.get("wall")) == "2" else "1"
                paper = "Letter" if str(args.get("paper") or "").lower().startswith("letter") else "A4"
                frames, last = [], None
                deadline = time.monotonic() + ROOM_HOLD_S
                while time.monotonic() < deadline and len(frames) < ROOM_HOLD_MAX_FRAMES:
                    frame = vision.get("last_frame")
                    if frame is not None and frame is not last:
                        last = frame
                        (sweep_dir / "room_frames").mkdir(parents=True, exist_ok=True)
                        ref = f"wall{wall}_{len(frames)}.jpg"        # a new hold on the same wall replaces these
                        (sweep_dir / "room_frames" / ref).write_bytes(frame)
                        frames.append({"frame_ref": ref, "path": str((sweep_dir / "room_frames" / ref).resolve())})
                    await asyncio.sleep(0.2)
                if not frames:
                    return {"error": "No camera frames arrived. Ask them to turn the camera on and try again."}
                sweep_id = session.sweep_id

                async def measure_wall():
                    """In the background, on its own client (the call may end first); a failure only shows in the report."""
                    try:
                        async with __import__("httpx").AsyncClient(timeout=180.0) as client:
                            response = await client.post(f"{WORKER_URL}/room/{sweep_id}/wall",
                                                         json={"wall": wall, "paper": paper, "frames": frames})
                            response.raise_for_status()
                    except Exception:
                        logger.warning("Room wall %s not measured for %s", wall, sweep_id, exc_info=True)

                room_measuring.append(asyncio.create_task(measure_wall()))     # keep a reference until it finishes
                if room_step["active"]:
                    room_step["walls"].add(wall)
                    if {"1", "2"} <= room_step["walls"]:
                        room_step["ending"] = "both walls captured"
                return {"wall": wall, "frames": len(frames), "captured": True,
                        "next": "Frames taken; the distance is measured in the background and only shown in the report. "
                                "Never say a size."}

            async def room_step_tool(args: dict) -> dict:
                """room_step: hold object notices while the room is measured; skip ends it early."""
                action = str(args.get("action") or "").lower()
                if action == "start":
                    room_step.update(active=True, started=time.monotonic(), held=[], walls=set(), ending=None)
                    return {"room_step": "started", "next": "Stay on measuring the room; object notices are held until it ends."}
                if action == "skip":
                    if room_step["active"]:
                        room_step["ending"] = "skipped"
                    return {"room_step": "skipped", "next": "The room is left unmeasured; go on to showing the items."}
                return {"error": "action must be start or skip"}

            session.analyse_tool, session.books_tool = analyse_tool, books_tool
            session.exclude_tool, session.copies_tool, session.confirm_tool = exclude_tool, copies_tool, confirm_tool
            session.room_tool, session.room_step_tool = room_tool, room_step_tool

            async def poll_books():
                """Relay the book agent's progress: new titles are spoken, running totals are context."""
                previous, announced = None, set()
                while True:
                    await asyncio.sleep(BOOK_POLL_S)
                    if not vision_http or not kept["frames"] or time.monotonic() < vision["down_until"]:
                        continue
                    try:
                        status = (await vision_http.get(f"{WORKER_URL}/books/{session.sweep_id}/status")).json()
                        code = country_code(session.country)
                        if (status.get("country"), status.get("currency")) != (code, session.currency) and code and session.currency:
                            status = (await vision_http.post(f"{WORKER_URL}/books/{session.sweep_id}/locale",
                                                             json={"country": code, "currency": session.currency})).json()
                    except Exception as exc:
                        await worker_down(exc)
                        continue
                    events = book_events(previous, status, announced)
                    previous = status
                    if events:
                        await feed(events)
                        await show_report_once(events)
                        await send({"type": "books", "books": status})

            async def deliver_notices():
                """Speak-worthy events go to the agent in a pause; progress goes in silently as context."""
                while True:
                    await asyncio.sleep(0.4)
                    if room_step["active"]:
                        if not room_step["ending"] and time.monotonic() - room_step["started"] > ROOM_STEP_MAX_S:
                            room_step["ending"] = "timed out"
                        # Objects and books seen while measuring the room: kept for one summary, not spoken now.
                        room_step["held"] += [e for e in queued_events if e["kind"] in ROOM_HELD_KINDS]
                        queued_events[:] = [e for e in queued_events if e["kind"] not in ROOM_HELD_KINDS]
                        if room_step["ending"]:
                            held, reason = room_step["held"], room_step["ending"]
                            room_step.update(active=False, held=[], ending=None)
                            text = (f"APP ROOM STEP: finished ({reason}). "
                                    + (f"While it ran the team logged: {describe(held)} Mention this briefly, then "
                                       if held else "Now ")
                                    + "ask them to show the things they want covered.")
                            voice["notice_at"] = time.monotonic()
                            await live_session.send_client_content(
                                turns=types.Content(role="user", parts=[types.Part(text=text)]), turn_complete=True)
                            continue
                    if not queued_events:
                        continue
                    now = time.monotonic()
                    speak = needs_reply(queued_events)
                    urgent = any(e["kind"] in PROMPT_KINDS for e in queued_events)
                    if speak and not ready_to_speak(now, urgent=urgent, **voice):
                        continue
                    batch = queued_events[:]
                    queued_events.clear()
                    if speak:
                        voice["notice_at"] = now
                    await live_session.send_client_content(
                        turns=types.Content(role="user", parts=[types.Part(text=notice_text(batch))]), turn_complete=speak)

            async def client_to_gemini():
                windows = {"text": deque(), "audio": deque(), "video": deque(), "camera_state": deque(), "locale": deque()}
                while True:
                    raw = await websocket.receive_text()
                    if len(raw) > MAX_MESSAGE_BYTES:
                        await send({"type": "error", "message": "Input exceeded the message size limit."})
                        continue
                    try:
                        message = json.loads(raw)
                        if not isinstance(message, dict):
                            raise ValueError("Expected a JSON object")
                        kind = message.get("type")
                        if kind == "close":
                            await finalize("Claimant")
                            await finalize("Agent")
                            return
                        if kind not in windows:
                            raise ValueError("Unknown input type")
                        now = time.monotonic()
                        window = windows[kind]
                        period, limit = (60, 20) if kind == "text" else (1, 100 if kind == "audio" else 5)
                        while window and now - window[0] >= period:
                            window.popleft()
                        if len(window) >= limit:
                            raise ValueError("Input rate limit reached; pause and try again")
                        window.append(now)
                        session.updated_at = now
                        if kind == "camera_state":
                            enabled = message.get("enabled")
                            if not isinstance(enabled, bool):
                                raise ValueError("Camera state must be true or false")
                            if set_camera_mode(session, enabled):
                                # A mode toggle updates context; the claimant's next spoken
                                # or typed description drives the response.
                                await live_session.send_client_content(
                                    turns=types.Content(role="user", parts=[types.Part(text=camera_mode_instruction(enabled))]),
                                    turn_complete=False,
                                )
                        elif kind == "locale":
                            before = (session.country, session.currency)
                            set_country(session, message.get("country", ""), message.get("currency", ""))
                            if (session.country, session.currency) != before:
                                await live_session.send_client_content(
                                    turns=types.Content(role="user", parts=[types.Part(text=locale_instruction(session.country, session.currency))]),
                                    turn_complete=False,
                                )
                            await send({"type": "state", "state": _ui_state(session)})
                        elif kind == "text":
                            text = message.get("text", "")
                            if not isinstance(text, str) or not text.strip() or len(text) > 8000:
                                raise ValueError("Text must contain 1–8000 characters")
                            turn_id = message.get("id") or uuid.uuid4().hex
                            if not isinstance(turn_id, str) or len(turn_id) > 100:
                                raise ValueError("Invalid turn identifier")
                            if any(t.get("id") == turn_id for t in session.transcript):
                                continue
                            append_turn(session, "Claimant", text, turn_id)
                            await send({"type": "transcript", "speaker": "Claimant", "text": text, "id": turn_id, "final": True})
                            await live_session.send_client_content(turns=types.Content(role="user", parts=[types.Part(text=text)]), turn_complete=True)
                        else:
                            encoded = message.get("data")
                            if not isinstance(encoded, str):
                                raise ValueError("Missing media data")
                            data = base64.b64decode(encoded, validate=True)
                            if not data or len(data) > (512000 if kind == "video" else 128000):
                                raise ValueError("Invalid media size")
                            if kind == "video":
                                if not data.startswith(b"\xff\xd8\xff"):
                                    raise ValueError("Camera frames must be JPEG images")
                                if set_camera_mode(session, True):
                                    await live_session.send_client_content(
                                        turns=types.Content(role="user", parts=[types.Part(text=camera_mode_instruction(True))]),
                                        turn_complete=False,
                                    )
                                session.last_frame = data
                                await live_session.send_realtime_input(video=types.Blob(data=data, mime_type="image/jpeg"))
                            else:
                                if len(data) % 2:
                                    raise ValueError("Audio must be PCM16")
                                await live_session.send_realtime_input(audio=types.Blob(data=data, mime_type="audio/pcm;rate=16000"))
                    except (ValueError, TypeError) as exc:
                        await send({"type": "error", "message": str(exc)})

            async def gemini_to_client():
                while True:
                    async for response in live_session.receive():
                        meta = getattr(response, "usage_metadata", None)
                        if meta:
                            # Recorded per message; live_tokens_seen keeps the highest running count in case
                            # the API reports session totals rather than per-turn counts.
                            session.usage.add_call("gemini_live", input_tokens=meta.prompt_token_count or 0,
                                                   output_tokens=meta.response_token_count or 0)
                            session.live_tokens_seen = max(session.live_tokens_seen, meta.total_token_count or 0)
                        if response.tool_call and response.tool_call.function_calls:
                            await finalize("Claimant")
                            for fc in response.tool_call.function_calls:
                                await launch_tool(fc, live_session)
                        if response.tool_call_cancellation:
                            for call_id in response.tool_call_cancellation.ids or []:
                                task = tool_tasks.get(str(call_id))
                                if task:
                                    task.cancel()
                        content = response.server_content
                        if not content:
                            continue
                        # Clear queued voice/video before forwarding any more content.
                        # An interrupted payload can still contain cancelled output.
                        if content.interrupted:
                            voice["agent_speaking"], voice["agent_done_at"] = False, time.monotonic()
                            await send({"type": "interrupted"})
                            await finalize("Agent")
                        for speaker, chunk in (("Claimant", content.input_transcription), ("Agent", content.output_transcription)):
                            if speaker == "Agent" and content.interrupted:
                                continue
                            if chunk and chunk.text:
                                if speaker == "Claimant":
                                    voice["user_voice_at"] = time.monotonic()
                                if speaker == "Agent":
                                    await finalize("Claimant")
                                if chunk.text != pending[speaker]["text"] or speaker == "Claimant":
                                    pending[speaker]["text"] += chunk.text
                                await send({"type": "transcript", "speaker": speaker, **pending[speaker], "final": False})
                            if chunk and getattr(chunk, "finished", False):
                                await finalize(speaker)
                        if content.model_turn and not content.interrupted:
                            for part in content.model_turn.parts or []:
                                if part.inline_data:
                                    media = audio_message(part.inline_data)
                                    if media:
                                        voice["agent_speaking"] = True
                                        await finalize("Claimant")
                                        await send(media)
                        if getattr(content, "turn_complete", False):
                            voice["agent_speaking"], voice["agent_done_at"] = False, time.monotonic()
                            await finalize("Claimant")
                            await finalize("Agent")
                            await send({"type": "turn_complete"})

            pair = [track(asyncio.create_task(client_to_gemini())), track(asyncio.create_task(gemini_to_client())),
                    track(asyncio.create_task(deliver_notices())), track(asyncio.create_task(poll_books()))]
            try:
                done, _ = await asyncio.wait(pair, timeout=20 * 60, return_when=asyncio.FIRST_COMPLETED)
                if not done:
                    await send({"type": "error", "message": "The live connection reached 20 minutes. Reconnect to continue this intake."})
                for task in done:
                    task.result()
            finally:
                session.on_frame = session.analyse_tool = session.books_tool = session.exclude_tool = None
                session.copies_tool = session.confirm_tool = session.room_tool = session.room_step_tool = None
                if vision_http:
                    await vision_http.aclose()
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("Gemini Live session failed")
        with contextlib.suppress(Exception):
            await send({"type": "error", "message": "Live connection ended. Reconnect to continue this intake."})
    finally:
        for task in list(tasks):
            task.cancel()
        await asyncio.gather(*list(tasks), return_exceptions=True)
        for item in session.tool_activity:
            if item["phase"] == "running":
                item.update(phase="cancelled", headline="Connection ended")
        session.live_socket = None
        set_camera_mode(session, False)
        session.updated_at = time.monotonic()
        with contextlib.suppress(Exception):
            await websocket.close()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(DEMO_DIR / "index.html")


@app.get("/index.html")
def index_alias():
    return FileResponse(DEMO_DIR / "index.html")


@app.get("/app.js")
def javascript():
    return FileResponse(DEMO_DIR / "app.js", media_type="text/javascript")


@app.get("/styles.css")
def styles():
    return FileResponse(DEMO_DIR / "styles.css", media_type="text/css")

