"""books_agent: Google ADK SequentialAgent that identifies and prices the books in a batch of frames.

Same design as the reference app's claim team (reference_app/agent.py): steps in a
SequentialAgent, passing results through ADK session.state. All steps are deterministic Python
except the match step, which uses an LLM and falls back to the rules on any model failure.
The live worker runs it on batches of frames in which Grounding DINO detected a book.

    ① ReadSpines     Python    docTR OCR, three orientations           -> state["readings"]
    ② SearchMarket   Python    top-10 catalogue results per reading     -> state["search_results"]
    ③ MatchBooks     LlmAgent  picks the closest of the top-10 results  -> state["matches"]
                               (rules keep their confident matches; if the model fails, rules decide;
                               BOOKS_MATCHER=rules: rules only)
    ④ VerifyMatches  Python    words actually read must support the pick -> state["verified"]
    ⑤ PriceBooks     Python    marketplace price by country, converted  -> state["priced"]

The LLM never produces a price or a total, and a pick that the OCR text does not support is
only "suggested" (review queue), never counted.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import os
import time
import uuid
from typing import Any, AsyncGenerator, Callable

from google.adk.agents import BaseAgent, LlmAgent, SequentialAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import override

from .catalog import _author_read, _found, _with_joins, first_publish_year, plausible, score_match, search, significant, tokens
from .pricing import APPRAISAL_YEAR
from .spine_ocr import candidates_from_lines
from .telemetry import record_call, stage

APP_NAME = "books_agent"
# By default the LLM picks the closest of the top-10 search results (one model call per batch of
# frames); if the model fails, the deterministic rules decide. BOOKS_MATCHER=rules: rules only.
MATCHER = os.getenv("BOOKS_MATCHER", "gemini")
MODEL = os.getenv("BOOKS_AGENT_MODEL", "gemini-3.6-flash")
MAX_READINGS = 40          # per batch of frames; readings with two or more real words first, then clearest
MIN_OCR_CONFIDENCE = 0.6
FALLBACK_CONFIDENCE = 0.9  # search retries with fewer words only above this OCR confidence
QUOTA_PAUSE_S = 120        # after a quota error, skip the model this long instead of waiting on failing calls
_model_paused_until = 0.0


# ---- ADK plumbing (same pattern as reference_app/agent.py) --------------------------------
def _state_event(author: str, text: str, updates: dict[str, Any]) -> Event:
    return Event(author=author, content=genai_types.Content(role="model", parts=[genai_types.Part(text=text)]),
                 actions=EventActions(state_delta=updates))


class FunctionNode(BaseAgent):
    """Deterministic step that reads and writes session state. Runs its handler in a thread,
    because OCR and catalogue lookups block."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    handler: Callable[[dict], Any]
    output_key: str
    summary: str

    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        before = dict(ctx.session.state)
        with stage(self.name):
            result = await asyncio.to_thread(self.handler, ctx.session.state)
        ctx.session.state[self.output_key] = result
        # Persist every key the handler wrote (for example the LLM step's request), not only output_key.
        delta = {k: v for k, v in ctx.session.state.items() if k not in before or before[k] is not v}
        yield _state_event(self.name, self.summary, delta)


class Match(BaseModel):
    reading_id: int
    work_key: str = Field(default="", description="work_key of the chosen search result, or empty if none is the same book")
    evidence: str = Field(default="", description="the OCR words that show it is this book")


class MatchResult(BaseModel):
    matches: list[Match] = Field(default_factory=list)


MATCH_INSTRUCTION = """You identify books from OCR text read off spines and covers, for an insurance inventory.
OCR text is noisy: words can be cut off ("TARTUP" for "STARTUP"), misspelt, merged with a
neighbouring book, or mirrored gibberish from reading the image in the wrong direction.

For each reading below you get the top catalogue search results. Pick the result that is the
same book as the reading, and give the OCR words that show it. Choose a result only when the
title words (allowing for OCR errors) or the author name are visible in the reading. If no
result fits, or the reading is gibberish, leave work_key empty. Never guess from popularity.
Edition does not matter; any result for the same work is fine.

Readings and search results (JSON):
{match_request}
"""


class MatchStep(BaseAgent):
    """③ MatchBooks: the rule-based scorer picks (default, no model calls). With an LLM sub-agent
    (BOOKS_MATCHER=gemini) the model picks, and the rules take over if the model is unavailable."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        everything = ctx.session.state.get("search_results", [])
        if not any(r["results"] for r in everything):
            yield _state_event(self.name, "Nothing to match.", {"matches": {"matches": [], "by": "none"}})
            return
        # Readings of books already identified in this sweep keep that match; only new ones are matched.
        seen = [{"reading_id": r["id"], "work_key": r["results"][0]["work_key"], "evidence": r["text"]}
                for r in everything if r.get("known")]
        request = [r for r in everything if not r.get("known")]
        global _model_paused_until
        if not any(r["results"] for r in request) or not self.sub_agents or time.monotonic() < _model_paused_until:
            by = ("seen earlier" if not any(r["results"] for r in request) else
                  "rules" if not self.sub_agents else "rules (model paused after quota error)")
            matches = {"matches": seen + _rule_matches(request), "by": by}
            ctx.session.state["matches"] = matches
            yield _state_event(self.name, f"Matched by {by}.", {"matches": matches})
            return
        matches = None
        timer = stage(self.name)
        timer.__enter__()
        for attempt in range(2):   # one retry: Gemini's transient server errors usually clear in seconds
            try:
                async for event in self.sub_agents[0].run_async(ctx):
                    if event.usage_metadata:
                        record_call("gemini_matcher", input_tokens=event.usage_metadata.prompt_token_count or 0,
                                    output_tokens=event.usage_metadata.candidates_token_count or 0)
                    yield event
                raw = ctx.session.state.get("llm_matches") or {}
                picked = [m for m in (raw.get("matches", []) if isinstance(raw, dict) else []) if m.get("work_key")]
                # The model decides readings the rules could not; matches the rules already accept are
                # kept even if the model skipped them, so adding the model never loses a book.
                decided = {m["reading_id"] for m in picked}
                kept = [m for m in _rule_matches(request) if m["reading_id"] not in decided]
                matches = {"matches": picked + kept, "by": MODEL + ("+rules" if kept else "")}
                break
            except Exception as exc:   # quota, network, malformed output: keep the sweep going
                error = exc
                record_call("gemini_matcher", ok=False)
                if "ResourceExhausted" in type(exc).__name__ or "429" in str(exc):
                    _model_paused_until = time.monotonic() + QUOTA_PAUSE_S
                if attempt == 0 and "ServerError" in type(exc).__name__:
                    await asyncio.sleep(2)
                    continue
                break
        timer.__exit__(None, None, None)
        if matches is None:
            matches = {"matches": _rule_matches(request), "by": f"rules (model failed: {type(error).__name__})"}
        matches["matches"] = seen + matches["matches"]
        ctx.session.state["matches"] = matches
        yield _state_event(self.name, f"Matched by {matches['by']}.", {"matches": matches})


def _rule_matches(request: list[dict]) -> list[dict]:
    """Fallback matcher: the existing title/author coverage rules."""
    out = []
    for item in request:
        scored = [(score_match(item["text"], r["_doc"]), r) for r in item["results"]]
        accepted = [(s, r) for s, r in scored if s["accepted"]]
        if accepted:
            s, r = max(accepted, key=lambda x: x[0]["score"])
            out.append({"reading_id": item["id"], "work_key": r["work_key"], "evidence": item["text"]})
    return out


# ---- step handlers ------------------------------------------------------------------------
def make_handlers(reader: Callable[[], Any], price: Callable[[dict, dict], dict | None],
                  known: Callable[[], list[dict]] | None = None) -> dict[str, Callable]:
    # One agent per sweep (BookJob), so this memo lives for one sweep and never reaches another.
    search_memo: dict = {}
    year_memo: dict[str, int | None] = {}

    def read_spines(state: dict) -> list[dict]:
        """OCR every frame in the batch. The same reading in several frames (a slow pan) is one
        reading that remembers all its frames, so it is searched and matched once."""
        from PIL import Image

        candidates, ocr_lines = [], {}
        for frame in state["frames"]:
            with Image.open(frame["path"]) as img:
                ocr_lines[frame["ref"]] = reader().read(img.convert("RGB"))
            candidates += candidates_from_lines(ocr_lines[frame["ref"]], frame["ref"])
        state["ocr_lines"] = ocr_lines
        readings: dict[str, dict] = {}
        # Two or more real words first: a one-word reading ("HOLDER") rarely identifies a book and
        # used to crowd out the readings that do.
        for cand in sorted(candidates, key=lambda c: (len(significant(tokens(c["text"]))) < 2, -c["ocr_confidence"])):
            key = " ".join(tokens(cand["text"]))
            if cand["ocr_confidence"] < MIN_OCR_CONFIDENCE or not plausible(cand["text"]):
                continue
            if key in readings:
                readings[key]["frame_refs"] = sorted(set(readings[key]["frame_refs"]) | {cand["frame_ref"]})
            elif len(readings) < MAX_READINGS:
                readings[key] = {"id": len(readings), "text": cand["text"], "confidence": cand["ocr_confidence"],
                                 "strips": cand.get("strips", 1), "frame_refs": [cand["frame_ref"]]}
        return list(readings.values())

    def search_market(state: dict) -> list[dict]:
        from concurrent.futures import ThreadPoolExecutor

        readings = [r for r in state.get("readings", []) if len(significant(tokens(r["text"]))) >= 2]
        # Books already identified in this sweep (catalogue-shaped). A reading the rules accept as
        # one of them needs no search and no model call: the camera passes the same spines many times.
        seen = known() if known else []

        def result_of(d: dict) -> dict:
            return {"work_key": d["key"], "title": d.get("title", ""), "author": ", ".join(d.get("author_name", [])[:2]),
                    "first_published": d.get("first_publish_year"), "_doc": d}

        def one(reading: dict) -> dict:
            for doc in seen:
                if score_match(reading["text"], doc)["accepted"]:
                    return {"id": reading["id"], "text": reading["text"], "query": "(identified earlier in this sweep)",
                            "retrieved_at": doc.get("retrieved_at", ""), "results": [result_of(doc)], "known": True}
            # Retrying with fewer words costs several extra queries; spend them only on confident
            # readings, where a single misread word is the likely reason nothing was found.
            try:
                docs, query, retrieved_at = search(reading["text"], limit=10, fallback=reading["confidence"] >= FALLBACK_CONFIDENCE,
                                                    memo=search_memo)
            except Exception as exc:   # a failed search loses this reading only, not the whole batch
                return {"id": reading["id"], "text": reading["text"], "query": "", "retrieved_at": "", "results": [],
                        "search_error": type(exc).__name__}
            return {"id": reading["id"], "text": reading["text"], "query": query, "retrieved_at": retrieved_at,
                    "results": [result_of(d) for d in docs]}

        # Each search runs in a copy of this context, so its requests count toward this sweep's usage.
        jobs = [(contextvars.copy_context(), r) for r in readings]
        with ThreadPoolExecutor(max_workers=6) as pool:
            out = list(pool.map(lambda job: job[0].run(one, job[1]), jobs))
        state["match_request"] = json.dumps([{"reading_id": r["id"], "reading": r["text"],
                                              "results": [{k: v for k, v in x.items() if k != "_doc"} for x in r["results"]]}
                                             for r in out if r["results"] and not r.get("known")], ensure_ascii=False)
        return out

    def verify_matches(state: dict) -> dict:
        """The pick must be supported by words actually read: title words, and the author or a distinctive title."""
        pool = _with_joins([t for r in state.get("readings", []) for t in tokens(r["text"])])
        by_id = {r["id"]: r for r in state.get("search_results", [])}
        readings_by_id = {r["id"]: r for r in state.get("readings", [])}
        accepted, suggested = [], []
        for m in state.get("matches", {}).get("matches", []):
            item = by_id.get(m.get("reading_id"))
            doc = next((r["_doc"] for r in (item or {}).get("results", []) if r["work_key"] == m.get("work_key")), None)
            if not item or not doc:
                continue
            # Catalogue titles often carry a subtitle the spine does not ("Spark: The Definitive
            # Guide: Big Data Processing Made Simple"); judge the main title as well, keep the best.
            title = doc.get("title", "")
            variants = [title] + [title[:i] for i, ch in enumerate(title) if ch == ":"]
            scored = []
            for variant in variants:
                words = significant(tokens(variant))
                if words:
                    scored.append((sum(_found(w, pool) for w in words) / len(words), words))
            coverage, title_words = max(scored, key=lambda x: (x[0], len(x[1])), default=(0.0, []))
            author_hit = _author_read(doc.get("author_name", []), pool, full_title=coverage >= 0.999)
            line = {"work_key": doc["key"], "title": doc.get("title", ""), "subtitle": doc.get("subtitle", ""),
                    "author": ", ".join(doc.get("author_name", [])[:3]), "first_publish_year": doc.get("first_publish_year"),
                    "reading": item["text"], "frame_refs": readings_by_id[item["id"]]["frame_refs"],
                    "evidence": m.get("evidence", ""), "title_coverage": round(coverage, 2),
                    "author_hit": author_hit, "matched_by": state["matches"].get("by"),
                    "published_year_seen": doc.get("published_year_seen"),
                    "catalogue": {"source": doc.get("source", "Open Library"), "url": doc.get("url"),
                                  "query": item["query"], "retrieved_at": item["retrieved_at"]}}
            supported = coverage >= 0.6 and (author_hit or len(title_words) >= 3)
            target = accepted if supported else suggested
            if all(x["work_key"] != line["work_key"] for x in accepted + suggested):
                target.append(line)
        matched = {m.get("reading_id") for m in state.get("matches", {}).get("matches", []) if m.get("work_key")}
        unread = [{"text": r["text"], "confidence": r["confidence"], "frame_refs": r["frame_refs"]} for r in state.get("readings", [])
                  if r["id"] not in matched and r["confidence"] >= 0.95]
        return {"accepted": accepted, "suggested": suggested, "unread": unread}

    def price_books(state: dict) -> list[dict]:
        out = []
        for book in state.get("verified", {}).get("accepted", []):
            if book["first_publish_year"] is None:
                # Google Books has edition dates only; the antiquarian check needs the work's first
                # appearance. An edition seen before 1900 already proves the work is that old.
                if book["work_key"] not in year_memo:
                    seen = book.get("published_year_seen")
                    year_memo[book["work_key"]] = first_publish_year(book["title"], book["author"]) or (
                        seen if seen and seen < APPRAISAL_YEAR else None)
                book["first_publish_year"] = year_memo[book["work_key"]]
            result = price(book, state)
            if result is not None:
                out.append({"work_key": book["work_key"], **result})
        return out

    return {"read": read_spines, "search": search_market, "verify": verify_matches, "price": price_books}


def create_books_agent(reader: Callable[[], Any], price: Callable[[dict, dict], dict | None],
                       matcher: BaseAgent | None = None, known: Callable[[], list[dict]] | None = None) -> SequentialAgent:
    """Build the agent for one sweep (its search memo is not shared with other sweeps). Matching is rule-based unless BOOKS_MATCHER=gemini; `matcher` replaces the
    LLM step (tests use a fake that writes llm_matches)."""
    h = make_handlers(reader, price, known)
    if matcher is None and MATCHER == "gemini":
        matcher = LlmAgent(
            name="MatchBooksLLM", model=MODEL, instruction=MATCH_INSTRUCTION,
            description="Picks the catalogue result that is the same book as each OCR reading.",
            output_schema=MatchResult, output_key="llm_matches",
            disallow_transfer_to_parent=True, disallow_transfer_to_peers=True,
            generate_content_config=genai_types.GenerateContentConfig(temperature=0),
        )
    return SequentialAgent(
        name="books_agent",
        description="Reads spines in a batch of frames, finds the books in the catalogue, verifies and prices them.",
        sub_agents=[
            FunctionNode(name="ReadSpines", handler=h["read"], output_key="readings", summary="Read spine text."),
            FunctionNode(name="SearchMarket", handler=h["search"], output_key="search_results", summary="Searched the catalogue."),
            MatchStep(name="MatchBooks", sub_agents=[matcher] if matcher else []),
            FunctionNode(name="VerifyMatches", handler=h["verify"], output_key="verified", summary="Checked picks against the OCR text."),
            FunctionNode(name="PriceBooks", handler=h["price"], output_key="priced", summary="Priced new books."),
        ],
    )


async def run_books_agent(agent: SequentialAgent, frames: list[dict], country: str, currency: str) -> dict:
    """Run books_agent on a batch of frames ([{ref, path}]) with a fresh session, like
    run_claim_workflow in agent.py. Batching keeps up with the camera: OCR runs per frame, but
    search and the model call run once for the whole batch."""
    service = InMemorySessionService()
    session_id = f"batch-{frames[0]['ref']}-{uuid.uuid4().hex[:6]}"
    await service.create_session(app_name=APP_NAME, user_id="live-worker", session_id=session_id,
                                 state={"frames": frames, "country": country, "currency": currency})
    runner = Runner(app_name=APP_NAME, agent=agent, session_service=service)
    refs = ", ".join(f["ref"] for f in frames)
    message = genai_types.Content(role="user", parts=[genai_types.Part(text=f"Identify and price the books in frames {refs}.")])
    async for _ in runner.run_async(user_id="live-worker", session_id=session_id, new_message=message):
        pass
    state = (await service.get_session(app_name=APP_NAME, user_id="live-worker", session_id=session_id)).state
    return {k: state.get(k) for k in ("ocr_lines", "readings", "matches", "verified", "priced")}
