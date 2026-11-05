"""books_agent: Google ADK SequentialAgent that identifies and prices the books in a batch of frames.

Steps in a Google ADK
SequentialAgent, passing results through ADK session.state. All steps are deterministic Python
except the model match, which only adds to what the rules found and does nothing if it fails.
The live worker runs it on batches of frames in which Grounding DINO detected a book.

    ① ReadSpines     Python    docTR OCR, four orientations             -> state["readings"]
    ② SearchMarket   Python    top-10 catalogue results per reading     -> state["search_results"]
    ③ MatchRules     Python    score_match picks; accepted as they are  -> state["matches"], state["verified"]
    ④ MatchModel     LlmAgent  readings the rules left: the model picks, -> state["verified"] (added to)
                     + verify  and each pick must be supported by the words read; unsupported picks
                               are only "suggested". Skipped when nothing is left, the model is off
                               (BOOKS_MATCHER=rules) or paused; on a model failure it adds nothing.
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
from rapidfuzz import fuzz
from typing_extensions import override

from .catalog import (_author_read, _with_joins, beyond_title, first_publish_year, plausible, score_match, search,
                      significant, title_coverage, tokens)
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
SAME_READING = 85          # token_sort_ratio: two readings of one spine (word order or a misread letter apart)
FALLBACK_CONFIDENCE = 0.9  # search retries with fewer words only above this OCR confidence
QUOTA_PAUSE_S = 120        # after a quota error, skip the model this long instead of waiting on failing calls
_model_paused_until = 0.0


# ---- ADK plumbing ---------------------------------------------------------------------------
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
    """④ MatchModel: the model picks for the readings the rules could not match, and each pick is
    checked against the words read (verify). Rules' matches are never touched; if the model is
    off, paused or fails, nothing is added and nothing is verified."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    verify: Callable[[dict, list[dict]], dict]

    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        global _model_paused_until
        state = ctx.session.state
        left = state.get("left_for_model", [])
        if not left or not self.sub_agents or time.monotonic() < _model_paused_until:
            why = ("nothing left for the model" if not left else "model off (BOOKS_MATCHER=rules)" if not self.sub_agents
                   else "model paused after a quota error")
            if left and self.sub_agents:
                state["matches"]["by"] = "rules (model paused after quota error)"
            yield _state_event(self.name, f"Skipped: {why}.", {"matches": state["matches"]})
            return
        picks, error = None, None
        with stage(self.name):
            for attempt in range(2):   # one retry: Gemini's transient server errors usually clear in seconds
                try:
                    async for event in self.sub_agents[0].run_async(ctx):
                        if event.usage_metadata:
                            record_call("gemini_matcher", input_tokens=event.usage_metadata.prompt_token_count or 0,
                                        output_tokens=event.usage_metadata.candidates_token_count or 0)
                        yield event
                    raw = state.get("llm_matches") or {}
                    allowed = {r["id"] for r in left}
                    picks = [m for m in (raw.get("matches", []) if isinstance(raw, dict) else [])
                             if m.get("work_key") and m.get("reading_id") in allowed]
                    break
                except Exception as exc:   # quota, network, malformed output: keep what the rules found
                    error = exc
                    record_call("gemini_matcher", ok=False)
                    if "ResourceExhausted" in type(exc).__name__ or "429" in str(exc):
                        _model_paused_until = time.monotonic() + QUOTA_PAUSE_S
                    if attempt == 0 and "ServerError" in type(exc).__name__:
                        await asyncio.sleep(2)
                        continue
                    break
        if picks is None:
            state["matches"]["by"] = f"rules (model failed: {type(error).__name__})"
            yield _state_event(self.name, "Model failed; the rules' matches stand.", {"matches": state["matches"]})
            return
        checked = self.verify(state, picks)
        verified = state["verified"]
        verified["accepted"] += [b for b in checked["accepted"] if all(b["work_key"] != a["work_key"] for a in verified["accepted"])]
        verified["suggested"] += checked["suggested"]
        picked = {m["reading_id"] for m in picks}
        verified["unread"] = [u for u in verified["unread"] if u["reading_id"] not in picked]
        state["matches"]["matches"] += picks
        state["matches"]["by"] = f"rules+{MODEL}"
        yield _state_event(self.name, f"Model picked {len(picks)}; {len(checked['accepted'])} supported by the words read.",
                           {"matches": state["matches"], "verified": verified})


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
        # used to crowd out the readings that do. Single strips before pairs of neighbouring strips:
        # a pair helps only when a spine's title and author are on separate lines, and on a leaning
        # shelf most pairs join two different books; ranked first, they pushed clean readings past
        # MAX_READINGS.
        ranked = sorted(candidates, key=lambda c: (len(significant(tokens(c["text"]))) < 2, c.get("strips", 1) > 1,
                                                   -c["ocr_confidence"]))
        for cand in ranked:
            key = " ".join(tokens(cand["text"]))
            if cand["ocr_confidence"] < MIN_OCR_CONFIDENCE or not plausible(cand["text"]):
                continue
            # The same spine read in another frame, with the words in another order or a letter
            # misread, is the same reading too.
            key = next((k for k in readings if k == key or fuzz.token_sort_ratio(k, key) >= SAME_READING), key)
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
        return out

    def line(state: dict, item: dict, doc: dict, match: dict, by: str, coverage: float, author_hit: bool) -> dict:
        reading = next(r for r in state.get("readings", []) if r["id"] == item["id"])
        return {"work_key": doc["key"], "title": doc.get("title", ""), "subtitle": doc.get("subtitle", ""),
                "author": ", ".join(doc.get("author_name", [])[:3]), "first_publish_year": doc.get("first_publish_year"),
                "reading": item["text"], "frame_refs": reading["frame_refs"],
                "evidence": match.get("evidence", ""), "title_coverage": round(coverage, 2),
                "author_hit": author_hit, "matched_by": by, "published_year_seen": doc.get("published_year_seen"),
                "catalogue": {"source": doc.get("source", "Open Library"), "url": doc.get("url"),
                              "query": item["query"], "retrieved_at": item["retrieved_at"]}}

    def match_rules(state: dict) -> dict:
        """③ Readings of books seen earlier keep that book; the rest are matched by score_match. These
        matches are accepted as they are (score_match already required the title and the author or
        a distinctive title in the words read). Readings with results but no match are left for the model."""
        everything = state.get("search_results", [])
        accepted, matches, left = [], [], []
        for item in everything:
            if not item["results"]:
                continue
            scored = [(score_match(item["text"], r["_doc"]), r) for r in item["results"]]
            ok = [(sc, r) for sc, r in scored if sc["accepted"]]
            if not ok:
                left.append(item)
                continue
            sc, r = max(ok, key=lambda x: x[0]["score"])
            by = "seen earlier" if item.get("known") else "rules"
            match = {"reading_id": item["id"], "work_key": r["work_key"], "evidence": item["text"], "by": by}
            matches.append(match)
            if all(b["work_key"] != r["work_key"] for b in accepted):
                accepted.append(line(state, item, r["_doc"], match, by, sc["title_coverage"], sc["author_hit"]))
        found = [m["by"] for m in matches]
        state["matches"] = {"matches": matches, "by": "none" if not any(i["results"] for i in everything)
                            else "seen earlier" if found and all(b == "seen earlier" for b in found) and not left else "rules"}
        state["left_for_model"] = left
        state["match_request"] = json.dumps([{"reading_id": r["id"], "reading": r["text"],
                                              "results": [{k: v for k, v in x.items() if k != "_doc"} for x in r["results"]]}
                                             for r in left], ensure_ascii=False)
        matched = {m["reading_id"] for m in matches}
        unread = [{"reading_id": r["id"], "text": r["text"], "confidence": r["confidence"], "frame_refs": r["frame_refs"]}
                  for r in state.get("readings", []) if r["id"] not in matched and r["confidence"] >= 0.95]
        return {"accepted": accepted, "suggested": [], "unread": unread}

    def verify_picks(state: dict, picks: list[dict]) -> dict:
        """④ The model's picks must be supported by words actually read in this batch: most of the
        (main) title, and the author or a title of three or more words. Looser than score_match,
        which already rejected these readings, but the words must be there."""
        pool = _with_joins([t for r in state.get("readings", []) for t in tokens(r["text"])])
        by_id = {r["id"]: r for r in state.get("search_results", [])}
        accepted, suggested = [], []
        for m in picks:
            item = by_id.get(m.get("reading_id"))
            doc = next((r["_doc"] for r in (item or {}).get("results", []) if r["work_key"] == m.get("work_key")), None)
            if not item or not doc:
                continue
            # Catalogue titles often carry a subtitle the spine does not ("Spark: The Definitive
            # Guide: Big Data Processing Made Simple"); judge the main title as well, keep the best.
            title = doc.get("title", "")
            variants = [tokens(title)] + [tokens(title[:i]) for i, ch in enumerate(title) if ch == ":"]
            coverage, title_words = max(((title_coverage(v, pool), significant(v)) for v in variants if significant(v)),
                                        key=lambda x: (x[0], len(x[1])), default=(0.0, []))
            author_hit = _author_read(doc.get("author_name", []), beyond_title(pool, tokens(title)), full_title=coverage >= 0.999)
            supported = coverage >= 0.6 and (author_hit or len(set(title_words)) >= 3)
            target = accepted if supported else suggested
            if all(x["work_key"] != doc["key"] for x in accepted + suggested):
                target.append(line(state, item, doc, m, MODEL, coverage, author_hit))
        return {"accepted": accepted, "suggested": suggested}

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

    return {"read": read_spines, "search": search_market, "rules": match_rules, "verify": verify_picks, "price": price_books}


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
            FunctionNode(name="MatchRules", handler=h["rules"], output_key="verified", summary="Matched by the rules."),
            MatchStep(name="MatchModel", verify=h["verify"], sub_agents=[matcher] if matcher else []),
            FunctionNode(name="PriceBooks", handler=h["price"], output_key="priced", summary="Priced new books."),
        ],
    )


async def run_books_agent(agent: SequentialAgent, frames: list[dict], country: str, currency: str) -> dict:
    """Run books_agent on a batch of frames ([{ref, path}]) with a fresh session. Batching keeps up with the camera: OCR runs per frame, but
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
