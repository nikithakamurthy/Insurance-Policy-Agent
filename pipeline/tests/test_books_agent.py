import tempfile
import unittest
from pathlib import Path
from typing import AsyncGenerator
from unittest.mock import patch

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from PIL import Image
from pydantic import ConfigDict
from typing_extensions import override

from pipeline import books_agent

LEAN = {"key": "/works/OL1W", "title": "The Lean Startup", "author_name": ["Eric Ries"], "first_publish_year": 2011}
OTHER = {"key": "/works/OL2W", "title": "Lean Analytics", "author_name": ["Alistair Croll"], "first_publish_year": 2013}


class FakeReader:
    def read(self, image):
        line = {"text": "THE LEAN STARTUP ERIC RIES", "box": [0.1, 0.4, 0.9, 0.5], "confidence": 1.0}
        return {"horizontal": [line], "top_to_bottom": [], "bottom_to_top": []}


class FakeMatcher(BaseAgent):
    """Stands in for the LlmAgent: writes llm_matches like the real one, or fails."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    picks: list = []
    fail: bool = False

    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        if self.fail:
            raise RuntimeError("quota exhausted")
        result = {"matches": self.picks}
        ctx.session.state["llm_matches"] = result
        yield Event(author=self.name, actions=EventActions(state_delta={"llm_matches": result}))


def fake_search(text, limit=10, fallback=True, memo=None):
    return [LEAN, OTHER], "lean startup eric ries", "2026-11-02T12:00:00+0100"


class BooksAgentTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.frame = Path(self.tmp.name) / "f00001.jpg"
        Image.new("RGB", (64, 64)).save(self.frame)
        self.priced = []
        self.patch = patch.object(books_agent, "search", fake_search)
        self.patch.start()

    async def asyncTearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def price(self, book, state):
        self.priced.append(book["title"])
        return {"status": "identified", "replacement_cost": {"amount": 9.41, "currency": state["currency"]}}

    async def run_with(self, matcher):
        agent = books_agent.create_books_agent(FakeReader, self.price, matcher=matcher)
        return await books_agent.run_books_agent(agent, [{"ref": "f00001", "path": str(self.frame)}], "DE", "EUR")

    async def test_book_seen_earlier_needs_no_search_or_model_call(self):
        searched = []

        def counting_search(text, limit=10, fallback=True, memo=None):
            searched.append(text)
            return fake_search(text)
        matcher = FakeMatcher(name="M", picks=[], fail=True)          # would fail if the model were called
        agent = books_agent.create_books_agent(FakeReader, self.price, matcher=matcher, known=lambda: [LEAN])
        with patch.object(books_agent, "search", counting_search):
            result = await books_agent.run_books_agent(agent, [{"ref": "f00001", "path": str(self.frame)}], "DE", "EUR")
        self.assertEqual(searched, [])
        self.assertEqual(result["matches"]["by"], "seen earlier")
        self.assertEqual([b["title"] for b in result["verified"]["accepted"]], ["The Lean Startup"])

    async def test_supported_pick_is_accepted_and_priced(self):
        result = await self.run_with(FakeMatcher(name="M", picks=[{"reading_id": 0, "work_key": "/works/OL1W", "evidence": "LEAN STARTUP RIES"}]))
        self.assertEqual([b["title"] for b in result["verified"]["accepted"]], ["The Lean Startup"])
        self.assertEqual(result["verified"]["accepted"][0]["frame_refs"], ["f00001"])
        self.assertEqual(self.priced, ["The Lean Startup"])
        self.assertEqual(result["priced"][0]["replacement_cost"]["amount"], 9.41)

    async def test_pick_not_supported_by_ocr_is_only_suggested(self):
        result = await self.run_with(FakeMatcher(name="M", picks=[{"reading_id": 0, "work_key": "/works/OL2W", "evidence": "LEAN"}]))
        self.assertEqual(result["verified"]["accepted"], [])
        self.assertEqual([b["title"] for b in result["verified"]["suggested"]], ["Lean Analytics"])
        self.assertEqual(self.priced, [])                                   # suggestions are never priced

    async def test_batch_merges_the_same_reading_across_frames(self):
        second = Path(self.tmp.name) / "f00002.jpg"
        Image.new("RGB", (64, 64)).save(second)
        agent = books_agent.create_books_agent(FakeReader, self.price, matcher=FakeMatcher(
            name="M", picks=[{"reading_id": 0, "work_key": "/works/OL1W", "evidence": "LEAN STARTUP RIES"}]))
        result = await books_agent.run_books_agent(agent, [{"ref": "f00001", "path": str(self.frame)},
                                                           {"ref": "f00002", "path": str(second)}], "DE", "EUR")
        self.assertEqual(len(result["readings"]), 1)                        # same text on both frames: searched once
        self.assertEqual(result["verified"]["accepted"][0]["frame_refs"], ["f00001", "f00002"])
        self.assertEqual(sorted(result["ocr_lines"]), ["f00001", "f00002"])

    async def test_rules_only_mode_makes_no_model_call(self):
        with patch.object(books_agent, "MATCHER", "rules"):
            agent = books_agent.create_books_agent(FakeReader, self.price)
        self.assertEqual(agent.sub_agents[2].sub_agents, [])                  # no LlmAgent in the match step
        result = await books_agent.run_books_agent(agent, [{"ref": "f00001", "path": str(self.frame)}], "DE", "EUR")
        self.assertEqual(result["matches"]["by"], "rules")
        self.assertEqual([b["title"] for b in result["verified"]["accepted"]], ["The Lean Startup"])

    def test_default_uses_the_llm(self):
        agent = books_agent.create_books_agent(FakeReader, self.price)       # built only, never run here
        self.assertEqual(agent.sub_agents[2].sub_agents[0].name, "MatchBooksLLM")

    async def test_rules_keep_their_match_when_the_model_skips_it(self):
        result = await self.run_with(FakeMatcher(name="M", picks=[]))       # the model picked nothing
        self.assertEqual(result["matches"]["by"], f"{books_agent.MODEL}+rules")
        self.assertEqual([b["title"] for b in result["verified"]["accepted"]], ["The Lean Startup"])

    async def test_search_failure_loses_only_that_reading(self):
        def failing(text, limit=10, fallback=True, memo=None):
            raise OSError("Open Library unreachable")
        with patch.object(books_agent, "search", failing):
            result = await self.run_with(FakeMatcher(name="M", picks=[]))
        self.assertEqual(result["verified"]["accepted"], [])                  # nothing found, but no crash
        self.assertEqual(result["matches"]["by"], "none")

    async def test_quota_error_pauses_the_model(self):
        class QuotaError(Exception):
            pass
        QuotaError.__name__ = "_ResourceExhaustedError"
        calls = []

        class QuotaMatcher(FakeMatcher):
            @override
            async def _run_async_impl(self, ctx):
                calls.append(1)
                raise QuotaError("429 RESOURCE_EXHAUSTED")
                yield  # pragma: no cover

        with patch.object(books_agent, "_model_paused_until", 0.0):
            first = await self.run_with(QuotaMatcher(name="M"))
            second = await self.run_with(QuotaMatcher(name="M"))
        self.assertTrue(first["matches"]["by"].startswith("rules (model failed"))
        self.assertEqual(second["matches"]["by"], "rules (model paused after quota error)")
        self.assertEqual(len(calls), 1)                                     # the paused model was not called again
        self.assertEqual([b["title"] for b in second["verified"]["accepted"]], ["The Lean Startup"])

    async def test_model_failure_falls_back_to_rules(self):
        result = await self.run_with(FakeMatcher(name="M", fail=True))
        self.assertTrue(result["matches"]["by"].startswith("rules"))
        self.assertEqual([b["title"] for b in result["verified"]["accepted"]], ["The Lean Startup"])


if __name__ == "__main__":
    unittest.main()
