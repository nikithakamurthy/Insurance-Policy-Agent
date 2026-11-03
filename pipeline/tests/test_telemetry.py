import asyncio
import contextvars
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

from pipeline import telemetry
from pipeline.telemetry import Usage, cost, stage


class UsageTest(unittest.TestCase):
    def test_stage_summary(self):
        usage = Usage()
        for seconds in (1.0, 2.0, 3.0, 10.0):
            usage.add_stage("search", seconds)
        summary = usage.summary()["stages"]["search"]
        self.assertEqual((summary["count"], summary["total_s"], summary["max_s"]), (4, 16.0, 10.0))

    def test_cost_only_from_configured_rates(self):
        calls = {"gemini_matcher": {"requests": 3, "errors": 0, "input_tokens": 2_000_000, "output_tokens": 500_000},
                 "google_books": {"requests": 120, "errors": 2, "input_tokens": 0, "output_tokens": 0}}
        result = cost(calls, {"gemini_matcher": {"input_per_million": 0.5, "output_per_million": 2.0}})
        self.assertEqual(result["usd_by_service"], {"gemini_matcher": 2.0})
        self.assertEqual(result["unpriced_services"], ["google_books"])     # no rate: unknown, not zero
        self.assertIsNone(cost(calls, {})["usd_total"])

    def test_requests_in_worker_threads_count_toward_the_sweep(self):
        from pipeline import common
        usage = Usage()
        token = telemetry.current.set(usage)
        try:
            with mock.patch("urllib.request.urlopen") as urlopen:
                urlopen.return_value.__enter__.return_value.read.return_value = b'{"docs": []}'
                jobs = [(contextvars.copy_context(), q) for q in ("a", "b", "c")]
                with ThreadPoolExecutor(3) as pool:
                    list(pool.map(lambda job: job[0].run(common.get_cached, "https://openlibrary.org/search.json",
                                                         {"q": job[1]}, cache=False), jobs))
        finally:
            telemetry.current.reset(token)
        self.assertEqual(usage.summary()["calls"]["open_library"]["requests"], 3)

    def test_nothing_is_recorded_outside_a_sweep(self):
        with stage("anything"):
            pass
        telemetry.record_call("google_books")      # no usage in context: a no-op, not an error

    def test_to_thread_keeps_the_sweep(self):
        usage = Usage()

        async def run():
            telemetry.current.set(usage)
            await asyncio.to_thread(telemetry.record_call, "google_books")
        asyncio.run(run())
        self.assertEqual(usage.summary()["calls"]["google_books"]["requests"], 1)


if __name__ == "__main__":
    unittest.main()
