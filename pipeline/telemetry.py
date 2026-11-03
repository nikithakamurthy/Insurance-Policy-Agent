"""Latency per stage and outside-service usage (requests, tokens) per sweep.

A Usage object belongs to one sweep. Code running for that sweep finds it through a context
variable, so the catalogue and pricing helpers count their requests without being passed it.
asyncio.to_thread copies the context; plain thread pools do not, so callers submit work with
contextvars.copy_context().run.

Money is worked out only from rates the operator configures (RATES_FILE, per 1M tokens or per
request, in USD). Without a rate, cost is reported as unknown, never guessed.
"""

from __future__ import annotations

import contextvars
import json
import os
import statistics
import threading
import time
from contextlib import contextmanager
from pathlib import Path

current: contextvars.ContextVar["Usage | None"] = contextvars.ContextVar("sweep_usage", default=None)
RATES_FILE = Path(os.getenv("USAGE_RATES_FILE", Path(__file__).resolve().parent.parent / "usage_rates.json"))

# Hosts of outside services, by name in the report.
SERVICES = {"www.googleapis.com": "google_books", "openlibrary.org": "open_library",
            "www.ecb.europa.eu": "ecb_rates", "api.ebay.com": "ebay"}


class Usage:
    def __init__(self):
        self._lock = threading.Lock()
        self.stages: dict[str, list[float]] = {}
        self.calls: dict[str, dict] = {}

    def add_stage(self, name: str, seconds: float) -> None:
        with self._lock:
            self.stages.setdefault(name, []).append(seconds)

    def add_call(self, service: str, *, ok: bool = True, input_tokens: int = 0, output_tokens: int = 0) -> None:
        with self._lock:
            entry = self.calls.setdefault(service, {"requests": 0, "errors": 0, "input_tokens": 0, "output_tokens": 0})
            entry["requests"] += 1
            entry["errors"] += 0 if ok else 1
            entry["input_tokens"] += input_tokens
            entry["output_tokens"] += output_tokens

    def summary(self) -> dict:
        with self._lock:
            stages = {name: stage_summary(values) for name, values in self.stages.items()}
            calls = {name: dict(entry) for name, entry in self.calls.items()}
        return {"stages": stages, "calls": calls, "cost": cost(calls)}


def stage_summary(values: list[float]) -> dict:
    ordered = sorted(values)
    return {"count": len(ordered), "total_s": round(sum(ordered), 2), "mean_s": round(statistics.mean(ordered), 3),
            "p50_s": round(ordered[len(ordered) // 2], 3), "p95_s": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 3),
            "max_s": round(ordered[-1], 3)}


@contextmanager
def stage(name: str, usage: "Usage | None" = None):
    """Time a block as one occurrence of a stage, for the sweep in context (or the one given)."""
    usage = usage or current.get()
    started = time.perf_counter()
    try:
        yield
    finally:
        if usage is not None:
            usage.add_stage(name, time.perf_counter() - started)


def record_call(service: str, **kwargs) -> None:
    usage = current.get()
    if usage is not None:
        usage.add_call(service, **kwargs)


def service_of(url: str) -> str:
    from urllib.parse import urlparse

    return SERVICES.get(urlparse(url).hostname or "", urlparse(url).hostname or "other")


def load_rates() -> dict:
    """{service: {"per_request": usd, "input_per_million": usd, "output_per_million": usd}}."""
    try:
        return json.loads(RATES_FILE.read_text())
    except (OSError, ValueError):
        return {}


def cost(calls: dict, rates: dict | None = None) -> dict:
    """USD per service from configured rates; a service with usage but no rate is listed as unpriced."""
    rates = load_rates() if rates is None else rates
    lines, unpriced = {}, []
    for service, entry in calls.items():
        rate = rates.get(service)
        if rate is None:
            unpriced.append(service)
            continue
        usd = (entry["requests"] * rate.get("per_request", 0.0)
               + entry["input_tokens"] / 1e6 * rate.get("input_per_million", 0.0)
               + entry["output_tokens"] / 1e6 * rate.get("output_per_million", 0.0))
        lines[service] = round(usd, 4)
    return {"usd_by_service": lines, "usd_total": round(sum(lines.values()), 4) if lines else None,
            "unpriced_services": sorted(unpriced), "rates_file": str(RATES_FILE)}
