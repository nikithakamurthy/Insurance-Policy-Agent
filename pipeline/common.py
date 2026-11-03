"""Shared helpers: device selection, JSON IO, stage timing."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def pick_device(preferred: str | None = None) -> str:
    import torch

    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=_jsonable))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _jsonable(value: Any) -> Any:
    import numpy as np

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serialisable: {type(value).__name__}")


REPO = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO / ".cache" / "http"


def load_api_key() -> str:
    """GOOGLE_API_KEY from the environment, else from .env at the repo root or reference_app/."""
    import os
    import re

    if os.getenv("GOOGLE_API_KEY"):
        return os.environ["GOOGLE_API_KEY"]
    for env in (REPO / ".env", REPO / "reference_app" / ".env"):
        if env.exists():
            for line in env.read_text().splitlines():
                match = re.match(r"\s*GOOGLE_API_KEY\s*=\s*(.+)", line)
                if match:
                    return match.group(1).strip().strip("\"'")
    raise RuntimeError("GOOGLE_API_KEY is not set (environment, .env, or reference_app/.env)")
USER_AGENT = "insurance-agent-pipeline/0.1 (contents-claim research prototype)"


def get_cached(url: str, params: dict | None = None, *, max_age_s: float = 7 * 86400, raw: bool = False,
               cache: bool = True, attempts: int = 3) -> tuple[Any, str]:
    """GET a URL, with an on-disk cache unless cache=False. Returns (body, retrieved_at ISO timestamp).

    The retrieval time is the time of the original request, so a cached response keeps the date
    it was actually seen. Query parameters named 'key' are left out of the cache key.
    """
    import hashlib
    import urllib.parse
    import urllib.request

    full = url + ("?" + urllib.parse.urlencode(params) if params else "")
    safe = {k: v for k, v in (params or {}).items() if k != "key"}
    digest = hashlib.sha256((url + json.dumps(safe, sort_keys=True)).encode()).hexdigest()[:24]
    path = CACHE_DIR / f"{digest}.json"
    if cache and path.exists() and time.time() - path.stat().st_mtime < max_age_s:
        cached = json.loads(path.read_text())
        return cached["body"], cached["retrieved_at"]
    from .telemetry import record_call, service_of

    request = urllib.request.Request(full, headers={"User-Agent": USER_AGENT})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                text = response.read().decode("utf-8")
            record_call(service_of(url))
            break
        except (TimeoutError, OSError) as exc:
            record_call(service_of(url), ok=False)
            # HTTP 4xx are errors in the request itself; retrying will not help.
            if getattr(exc, "code", 500) < 500 or attempt == attempts - 1:
                raise
            time.sleep(2 * (attempt + 1))
    body = text if raw else json.loads(text)
    retrieved_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    if cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"url": url, "params": safe, "retrieved_at": retrieved_at, "body": body}))
    return body, retrieved_at


@contextmanager
def timed(timings: dict[str, float], stage: str):
    started = time.perf_counter()
    try:
        yield
    finally:
        timings[stage] = round(time.perf_counter() - started, 2)
