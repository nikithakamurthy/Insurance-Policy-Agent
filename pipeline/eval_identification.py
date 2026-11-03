"""Compare book identification modes on saved sweeps against a hand-made ground-truth sheet.

Both modes see exactly the same input: each sweep's saved OCR (spine_text.json, no re-reading)
and the same catalogue search results per batch of 6 frames. Only the matching step differs:

  rules   the rule-based scorer picks, then VerifyMatches checks the pick against the words read
  gemini  the Gemini matcher picks (the live default; same instruction and output schema), the
          rules keep their confident matches, then the same VerifyMatches

Books accepted across a sweep are merged with catalog.same_work, then scored against the sheet:
correct (title matches a ground-truth row), not in ground truth (listed for a manual check:
either wrong or a real book missing from the sheet), and missed (ground-truth rows not found).

    python -m pipeline.eval_identification --gt truth.csv --sweep sweeps/live-... [--sweep ...] --modes rules,gemini
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from rapidfuzz import fuzz

from .books_agent import MATCH_INSTRUCTION, MODEL, MatchResult, _rule_matches, make_handlers
from .catalog import same_work, tokens
from .common import load_api_key, read_json

BATCH = 6
TITLE_MATCH = 85


class SavedOCR:
    """Stands in for the OCR reader: returns the lines the live run already read, frame by frame."""

    def __init__(self, lines_in_order: list[dict]):
        self._lines = iter(lines_in_order)

    def read(self, image) -> dict:
        return next(self._lines)


def gemini_matches(match_request: str, request: list[dict], client) -> tuple[dict, dict]:
    started = time.perf_counter()
    response = client.models.generate_content(
        model=MODEL, contents=MATCH_INSTRUCTION.format(match_request=match_request),
        config={"temperature": 0, "response_mime_type": "application/json", "response_schema": MatchResult})
    picked = [m for m in json.loads(response.text).get("matches", []) if m.get("work_key")]
    decided = {m["reading_id"] for m in picked}
    kept = [m for m in _rule_matches(request) if m["reading_id"] not in decided]
    usage = response.usage_metadata
    return ({"matches": picked + kept, "by": MODEL + ("+rules" if kept else "")},
            {"seconds": time.perf_counter() - started, "input_tokens": usage.prompt_token_count or 0,
             "output_tokens": usage.candidates_token_count or 0})


def evaluate_sweep(sweep: Path, modes: list[str], client=None, cache_dir: Path | None = None) -> dict:
    """cache_dir keeps each batch's catalogue search results, so the modes can be compared later on
    exactly the same results without the catalogues (or with Gemini only once its quota is back)."""
    spine_text = read_json(sweep / "spine_text.json")["lines"]
    frames = [f for f in read_json(sweep / "frames.json")["frames"] if f["id"] in spine_text]
    found = {mode: [] for mode in modes}
    cost = {mode: {"seconds": 0.0, "input_tokens": 0, "output_tokens": 0, "calls": 0, "failed": 0} for mode in modes}
    search_seconds = 0.0
    for start in range(0, len(frames), BATCH):
        batch = frames[start:start + BATCH]
        reader = SavedOCR([spine_text[f["id"]] for f in batch])
        h = make_handlers(lambda: reader, lambda book, state: None)
        state = {"frames": [{"ref": f["id"], "path": str(sweep / f["path"])} for f in batch]}
        state["readings"] = h["read"](state)
        cached = cache_dir / sweep.name / f"batch-{start:04d}.json" if cache_dir else None
        if cached and cached.exists():
            saved = json.loads(cached.read_text())
            state["search_results"], state["match_request"] = saved["search_results"], saved["match_request"]
        else:
            started = time.perf_counter()
            state["search_results"] = h["search"](state)
            search_seconds += time.perf_counter() - started
            if cached:
                cached.parent.mkdir(parents=True, exist_ok=True)
                cached.write_text(json.dumps({"search_results": state["search_results"], "match_request": state["match_request"]}))
        request = [r for r in state["search_results"] if r["results"]]
        for mode in modes:
            if not request:
                matches = {"matches": [], "by": "none"}
            elif mode == "rules":
                matches = {"matches": _rule_matches(request), "by": "rules"}
            else:
                try:
                    matches, spent = gemini_matches(state["match_request"], request, client)
                    cost[mode]["calls"] += 1
                    for key in ("seconds", "input_tokens", "output_tokens"):
                        cost[mode][key] += spent[key]
                except Exception as exc:     # a failed call is counted; that batch falls back to the rules, as live
                    cost[mode]["failed"] += 1
                    matches = {"matches": _rule_matches(request), "by": f"rules (model failed: {type(exc).__name__})"}
            verified = h["verify"]({**state, "matches": matches})
            for book in verified["accepted"]:
                if not any(same_work(b["title"], b["author"], book["title"], book["author"]) for b in found[mode]):
                    found[mode].append({"title": book["title"], "author": book["author"], "reading": book["reading"]})
        print(f"  {sweep.name}: frames {start + 1}-{start + len(batch)} of {len(frames)}", flush=True)
    return {"found": found, "cost": cost, "search_seconds": round(search_seconds, 1), "frames": len(frames)}


def score(found: list[dict], truth: list[str]) -> dict:
    def norm(text: str) -> str:            # same normalisation as matching: case, accents, odd spaces
        return " ".join(tokens(text))

    def best(title: str) -> tuple[str, int]:
        name = max(truth, key=lambda t: fuzz.token_set_ratio(norm(title), norm(t)))
        return name, fuzz.token_set_ratio(norm(title), norm(name))

    correct, other = [], []
    for book in found:
        name, ratio = best(book["title"])
        (correct if ratio >= TITLE_MATCH else other).append({**book, "truth": name if ratio >= TITLE_MATCH else None})
    hit = {b["truth"] for b in correct}
    return {"correct": correct, "not_in_ground_truth": other, "missed": [t for t in truth if t not in hit]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gt", type=Path, required=True, help="ground-truth CSV with a Name column (title and author)")
    parser.add_argument("--sweep", type=Path, action="append", required=True)
    parser.add_argument("--modes", default="rules,gemini")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--search-cache", type=Path, default=None,
                        help="folder for each batch's search results; reused on later runs")
    args = parser.parse_args()
    truth = [row["Name"] for row in csv.DictReader(args.gt.open())]
    modes = args.modes.split(",")
    client = None
    if "gemini" in modes:
        from google import genai
        client = genai.Client(api_key=load_api_key())

    per_sweep = {sweep.name: evaluate_sweep(sweep, modes, client, args.search_cache) for sweep in args.sweep}
    report = {"ground_truth_rows": len(truth), "sweeps": {}, "modes": {}}
    for mode in modes:
        merged: list[dict] = []
        for result in per_sweep.values():
            for book in result["found"][mode]:
                if not any(same_work(b["title"], b["author"], book["title"], book["author"]) for b in merged):
                    merged.append(book)
        scored = score(merged, truth)
        report["modes"][mode] = {**scored, "cost": {s: r["cost"][mode] for s, r in per_sweep.items()}}
        print(f"\n== {mode}: {len(scored['correct'])} correct, {len(scored['not_in_ground_truth'])} not in ground truth, "
              f"{len(scored['missed'])} of {len(truth)} ground-truth books missed")
        for b in scored["not_in_ground_truth"]:
            print(f"   not in ground truth: {b['title']} ({b['author']}) <- {b['reading'][:50]!r}")
        for s, c in report["modes"][mode]["cost"].items():
            if mode != "rules":
                print(f"   {s}: {c['calls']} model calls ({c['failed']} failed), {c['seconds']:.0f} s, "
                      f"{c['input_tokens']:,} in / {c['output_tokens']:,} out tokens")
    report["sweeps"] = {s: {"frames": r["frames"], "search_seconds": r["search_seconds"]} for s, r in per_sweep.items()}
    if args.out:
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
