"""Run the sweep pipeline on a video.

    .venv/bin/python -m pipeline.run --video sweep.mp4 --out sweeps/my-room

Each stage writes its own output under --out and is skipped on later runs if that output
exists; pass --force <stage> (or --force all) to recompute. Timings per stage go to run.json.
"""

from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

import numpy as np

from .common import pick_device, read_json, timed, write_json

STAGES = ["frames", "detect", "spines", "reconstruct", "room", "items", "ocr", "identify", "price"]


def write_ply(path: Path, points: np.ndarray, colours: np.ndarray) -> None:
    header = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(points)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    rows = np.empty(len(points), dtype=[("p", "<f4", 3), ("c", "u1", 3)])
    rows["p"], rows["c"] = points, colours
    with open(path, "wb") as fh:
        fh.write(header.encode())
        fh.write(rows.tobytes())


def load_recon(sweep: Path) -> dict:
    data = np.load(sweep / "reconstruction.npz")
    return {k: data[k] for k in data.files}


def confident_points(recon: dict, quantile: float = 0.4) -> tuple[np.ndarray, np.ndarray]:
    conf = recon["conf"]
    mask = conf >= np.quantile(conf, quantile)
    return recon["points"][mask], recon["images"][mask]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, help="sweep video (needed for the frames stage)")
    parser.add_argument("--out", type=Path, required=True, help="sweep output folder")
    parser.add_argument("--stages", default=",".join(STAGES), help="comma-separated stages to run")
    parser.add_argument("--force", default="", help="stages to recompute, or 'all'")
    parser.add_argument("--device", default=None, help="cuda | mps | cpu (auto by default)")
    parser.add_argument("--keep-fps", type=float, default=2.0, help="frames kept per second of video")
    parser.add_argument("--keep-duplicates", action="store_true", help="do not skip near-identical consecutive frames")
    parser.add_argument("--recon-frames", type=int, default=24, help="frames given to VGGT")
    parser.add_argument("--depth-model", default=None, help="metric depth model id (default Depth Pro)")
    parser.add_argument("--country", default=None, help="policyholder country, ISO code (e.g. DE); needed for prices")
    parser.add_argument("--currency", default=None, help="policyholder currency, ISO code (e.g. EUR)")
    parser.add_argument("--second-country", default=None, help="also price the first 10 books here (e.g. GB)")
    parser.add_argument("--second-currency", default=None, help="currency for --second-country (e.g. GBP)")
    parser.add_argument("--appraisal-threshold", type=float, default=150.0, help="prices above this go to appraisal")
    parser.add_argument("--book-language", default="en", help="edition language to price (the spines' language), ISO 639-1")
    args = parser.parse_args()

    sweep: Path = args.out
    sweep.mkdir(parents=True, exist_ok=True)
    stages = [s for s in args.stages.split(",") if s]
    force = set(STAGES) if args.force == "all" else set(filter(None, args.force.split(",")))
    device = pick_device(args.device)
    run_path = sweep / "run.json"
    run = read_json(run_path) if run_path.exists() else {"timings_s": {}}
    run.update({"device": device, "python": platform.python_version(), "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    timings = run["timings_s"]

    def needed(stage: str, output: str) -> bool:
        return stage in stages and (stage in force or not (sweep / output).exists())

    if needed("frames", "frames.json"):
        from .frames import extract_frames

        if not args.video:
            parser.error("--video is required for the frames stage")
        print("[frames] extracting", flush=True)
        with timed(timings, "frames"):
            frames, duplicates = extract_frames(args.video, sweep, keep_fps=args.keep_fps, skip_duplicates=not args.keep_duplicates)
        write_json(sweep / "frames.json", {"video": str(args.video), "frames": frames, "skipped_duplicates": duplicates})
        run["video"] = str(args.video)
    frames_data = read_json(sweep / "frames.json")
    frames = frames_data["frames"]
    print(f"[frames] {len(frames)} frames, {sum(f['blurry'] for f in frames)} blurry, "
          f"{len(frames_data.get('skipped_duplicates', []))} near-duplicates skipped", flush=True)

    if needed("detect", "detections.json"):
        from .detect import detect_frames

        print(f"[detect] Grounding DINO on {len(frames)} frames ({device})", flush=True)
        with timed(timings, "detect"):
            detections = detect_frames(sweep, frames, device)
        write_json(sweep / "detections.json", detections)

    if needed("spines", "spine_frames.json"):
        from .spines import select_spine_frames

        detections = read_json(sweep / "detections.json")
        with timed(timings, "spines"):
            spine_frames = select_spine_frames(sweep, frames, detections)
        write_json(sweep / "spine_frames.json", {"criteria": "non-blurry and (>=3 books, or a bookshelf >=15% of frame with a book)",
                                                 "frames": spine_frames})
        print(f"[spines] {len(spine_frames)} of {len(frames)} frames show spines", flush=True)

    if needed("reconstruct", "scale.json"):
        from .frames import select_spread
        from .reconstruct import DEPTH_ID, reconstruct

        chosen = select_spread(frames, args.recon_frames)
        with timed(timings, "reconstruct"):
            scale = reconstruct(sweep, chosen, device, args.depth_model or DEPTH_ID)
        write_json(sweep / "scale.json", scale)
        print(f"[reconstruct] scale {scale['scale']:.4f} m/unit, relative IQR {scale['relative_iqr']}", flush=True)

    if needed("room", "room.json"):
        from .room import floorplan_image, measure_room, up_hint_from_cameras

        recon = load_recon(sweep)
        points, colours = confident_points(recon)
        with timed(timings, "room"):
            room = measure_room(points, up_hint_from_cameras(recon["extrinsic"]))
        scale = read_json(sweep / "scale.json")
        room["scale_method"] = f"{scale['scale_method']} ({scale['depth_model']} vs {scale['geometry_model']})"
        room["scale_relative_iqr"] = scale["relative_iqr"]
        room["confidence"] = round(room["geometry_confidence"] * max(0.0, 1 - 2 * scale["relative_iqr"]), 2)
        write_json(sweep / "room.json", room)
        cams = -np.einsum("sij,si->sj", recon["extrinsic"][:, :, :3], recon["extrinsic"][:, :, 3])
        floorplan_image(points, room, cams, sweep / "floorplan.png")
        sample = np.random.default_rng(0).choice(len(points), min(len(points), 400000), replace=False)
        write_ply(sweep / "room_points.ply", points[sample], colours[sample])
        print(f"[room] {room['length_m']} x {room['width_m']} x {room['height_m']} m, floor {room['floor_area_m2']} m2, "
              f"walls {room['wall_area_m2']} m2, confidence {room['confidence']}", flush=True)

    if needed("items", "items.json"):
        from .items import locate_items

        with timed(timings, "items"):
            items = locate_items(load_recon(sweep), read_json(sweep / "detections.json"), read_json(sweep / "room.json"))
        write_json(sweep / "items.json", items)
        room = read_json(sweep / "room.json")
        room["shelved_wall_area_m2"] = items["shelved_wall_area_m2"]
        write_json(sweep / "room.json", room)
        print(f"[items] {len(items['items'])} objects located, shelved wall area {items['shelved_wall_area_m2']} m2", flush=True)

    if needed("ocr", "spine_text.json"):
        from .spine_ocr import read_spines

        spine_frames = read_json(sweep / "spine_frames.json")["frames"]
        print(f"[ocr] reading spines in {len(spine_frames)} frames", flush=True)
        with timed(timings, "ocr"):
            spine_text = read_spines(sweep, spine_frames)
        write_json(sweep / "spine_text.json", spine_text)

    if needed("identify", "books_identified.json"):
        from .catalog import identify_books

        with timed(timings, "identify"):
            books = identify_books(read_json(sweep / "spine_text.json"))
        write_json(sweep / "books_identified.json", books)
        print(f"[identify] {len(books['identified'])} books identified, {len(books['unidentified_readings'])} "
              f"readable spines not matched ({books['queries']} catalogue queries)", flush=True)

    if needed("price", "books_priced.json"):
        from .fx import load_rates
        from .pricing import price_books

        if not (args.country and args.currency):
            parser.error("--country and --currency are required for the price stage")
        identified = read_json(sweep / "books_identified.json")["identified"]
        rates = load_rates()
        with timed(timings, "price"):
            priced = price_books(identified, args.country, args.currency, rates, threshold=args.appraisal_threshold,
                                 language=args.book_language)
        write_json(sweep / "books_priced.json", priced)
        found = [b for b in priced["books"] if (b.get("replacement_cost") or {}).get("amount") is not None]
        print(f"[price] {len(found)}/{len(priced['books'])} priced in {args.currency.upper()}, "
              f"{sum(b['replacement_cost'].get('converted', False) for b in found)} converted, "
              f"{len(priced['review_queue'])} to review", flush=True)
        if args.second_country:
            if not args.second_currency:
                parser.error("--second-currency is required with --second-country")
            second = price_books(identified[:10], args.second_country, args.second_currency, rates,
                                 threshold=args.appraisal_threshold, language=args.book_language)
            write_json(sweep / f"books_priced_{args.second_country.upper()}.json", second)

    write_json(run_path, run)
    print(f"timings (s): {timings}")


if __name__ == "__main__":
    main()
