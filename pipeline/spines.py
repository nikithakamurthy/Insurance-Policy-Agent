"""Stage 3: choose the frames that show book spines, as input for the OCR stage."""

from __future__ import annotations

from pathlib import Path

import cv2


def is_spine_frame(detections: list[dict], *, min_books: int = 3, min_shelf_area: float = 0.15) -> bool:
    books = [d for d in detections if d["category"] == "book"]
    shelves = [d for d in detections if d["category"] == "bookshelf"]
    return len(books) >= min_books or (bool(books) and any(s["area_frac"] >= min_shelf_area for s in shelves))


def select_spine_frames(sweep_dir: Path, frames: list[dict], detections: dict, *, previews: bool = True) -> list[dict]:
    """Keep non-blurry frames that contain spines, ranked by spine count then sharpness."""
    chosen = []
    for frame in frames:
        dets = detections["frames"].get(frame["id"], [])
        if frame["blurry"] or not is_spine_frame(dets):
            continue
        books = [d for d in dets if d["category"] == "book"]
        chosen.append({
            "frame_ref": frame["id"], "path": frame["path"], "t": frame["t"], "sharpness": frame["sharpness"],
            "book_boxes": [d["box"] for d in books],
            "bookshelf_boxes": [d["box"] for d in dets if d["category"] == "bookshelf"],
            "n_books_detected": len(books),
        })
        if previews:
            _write_preview(sweep_dir, frame, dets)
    return sorted(chosen, key=lambda f: (-f["n_books_detected"], -f["sharpness"]))


COLOURS = {"book": (0, 200, 255), "bookshelf": (255, 120, 0)}


def _write_preview(sweep_dir: Path, frame: dict, detections: list[dict]) -> None:
    image = cv2.imread(str(sweep_dir / frame["path"]))
    for d in detections:
        x0, y0, x1, y1 = map(int, d["box"])
        colour = COLOURS.get(d["category"], (80, 220, 80))
        cv2.rectangle(image, (x0, y0), (x1, y1), colour, 2)
        cv2.putText(image, f"{d['category']} {d['score']:.2f}", (x0 + 3, max(14, y0 + 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    out = sweep_dir / "previews" / f"{frame['id']}.jpg"
    out.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out), image, [cv2.IMWRITE_JPEG_QUALITY, 80])
