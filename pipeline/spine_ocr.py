"""Book agent, step 1: read spine text from the spine frames (docTR OCR).

Spines carry text in four directions: horizontal (books lying flat), top-to-bottom (common
for English-language spines), bottom-to-top (common for German and French spines) and upside
down (a book lying flat the other way round, or held upside down). Each frame is read four
times: upright, rotated 90 degrees counter-clockwise, clockwise, and 180 degrees. The wrong
rotation produces mirrored gibberish, which the catalogue match later rejects.

Within one reading, a spine is a horizontal strip, so text lines sitting end to end on the
same strip belong to the same spine. Each strip, and each pair of neighbouring strips (title
and author are sometimes on separate lines of a wide spine), becomes a candidate text.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

ORIENTATIONS = {
    "horizontal": None,
    "top_to_bottom": Image.Transpose.ROTATE_90,     # counter-clockwise: top-to-bottom text becomes left-to-right
    "bottom_to_top": Image.Transpose.ROTATE_270,    # clockwise
    "upside_down": Image.Transpose.ROTATE_180,
}
LYING_FLAT = {"horizontal", "upside_down"}          # text runs along the frame's width: the book lies flat


def to_original(box: tuple[float, float, float, float], orientation: str) -> list[float]:
    """Map a normalised (x0, y0, x1, y1) box from a rotated reading back to the original frame."""
    x0, y0, x1, y1 = box
    if orientation == "top_to_bottom":      # rotated (x', y') = (y, 1 - x)
        return [1 - y1, x0, 1 - y0, x1]
    if orientation == "bottom_to_top":      # rotated (x', y') = (1 - y, x)
        return [y0, 1 - x1, y1, 1 - x0]
    if orientation == "upside_down":        # rotated (x', y') = (1 - x, 1 - y)
        return [1 - x1, 1 - y1, 1 - x0, 1 - y0]
    return [x0, y0, x1, y1]


def _span_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0)) / max(1e-9, min(a1 - a0, b1 - b0))


def group_strips(lines: list[dict], min_y_overlap: float = 0.6, max_x_overlap: float = 0.2) -> list[list[dict]]:
    """Group lines into spine strips, top to bottom.

    Two lines are on the same spine when they sit end to end along the strip: their vertical
    extents overlap strongly and their horizontal extents barely overlap. Lines stacked above
    one another (overlapping horizontally) are neighbouring spines, or two lines of one wide
    spine, which the neighbouring-strip pairs cover.
    """
    strips: list[list[dict]] = []
    for line in sorted(lines, key=lambda l: (l["box"][1] + l["box"][3]) / 2):
        x0, y0, x1, y1 = line["box"]
        for strip in strips:
            sy0, sy1 = min(l["box"][1] for l in strip), max(l["box"][3] for l in strip)
            if _span_overlap(y0, y1, sy0, sy1) >= min_y_overlap and \
                    all(_span_overlap(x0, x1, l["box"][0], l["box"][2]) <= max_x_overlap for l in strip):
                strip.append(line)
                break
        else:
            strips.append([line])
    return [sorted(s, key=lambda l: l["box"][0]) for s in strips]


def candidates_from_lines(lines_by_orientation: dict[str, list[dict]], frame_ref: str) -> list[dict]:
    """Candidate texts: every strip, and every pair of neighbouring strips."""
    candidates = []
    for orientation, lines in lines_by_orientation.items():
        strips = group_strips([{**l, "box": tuple(l["box"])} for l in lines])
        for i, strip in enumerate(strips):
            candidates.append(_strip_candidate([strip], orientation, frame_ref))
            if i + 1 < len(strips):
                candidates.append(_strip_candidate([strip, strips[i + 1]], orientation, frame_ref))
    return candidates


def _strip_candidate(strips: list[list[dict]], orientation: str, frame_ref: str) -> dict:
    lines = [l for s in strips for l in s]
    xs0, ys0, xs1, ys1 = zip(*(l["box"] for l in lines))
    return {"frame_ref": frame_ref, "orientation": orientation,
            "text": " ".join(" ".join(l["text"] for l in s) for s in strips),
            "ocr_confidence": round(float(np.mean([l["confidence"] for l in lines])), 3),
            "box_norm": [round(v, 4) for v in to_original((min(xs0), min(ys0), max(xs1), max(ys1)), orientation)],
            "lines": len(lines), "strips": len(strips)}


class SpineReader:
    def __init__(self, min_word_conf: float = 0.5, device: str | None = None):
        from doctr.models import ocr_predictor

        from .common import pick_device

        self.model = ocr_predictor(det_arch="db_resnet50", reco_arch="parseq", pretrained=True,
                                   assume_straight_pages=True, detect_orientation=False)
        # On the GPU: 0.92 s per frame for three orientations (about 1.2 s for four) instead of
        # 2.98 s for three on the CPU (M5), with identical readings. Live sweeps need the GPU.
        self.model = self.model.to(pick_device(device))
        self.min_word_conf = min_word_conf

    def read(self, image: Image.Image) -> dict[str, list[dict]]:
        """Raw OCR lines per orientation (boxes normalised in that orientation's view)."""
        out = {}
        for orientation, transpose in ORIENTATIONS.items():
            view = image if transpose is None else image.transpose(transpose)
            page = self.model([np.asarray(view)]).pages[0]
            lines = []
            for block in page.blocks:
                for line in block.lines:
                    words = [w for w in line.words if w.confidence >= self.min_word_conf and any(c.isalnum() for c in w.value)]
                    if not words or not any(sum(c.isalpha() for c in w.value) >= 3 for w in words):
                        continue
                    (x0, y0), (x1, y1) = line.geometry
                    lines.append({"text": " ".join(w.value for w in words), "box": [round(v, 4) for v in (x0, y0, x1, y1)],
                                  "confidence": round(float(np.mean([w.confidence for w in words])), 3)})
            out[orientation] = lines
        return out


def read_spines(sweep_dir: Path, spine_frames: list[dict]) -> dict:
    """OCR every spine frame. Stores raw lines, so grouping can be re-tuned without re-reading."""
    reader = SpineReader()
    lines = {}
    for i, frame in enumerate(spine_frames, 1):
        with Image.open(sweep_dir / frame["path"]) as img:
            lines[frame["frame_ref"]] = reader.read(img.convert("RGB"))
        if i % 10 == 0 or i == len(spine_frames):
            print(f"  read {i}/{len(spine_frames)} spine frames", flush=True)
    return {"model": "docTR db_resnet50 + parseq, three orientations", "lines": lines}


def spine_candidates(spine_text: dict) -> dict[str, list[dict]]:
    return {ref: candidates_from_lines(by_orientation, ref) for ref, by_orientation in spine_text["lines"].items()}
