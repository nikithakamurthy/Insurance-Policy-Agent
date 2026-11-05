"""Book agent, step 1: read spine text from the spine frames (docTR OCR).

Spines carry text in four directions: horizontal (books lying flat), top-to-bottom (common
for English-language spines), bottom-to-top (common for German and French spines) and upside
down (a book lying flat the other way round, or held upside down). Each frame is read four
times: upright, rotated 90 degrees counter-clockwise, clockwise, and 180 degrees.

docTR runs with rotated boxes (assume_straight_pages=False): each line gets a box along its own
slant, and each word is turned upright before it is read, so a leaning spine reads as well as a
straight one. The price is that several rotations now read the same text. Lines from different
rotations that cover the same place in the frame are one text; per place the reading whose words
needed no turning is kept (that rotation saw the text upright, so its word order is right), plus
any other reading that adds at least two confidently read words the kept ones lack (a rotation
sometimes sees only part of a slanted spine). Measured on a tilted shelf photo and six sweep
frames against the hand-made sheet: 45 books read with 248 readings, against 36 books with 377
readings for straight boxes.

Within one reading, a spine is a horizontal strip, so text lines sitting end to end on the
same strip belong to the same spine. Each strip, and each pair of neighbouring strips (title
and author are sometimes on separate lines of a wide spine), becomes a candidate text.
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from rapidfuzz import fuzz, process

ORIENTATIONS = {
    "horizontal": None,
    "top_to_bottom": Image.Transpose.ROTATE_90,     # counter-clockwise: top-to-bottom text becomes left-to-right
    "bottom_to_top": Image.Transpose.ROTATE_270,    # clockwise
    "upside_down": Image.Transpose.ROTATE_180,
}
LYING_FLAT = {"horizontal", "upside_down"}          # text runs along the frame's width: the book lies flat
SAME_PLACE_IOU = 0.5        # lines from two rotations overlapping this much in the frame are one text
NEW_WORDS = 2               # another reading of the place is kept when it adds this many words ...
NEW_WORD_CONFIDENCE = 0.9   # ... each read at least this confidently ...
SAME_WORD = 85              # ... and none already kept (fuzzy ratio below this)


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


def point_to_original(x: float, y: float, orientation: str) -> tuple[float, float]:
    """Map a normalised point from a rotated reading back to the original frame (as to_original)."""
    if orientation == "top_to_bottom":
        return 1 - y, x
    if orientation == "bottom_to_top":
        return y, 1 - x
    if orientation == "upside_down":
        return 1 - x, 1 - y
    return x, y


def _words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower().replace("'", "")) if sum(c.isalpha() for c in w) >= 3]


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter, _ = cv2.intersectConvexConvex(a, b)
    return inter / max(1e-9, cv2.contourArea(a) + cv2.contourArea(b) - inter)


def keep_best_readings(lines_by_orientation: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Drop readings of the same place that add nothing (see the module notes).

    Lines need "poly" (corners in the original frame, normalised), "turn" (mean degrees docTR
    turned the words upright) and "word_confidences" ([(word, confidence)]); lines without
    them are kept as they are.
    """
    lines = [(o, l) for o, ls in lines_by_orientation.items() for l in ls if "poly" in l]
    hull = {id(l): cv2.convexHull(np.float32(l["poly"])) for _, l in lines}
    groups: list[list[tuple[str, dict]]] = []
    for o, l in sorted(lines, key=lambda ol: -cv2.contourArea(hull[id(ol[1])])):
        group = next((g for g in groups if _iou(hull[id(g[0][1])], hull[id(l)]) > SAME_PLACE_IOU), None)
        (group.append((o, l)) if group else groups.append([(o, l)]))
    dropped = set()
    for group in groups:
        ranked = sorted(group, key=lambda ol: (ol[1]["turn"], -ol[1]["confidence"], -len(ol[1]["word_confidences"])))
        have = {w for word, _ in ranked[0][1]["word_confidences"] for w in _words(word)}
        for _, l in ranked[1:]:
            new = {w for word, c in l["word_confidences"] if c >= NEW_WORD_CONFIDENCE for w in _words(word)
                   if not have or process.extractOne(w, list(have), scorer=fuzz.ratio)[1] < SAME_WORD}
            if len(new) >= NEW_WORDS:
                have |= {w for word, _ in l["word_confidences"] for w in _words(word)}
            else:
                dropped.add(id(l))
    return {o: [l for l in ls if id(l) not in dropped] for o, ls in lines_by_orientation.items()}


def _span_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0)) / max(1e-9, min(a1 - a0, b1 - b0))


def group_strips(lines: list[dict], min_y_overlap: float = 0.6, max_x_overlap: float = 0.2) -> list[list[dict]]:
    """Group lines into spine strips, top to bottom.

    Two lines are on the same spine when they sit end to end along the strip: their extents
    across the strip overlap strongly and their extents along it barely overlap. Lines stacked
    above one another (overlapping along the strip) are neighbouring spines, or two lines of one
    wide spine, which the neighbouring-strip pairs cover.

    With rotated OCR boxes ("poly"), along and across follow the slant of the strip's first line,
    so lines of neighbouring slanted spines are not joined; otherwise they are the box's x and y.
    """
    slanted = bool(lines) and all("poly" in l for l in lines)

    def extents(line: dict, axes: tuple[np.ndarray, np.ndarray] | None) -> tuple[float, float, float, float]:
        if axes is None:
            x0, y0, x1, y1 = line["box"]
            return x0, x1, y0, y1
        along, across = (np.asarray(line["poly"], dtype=float) @ v for v in axes)
        return along.min(), along.max(), across.min(), across.max()

    strips: list[tuple[list[dict], tuple | None]] = []
    for line in sorted(lines, key=lambda l: (l["box"][1] + l["box"][3]) / 2):
        for strip, axes in strips:
            a0, a1, c0, c1 = extents(line, axes)
            sc0 = min(extents(l, axes)[2] for l in strip)
            sc1 = max(extents(l, axes)[3] for l in strip)
            if _span_overlap(c0, c1, sc0, sc1) >= min_y_overlap and \
                    all(_span_overlap(a0, a1, *extents(l, axes)[:2]) <= max_x_overlap for l in strip):
                strip.append(line)
                break
        else:
            strips.append(([line], _axes(line["poly"]) if slanted else None))
    return [sorted(s, key=lambda l: l["box"][0]) for s, _ in strips]


def _axes(poly: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    """Unit vectors along and across a slanted line box."""
    (_, _), (w, h), angle = cv2.minAreaRect(np.float32(poly))
    a = np.deg2rad(angle if w >= h else angle + 90)
    return np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])


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
                                   assume_straight_pages=False, detect_orientation=False)
        # On the GPU (M5): about 1.3 s per frame for four orientations with rotated boxes (1.2 s
        # with straight boxes); the CPU is about three times slower. Live sweeps need the GPU.
        self.model = self.model.to(pick_device(device))
        self.min_word_conf = min_word_conf

    def read(self, image: Image.Image) -> dict[str, list[dict]]:
        """OCR lines per orientation. "box" is the line's upright bounding box, normalised in that
        orientation's view; "poly" its slanted corners, normalised in the original frame."""
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
                    corners = np.asarray(line.geometry, dtype=float).reshape(-1, 2)
                    (x0, y0), (x1, y1) = corners.min(axis=0), corners.max(axis=0)
                    turns = [abs(w.crop_orientation["value"]) % 360 for w in words]
                    lines.append({"text": " ".join(w.value for w in words), "box": [round(float(v), 4) for v in (x0, y0, x1, y1)],
                                  "confidence": round(float(np.mean([w.confidence for w in words])), 3),
                                  "poly": [[round(v, 4) for v in point_to_original(x, y, orientation)] for x, y in corners],
                                  "turn": round(float(np.mean([min(t, 360 - t) for t in turns])), 1),
                                  "word_confidences": [(w.value, round(float(w.confidence), 3)) for w in words]})
            out[orientation] = lines
        return {o: [{k: v for k, v in l.items() if k != "word_confidences"} for l in ls]
                for o, ls in keep_best_readings(out).items()}


def read_spines(sweep_dir: Path, spine_frames: list[dict]) -> dict:
    """OCR every spine frame. Stores raw lines, so grouping can be re-tuned without re-reading."""
    reader = SpineReader()
    lines = {}
    for i, frame in enumerate(spine_frames, 1):
        with Image.open(sweep_dir / frame["path"]) as img:
            lines[frame["frame_ref"]] = reader.read(img.convert("RGB"))
        if i % 10 == 0 or i == len(spine_frames):
            print(f"  read {i}/{len(spine_frames)} spine frames", flush=True)
    return {"model": "docTR db_resnet50 + parseq, rotated boxes, four orientations", "lines": lines}


