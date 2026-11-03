"""Book agent, measurement: spine height and thickness in centimetres, and shelf run in metres.

Live, each frame is measured right after its OCR (FrameMeasurer: outlines, card, depth), in
its own queue so reading never waits; finish() then links titles and combines frames whenever
asked. Offline, measure_sweep does both on a saved sweep (spine_text.json, books_identified.json).

Spines
  Each OCR text strip prompts SAM ("outline the object under these points"), so the outline
  returned is the spine that text is printed on; the title linked to the text is linked to the
  outline. A rotated rectangle fitted to the outline gives the spine's length (the book's
  height) and width (its thickness) in pixels, upright, leaning or lying flat. Two prompts
  that outline the same spine (mirrored readings of one spine) are kept once. A spine cut off
  by the frame edge gets no height from that frame.

Scale: pixels to centimetres, three methods, each reported separately
  card   A bank card (ISO/IEC 7810 ID-1, 85.60 x 53.98 mm) held flat against the books, back
         side out so no card number is filmed. Found by Grounding DINO, outlined by SAM;
         accepted only if the outline's aspect ratio is within 12% of the card's 1.586. Its sides give pixels per centimetre where the card is. If
         Depth Pro ran on the frame, the scale is moved to each spine's own distance with the
         depth ratio (relative depth only; Depth Pro's absolute metres are not used).
  ruler  Frames without a card: books already measured with a card are rulers. Their pixel
         height over their card-measured height gives the scale in that frame.
  model  Depth Pro's metric depth and its own focal-length estimate:
         cm = pixels x depth_m x 100 / focal_px. A learned estimate, labelled as such.
  A book's reported size comes from card, else ruler, else model; the other methods' values are
  kept beside it for comparison.
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .catalog import _found, _with_joins, significant, tokens
from .common import pick_device, read_json, write_json
from .spine_ocr import LYING_FLAT, group_strips, to_original
from .telemetry import stage

SAM_ID = "facebook/sam-vit-base"
DEPTH_ID = "apple/DepthPro-hf"
CARD_CM = (8.560, 5.398)                 # ISO/IEC 7810 ID-1
CARD_ASPECT = CARD_CM[0] / CARD_CM[1]
CARD_ASPECT_TOLERANCE = 0.12
CARD_PHRASES = ["bank card", "credit card", "plastic card"]
HEIGHT_RANGE_CM = (10.0, 40.0)           # pocket books to large art books
THICKNESS_RANGE_CM = (0.3, 10.0)
METHOD_ORDER = ("card", "ruler", "model")
MIN_OCR_CONFIDENCE = 0.5


@dataclass
class Spine:
    frame_ref: str
    text: str
    orientation: str                     # OCR reading direction: "horizontal" or "upside_down" means the book lies flat
    length_px: float
    thickness_px: float
    centre: tuple[float, float]
    angle: float
    touches_edge: bool
    score: float
    mask: np.ndarray | None = field(repr=False)     # dropped once the frame is measured
    depth_m: float | None = None
    book_id: str | None = None
    sides: tuple | None = None           # pixel vectors of the long and short side, for per-axis scale
    behind_card: bool = False            # the card covers part of it: no height from this frame

    @property
    def upright(self) -> bool:
        return self.orientation not in LYING_FLAT


# ---- geometry --------------------------------------------------------------------------------
def fit_rectangle(mask: np.ndarray) -> tuple[float, float, tuple[float, float], float]:
    """Long side, short side, centre and angle of the smallest rotated rectangle around a mask."""
    ys, xs = np.nonzero(mask)
    (cx, cy), (w, h), angle = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
    return max(w, h), min(w, h), (cx, cy), angle


def side_vectors(mask: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    """Pixel vectors (dx, dy) of the fitted rectangle's long side and short side."""
    ys, xs = np.nonzero(mask)
    corners = cv2.boxPoints(cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32)))
    a, b = corners[1] - corners[0], corners[2] - corners[1]
    long_side, short_side = (a, b) if np.hypot(*a) >= np.hypot(*b) else (b, a)
    return (float(long_side[0]), float(long_side[1])), (float(short_side[0]), float(short_side[1]))


def axis_scales(long_side: tuple[float, float], short_side: tuple[float, float]) -> tuple[float, float] | None:
    """Pixels per cm along x and along y, from the card's two sides of known length.

    A camera looking down at an upright card shortens its vertical side more than its
    horizontal one (6% on the sample sweep); upright spines are shortened the same way, so
    heights use the vertical scale. Solves (dx/sx)^2 + (dy/sy)^2 = length^2 for both sides;
    None when the card is turned so far that the two are not separable."""
    (ax, ay), (bx, by) = long_side, short_side
    matrix = np.array([[ax * ax, ay * ay], [bx * bx, by * by]])
    if abs(np.linalg.det(matrix)) < 0.25 * abs(matrix[0, 0] * matrix[1, 1] + matrix[0, 1] * matrix[1, 0]):
        return None
    u, v = np.linalg.solve(matrix, [CARD_CM[0] ** 2, CARD_CM[1] ** 2])
    if u <= 0 or v <= 0:
        return None
    return float(1 / np.sqrt(u)), float(1 / np.sqrt(v))


def touches_edge(mask: np.ndarray, margin: int = 2) -> bool:
    return bool(mask[:margin].any() or mask[-margin:].any() or mask[:, :margin].any() or mask[:, -margin:].any())


def card_scale(long_px: float, short_px: float) -> float | None:
    """Pixels per centimetre from a card outline, or None if its shape is not a card's."""
    if short_px <= 0 or abs(long_px / short_px / CARD_ASPECT - 1) > CARD_ASPECT_TOLERANCE:
        return None
    return (long_px / CARD_CM[0] + short_px / CARD_CM[1]) / 2


def model_scale(depth_m: float, focal_px: float) -> float:
    """Pixels per centimetre at a distance, from the pinhole model."""
    return focal_px / (depth_m * 100)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def in_range(value: float | None, bounds: tuple[float, float]) -> float | None:
    return round(value, 1) if value is not None and bounds[0] <= value <= bounds[1] else None


# ---- models ----------------------------------------------------------------------------------
class Segmenter:
    """SAM with point or box prompts on one image embedding per frame."""

    def __init__(self, device: str, model_id: str = SAM_ID):
        import torch
        from transformers import SamModel, SamProcessor

        self.torch, self.device = torch, device
        self.processor = SamProcessor.from_pretrained(model_id)
        self.model = SamModel.from_pretrained(model_id).to(device).eval()
        self._image = self._embedding = None

    def set_image(self, image: Image.Image) -> None:
        inputs = self.processor(image, return_tensors="pt")
        with self.torch.no_grad():
            self._embedding = self.model.get_image_embeddings(inputs["pixel_values"].to(self.device))
        self._image = image

    def masks(self, *, points: list[list[float]] | None = None, box: list[float] | None = None) -> list[tuple[np.ndarray, float]]:
        kwargs = {"input_points": [[points]]} if points else {"input_boxes": [[box]]}
        prompt = self.processor(self._image, return_tensors="pt", **kwargs)
        key = "input_points" if points else "input_boxes"
        with self.torch.no_grad():
            out = self.model(image_embeddings=self._embedding, multimask_output=True,
                             **{key: prompt[key].float().to(self.device)})   # MPS has no float64
        masks = self.processor.image_processor.post_process_masks(
            out.pred_masks.cpu(), prompt["original_sizes"], prompt["reshaped_input_sizes"])[0][0].numpy()
        return list(zip(masks, out.iou_scores.float().cpu().numpy()[0][0].tolist()))


class DepthModel:
    """Depth Pro: metric depth per pixel and an estimate of the camera's focal length."""

    def __init__(self, device: str, model_id: str = DEPTH_ID):
        import torch
        from transformers import DepthProForDepthEstimation, DepthProImageProcessor

        self.torch, self.device = torch, device
        self.processor = DepthProImageProcessor.from_pretrained(model_id)
        self.model = DepthProForDepthEstimation.from_pretrained(model_id).to(device).eval()

    def __call__(self, image: Image.Image) -> tuple[np.ndarray, float]:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model(**inputs)
        post = self.processor.post_process_depth_estimation(out, target_sizes=[(image.height, image.width)])[0]
        return post["predicted_depth"].float().cpu().numpy(), float(post["focal_length"])


# ---- per frame -------------------------------------------------------------------------------
def text_strips(lines_by_orientation: dict[str, list[dict]], width: int, height: int) -> list[dict]:
    """OCR strips (lines of one spine) with their box in frame pixels."""
    strips = []
    for orientation, lines in lines_by_orientation.items():
        kept = [{**l, "box": tuple(l["box"])} for l in lines if l.get("confidence", 1) >= MIN_OCR_CONFIDENCE]
        for strip in group_strips(kept):
            xs0, ys0, xs1, ys1 = zip(*(l["box"] for l in strip))
            x0, y0, x1, y1 = to_original((min(xs0), min(ys0), max(xs1), max(ys1)), orientation)
            text = " ".join(l["text"] for l in strip)
            if len(text.strip()) >= 4:
                strips.append({"text": text, "orientation": orientation, "box": [x0 * width, y0 * height, x1 * width, y1 * height]})
    return strips


def segment_spines(segmenter: Segmenter, strips: list[dict], frame_ref: str, frame_area: int) -> list[Spine]:
    """One outline per spine, prompted with points along its text."""
    spines: list[Spine] = []
    for strip in strips:
        x0, y0, x1, y1 = strip["box"]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        along = [(x0 + (x1 - x0) * f, cy) for f in (0.25, 0.5, 0.75)] if strip["orientation"] in LYING_FLAT \
            else [(cx, y0 + (y1 - y0) * f) for f in (0.25, 0.5, 0.75)]
        best = None
        for mask, score in segmenter.masks(points=[list(p) for p in along]):
            if mask.sum() < 200 or mask.sum() > 0.2 * frame_area:
                continue
            covered = mask[int(y0):int(y1) + 1, int(x0):int(x1) + 1].mean()
            length, thickness, centre, angle = fit_rectangle(mask)
            if covered < 0.6 or length / max(thickness, 1) < 2:      # must hold the text and be spine-shaped
                continue
            if best is None or score > best.score:
                best = Spine(frame_ref, strip["text"], strip["orientation"], length, thickness, centre, angle,
                             touches_edge(mask), score, mask, sides=side_vectors(mask))
        if best is None:
            continue
        twin = next((s for s in spines if mask_iou(s.mask, best.mask) > 0.5), None)
        if twin is None:
            spines.append(best)
        else:                         # the same spine from a second strip: keep the better outline, both texts
            keep, other = (best, twin) if best.score > twin.score else (twin, best)
            keep.text = f"{keep.text} {other.text}"
            spines[spines.index(twin)] = keep
    return spines


def find_card(detector, segmenter: Segmenter, image: Image.Image, spines: list[Spine]) -> dict | None:
    """The reference card in a frame, or None. The outline must have a card's proportions and
    must not be one of the spines."""
    for hit in detector.find(image, CARD_PHRASES, threshold=0.3)[:3]:
        for mask, score in sorted(segmenter.masks(box=hit["box"]), key=lambda m: -m[1]):
            if mask.sum() < 400 or touches_edge(mask) or any(mask_iou(mask, s.mask) > 0.3 for s in spines):
                continue
            long_px, short_px, centre, angle = fit_rectangle(mask)
            scale = card_scale(long_px, short_px)
            if scale:
                per_axis = axis_scales(*side_vectors(mask))
                return {"px_per_cm": round(scale, 3),
                        "px_per_cm_xy": [round(v, 3) for v in per_axis] if per_axis else None,
                        "long_px": round(long_px, 1), "short_px": round(short_px, 1),
                        "aspect": round(long_px / short_px, 3), "centre": [round(c, 1) for c in centre],
                        "detector_score": hit["score"], "sam_score": round(score, 3), "mask": mask}
    return None


def link_books(spines: list[Spine], books: list[dict]) -> None:
    """Attach a book to a spine when most of the book's title words were read on it."""
    for spine in spines:
        spine.book_id = None
        pool = _with_joins(tokens(spine.text))
        best, best_cover = None, 0.0
        for book in books:
            words = significant(tokens(book["title"].split(":")[0]))
            if not words:
                continue
            cover = sum(_found(w, pool) for w in words) / len(words)
            if cover > best_cover:
                best, best_cover = book, cover
        if best is not None and best_cover >= 0.5:
            spine.book_id = best["id"]


def median_depth(depth: np.ndarray, mask: np.ndarray) -> float:
    inner = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    values = depth[inner if inner.sum() > 50 else mask]
    return float(np.median(values))


def mark_behind_card(spines: list[Spine], card_mask: np.ndarray, reach: float = 0.3) -> None:
    """A card held across spines hides part of them, so their outlines stop at the card (heights
    27-30% short on a live sweep). A spine gives no height from this frame when the card lies
    where the spine would continue: just past either end, along its length, within its width.
    A card standing beside a spine does not count; that dropped good views (Spark: 23.1-23.6 cm
    against a true 23.6) in a first version that checked for any contact."""
    for spine in spines:
        (lx, ly), _ = spine.sides
        length = float(np.hypot(lx, ly))
        if length == 0:
            continue
        u = np.array([lx, ly]) / length                      # along the spine
        v = np.array([-u[1], u[0]]) * spine.thickness_px * 0.4  # 80% of its width, centred
        centre = np.array(spine.centre)
        zone = np.zeros(card_mask.shape, dtype=np.uint8)
        for sign in (1, -1):
            near, far = centre + sign * u * length / 2, centre + sign * u * length * (0.5 + reach)
            cv2.fillPoly(zone, [np.array([near + v, far + v, far - v, near - v]).astype(np.int32)], 1)
        hidden = np.logical_and(zone.astype(bool), card_mask).sum()
        spine.behind_card = bool(hidden > 0.15 * max(zone.sum(), 1) / 2)


def sizes(spine: Spine, px_per_cm: float | tuple[float, float]) -> dict:
    """Height and thickness in cm. A (x, y) scale converts each side by its direction."""
    if isinstance(px_per_cm, tuple) and spine.sides:
        sx, sy = px_per_cm
        length, thickness = (float(np.hypot(dx / sx, dy / sy)) for dx, dy in spine.sides)
    else:
        scale = float(np.sqrt(px_per_cm[0] * px_per_cm[1])) if isinstance(px_per_cm, tuple) else px_per_cm
        length, thickness = spine.length_px / scale, spine.thickness_px / scale
    return {"height_cm": None if spine.touches_edge or spine.behind_card else length, "thickness_cm": thickness}


# ---- the sweep -------------------------------------------------------------------------------
@dataclass
class FrameResult:
    ref: str
    spines: list[Spine]
    card: dict | None
    focal_px: float | None
    measurements: list[dict] = field(default_factory=list)


class FrameMeasurer:
    """The per-frame work: spine outlines, the card and depth. It runs right after a frame's OCR
    (it needs the text strips as prompts) and keeps only sizes, not masks."""

    def __init__(self, device: str | None = None, *, detector=None, use_model: bool = True):
        from .detect import Detector

        device = pick_device(device)
        self.segmenter = Segmenter(device)
        self.detector = detector or Detector(device)
        self.depth_model = DepthModel(device) if use_model else None

    def measure_frame(self, ref: str, image_path: Path, lines: dict[str, list[dict]],
                      overlay_dir: Path | None = None) -> FrameResult:
        image = Image.open(image_path).convert("RGB")
        with stage("measure: spine outlines (SAM)"):
            self.segmenter.set_image(image)
            spines = segment_spines(self.segmenter, text_strips(lines, image.width, image.height), ref,
                                    image.width * image.height)
        with stage("measure: card (Grounding DINO + SAM)"):
            card = find_card(self.detector, self.segmenter, image, spines)
        if card:
            mark_behind_card(spines, card["mask"])
        with stage("measure: depth (Depth Pro)"):
            depth, focal = self.depth_model(image) if self.depth_model else (None, None)
        if depth is not None:
            for spine in spines:
                spine.depth_m = median_depth(depth, spine.mask)
            if card:
                card["depth_m"] = round(median_depth(depth, card["mask"]), 3)
        result = FrameResult(ref, spines, card, focal)
        if overlay_dir is not None:
            _overlay(overlay_dir, image_path, result)
        for spine in spines:          # a mask is 2 MB at 1080p; the sizes and side vectors are kept
            spine.mask = None
        if card:
            card.pop("mask")
        return result


def finish(results: list[FrameResult], books: list[dict], *, use_model: bool = True) -> dict:
    """Link titles, apply the scales and combine frames into one size per book. Cheap, so it is
    rerun whenever more books are identified."""
    for result in results:
        result.measurements = []
        link_books(result.spines, books)     # a title is often read in one frame and identified from another
        card = result.card
        for spine in result.spines if card else []:
            # The card's scale holds at the card's distance; move it to the spine's distance.
            ratio = card["depth_m"] / spine.depth_m if spine.depth_m and card.get("depth_m") else 1.0
            scale = (card["px_per_cm_xy"][0] * ratio, card["px_per_cm_xy"][1] * ratio) if card["px_per_cm_xy"] \
                else card["px_per_cm"] * ratio
            result.measurements.append({"spine": spine, "method": "card", **sizes(spine, scale)})
    # One sweep is one camera, so its focal length is the same in every frame. Depth Pro's
    # per-frame guess varied 1718-2032 px on one sweep; the median steadies the model sizes.
    focals = [r.focal_px for r in results if r.focal_px]
    sweep_focal = statistics.median(focals) if focals else None
    for result in results:
        for spine in result.spines:
            if sweep_focal and spine.depth_m:
                result.measurements.append({"spine": spine, "method": "model", **sizes(spine, model_scale(spine.depth_m, sweep_focal))})
    _add_ruler_measurements(results)
    return _combine(results, books, use_model)


def measure_sweep(sweep_dir: Path, *, use_model: bool = True, device: str | None = None, overlays: bool = True,
                  measurer: FrameMeasurer | None = None) -> dict:
    """Measure every spine in a saved sweep's kept frames (offline)."""
    spine_text = read_json(sweep_dir / "spine_text.json")["lines"]
    books = read_json(sweep_dir / "books_identified.json").get("identified", [])
    frames = {f["id"]: f for f in read_json(sweep_dir / "frames.json")["frames"]}
    measurer = measurer or FrameMeasurer(device, use_model=use_model)
    results = [measurer.measure_frame(ref, sweep_dir / frames[ref]["path"], lines, sweep_dir / "measure" if overlays else None)
               for ref, lines in spine_text.items() if ref in frames]
    report = finish(results, books, use_model=measurer.depth_model is not None)
    write_json(sweep_dir / "books_measured.json", report)
    return report


def _add_ruler_measurements(results: list[FrameResult]) -> None:
    """Frames without a card: books measured with a card elsewhere set the scale."""
    card_heights: dict[str, list[float]] = {}
    for result in results:
        for m in result.measurements:
            if m["method"] == "card" and m["spine"].book_id and in_range(m["height_cm"], HEIGHT_RANGE_CM):
                card_heights.setdefault(m["spine"].book_id, []).append(m["height_cm"])
    known = {book_id: statistics.median(h) for book_id, h in card_heights.items()}
    for result in results:
        if result.card:
            continue
        rulers = [s for s in result.spines if s.book_id in known and not s.touches_edge and not s.behind_card]
        if not rulers:
            continue
        for spine in result.spines:
            scales = []
            for r in rulers:
                if r is spine:
                    continue
                scale = r.length_px / known[r.book_id]
                if r.depth_m and spine.depth_m:
                    scale *= r.depth_m / spine.depth_m
                scales.append(scale)
            if scales:
                result.measurements.append({"spine": spine, "method": "ruler", "rulers": len(scales),
                                            **sizes(spine, statistics.median(scales))})


def _summary(values: list[dict]) -> dict:
    heights = [v["height_cm"] for v in values if in_range(v["height_cm"], HEIGHT_RANGE_CM)]
    thicknesses = [v["thickness_cm"] for v in values if in_range(v["thickness_cm"], THICKNESS_RANGE_CM)]
    return {"height_cm": round(statistics.median(heights), 1) if heights else None,
            "thickness_cm": round(statistics.median(thicknesses), 1) if thicknesses else None,
            "views": len(values), "heights_seen": [round(h, 1) for h in heights],
            "thicknesses_seen": [round(t, 1) for t in thicknesses]}


def closest_thickness(values: list[dict], keep: float = 0.75) -> float | None:
    """Thickness from the closest views only: those with at least `keep` of the best pixels per
    cm. A 2 cm spine is ~40 px wide in a whole-bookcase shot and the outline's edges add 30-70%
    there; on the bank-card sample, close-ups cut the error on 4 of 7 books to 10% or less."""
    seen = [(v["spine"].thickness_px / v["thickness_cm"], v["thickness_cm"]) for v in values
            if in_range(v["thickness_cm"], THICKNESS_RANGE_CM)]
    if not seen:
        return None
    best = max(res for res, _ in seen)
    return round(statistics.median(t for res, t in seen if res >= keep * best), 1)


def _combine(results: list[FrameResult], books: list[dict], use_model: bool) -> dict:
    by_book: dict[str, dict[str, list[dict]]] = {}
    unidentified: dict[str, dict] = {}
    for result in results:
        for m in result.measurements:
            spine = m["spine"]
            if spine.book_id:
                by_book.setdefault(spine.book_id, {}).setdefault(m["method"], []).append(m)
            else:
                key = " ".join(sorted(set(significant(tokens(spine.text)))))[:60] or spine.text
                entry = unidentified.setdefault(key, {"text": spine.text, "upright": spine.upright, "frames": set(), "methods": {}})
                entry["frames"].add(spine.frame_ref)
                entry["methods"].setdefault(m["method"], []).append(m)

    def choose(methods: dict[str, list[dict]]) -> dict:
        per_method = {name: _summary(methods[name]) for name in METHOD_ORDER if name in methods}
        # Each dimension from the first method that has it (per_method is in METHOD_ORDER): a book
        # cut off at the edge of the card frame still gets its height from a ruler frame.
        height_from = next((name for name, v in per_method.items() if v["height_cm"] is not None), None)
        # Thickness: the closest views measured with the card's scale (card or ruler frames).
        measured = methods.get("card", []) + methods.get("ruler", [])
        thickness = closest_thickness(measured) if measured else None
        thickness_from = "closest card or ruler views" if thickness is not None else None
        if thickness is None:
            thickness_from = next((name for name, v in per_method.items() if v["thickness_cm"] is not None), None)
            thickness = per_method[thickness_from]["thickness_cm"] if thickness_from else None
        return {"spine_height_cm": per_method[height_from]["height_cm"] if height_from else None,
                "spine_thickness_cm": thickness, "scale_method": height_from, "thickness_from": thickness_from,
                "estimated": "model" in (height_from, thickness_from), "by_method": per_method}

    lines = []
    for book in books:
        methods = by_book.get(book["id"])
        line = {"id": book["id"], "title": book["title"], "author": book.get("author", "")}
        if methods is None:
            line.update(spine_height_cm=None, spine_thickness_cm=None, scale_method=None,
                        reason="spine outline not found in any frame where the title was read")
        else:
            frames = sorted({m["spine"].frame_ref for ms in methods.values() for m in ms})
            line.update(choose(methods), frames=frames,
                        upright=any(m["spine"].upright for ms in methods.values() for m in ms))
        lines.append(line)
    unread = [{"text": u["text"], "frames": sorted(u["frames"]), "upright": u["upright"], **choose(u["methods"])}
              for u in unidentified.values()]

    upright = [l for l in lines if l.get("upright") and l.get("spine_thickness_cm")]
    card_frames = [{"frame_ref": r.ref, **{k: v for k, v in r.card.items() if k != "mask"}} for r in results if r.card]
    return {
        "method": {"spines": f"SAM ({SAM_ID}) prompted with OCR text strips; rotated-rectangle fit",
                   "card": "bank card, ISO/IEC 7810 ID-1, 85.60 x 53.98 mm, held against the books",
                   "ruler": "books measured with the card, in frames without a card",
                   "model": f"{DEPTH_ID} metric depth and focal-length estimate (estimated, not measured)" if use_model else None,
                   "order": "card, then ruler, then model"},
        "card_frames": card_frames,
        "focal_px_per_frame": sorted(round(r.focal_px) for r in results if r.focal_px),
        "focal_px_used": round(statistics.median([r.focal_px for r in results if r.focal_px])) if any(r.focal_px for r in results) else None,
        "books": lines,
        "unidentified_spines": unread,
        "shelf_run_m": round(sum(l["spine_thickness_cm"] for l in upright) / 100, 2) if upright else None,
        "shelf_run_note": "sum of thickness of upright identified books only. Unidentified spines are measured "
                          "(unidentified_spines) but left out: the same unread spine in two frames cannot yet be "
                          "told apart from two spines, so adding them would double-count.",
        "frames_measured": len(results),
    }


def _overlay(out_dir: Path, image_path: Path, result: FrameResult) -> None:
    """Spine outlines with the text read on each and its size in pixels, and the card in green."""
    image = cv2.imread(str(image_path))
    for i, spine in enumerate(result.spines):
        colour = tuple(int(c) for c in np.random.default_rng(i).integers(60, 255, 3))
        image[spine.mask] = (0.5 * image[spine.mask] + 0.5 * np.array(colour)).astype(np.uint8)
        label = f"{spine.text[:20]} {spine.length_px:.0f}x{spine.thickness_px:.0f}px" + (" cut" if spine.touches_edge else "")
        x, y = int(spine.centre[0]), int(spine.centre[1])
        cv2.putText(image, label, (x - 60, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(image, label, (x - 60, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    if result.card:
        contours, _ = cv2.findContours(result.card["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (0, 255, 0), 4)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"{result.ref}.jpg"), image)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure book spines in a saved sweep.")
    parser.add_argument("sweep", type=Path, help="sweep folder with frames.json, spine_text.json, books_identified.json")
    parser.add_argument("--no-model", action="store_true", help="skip Depth Pro (card and ruler only)")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    report = measure_sweep(args.sweep, use_model=not args.no_model, device=args.device)
    for line in report["books"]:
        print(f"{line['title'][:34]:34s} {line['spine_height_cm']} x {line['spine_thickness_cm']} cm  ({line['scale_method']})")
    print(f"{len(report['unidentified_spines'])} unidentified spines; shelf run {report['shelf_run_m']} m; "
          f"card in {len(report['card_frames'])} frame(s)")


if __name__ == "__main__":
    main()
