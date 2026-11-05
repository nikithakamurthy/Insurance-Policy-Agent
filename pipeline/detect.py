"""Stage 2: open-vocabulary object detection on every kept frame (Grounding DINO)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

DETECTOR_ID = "IDEA-Research/grounding-dino-base"

# Prompt phrase -> canonical category. Phrases are what Grounding DINO is asked to find;
# categories are what the inventory and the spine filter use.
VOCABULARY: dict[str, str] = {
    "book": "book",
    "book spine": "book",
    "bookshelf": "bookshelf",
    "chair": "chair",
    "armchair": "chair",
    "sofa": "sofa",
    "table": "table",
    "desk": "desk",
    "lamp": "lamp",
    "picture frame": "framed_art",
    "painting": "framed_art",
    "poster": "framed_art",
    "mirror": "mirror",
    "rug": "rug",
    "television": "electronics",
    "computer monitor": "electronics",
    "laptop": "electronics",
    "speaker": "electronics",
    "coffee machine": "appliance",
    "potted plant": "decor",
    "vase": "decor",
    "clock": "decor",
    "cabinet": "cabinet",
    "window": "window",
    "door": "door",
}
# One prompt with the whole vocabulary (well under the 256-token text limit). Measured on
# sample-1: 2.8x faster than three prompt groups with nearly the same detections.
PROMPT = list(VOCABULARY)
# Grounding DINO upscales to an 800 px short side by default. Upscaling a smaller frame adds no
# detail, so cap at the frame's own size: 0.42 s instead of 0.94 s per 480x854 frame on an M5,
# at the cost of a few small decor items and lamps.
DEFAULT_SHORT_SIDE, DEFAULT_LONG_SIDE = 800, 1333


def _canonical(label: str) -> str | None:
    """Map a (possibly merged) Grounding DINO phrase back to one vocabulary category."""
    label = label.strip().lower()
    if label in VOCABULARY:
        return VOCABULARY[label]
    words = set(label.split())
    best, overlap = None, 0
    for phrase, category in VOCABULARY.items():
        common = len(words & set(phrase.split()))
        if common > overlap:
            best, overlap = category, common
    return best


def _nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> list[int]:
    import torch
    from torchvision.ops import nms

    if len(boxes) == 0:
        return []
    return nms(torch.as_tensor(boxes, dtype=torch.float32), torch.as_tensor(scores, dtype=torch.float32), iou).tolist()


class Detector:
    def __init__(self, device: str, model_id: str = DETECTOR_ID, threshold: float = 0.3, text_threshold: float = 0.25):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.torch = torch
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()
        self.threshold = threshold
        self.text_threshold = text_threshold
        self.model_id = model_id

    def _run(self, image: Image.Image, phrases: list[str], threshold: float) -> dict:
        w, h = image.size
        self.processor.image_processor.size = {"shortest_edge": min(DEFAULT_SHORT_SIDE, min(w, h)),
                                               "longest_edge": min(DEFAULT_LONG_SIDE, max(w, h))}
        inputs = self.processor(images=image, text=[phrases], return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        return self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=threshold, text_threshold=self.text_threshold,
            target_sizes=[(h, w)],
        )[0]

    def find(self, image: Image.Image, phrases: list[str], threshold: float = 0.3) -> list[dict]:
        """Boxes for phrases outside the inventory vocabulary (a reference card), best first."""
        result = self._run(image, phrases, threshold)
        labels = result.get("text_labels", result.get("labels"))
        found = [{"phrase": str(label), "score": round(score, 3), "box": [round(v, 1) for v in box]}
                 for box, score, label in zip(result["boxes"].tolist(), result["scores"].tolist(), labels)]
        return sorted(found, key=lambda d: -d["score"])

    def detect(self, image: Image.Image) -> list[dict]:
        w, h = image.size
        result = self._run(image, PROMPT, self.threshold)
        found = []
        labels = result.get("text_labels", result.get("labels"))
        for box, score, label in zip(result["boxes"].tolist(), result["scores"].tolist(), labels):
            category = _canonical(str(label))
            if category is None:
                continue
            x0, y0, x1, y1 = (max(0.0, box[0]), max(0.0, box[1]), min(float(w), box[2]), min(float(h), box[3]))
            area = (x1 - x0) * (y1 - y0) / (w * h)
            # A box covering nearly the whole frame is usually the prompt matching the scene.
            if area <= 0 or (area > 0.9 and category not in {"bookshelf", "rug"}):
                continue
            found.append({"category": category, "phrase": str(label), "score": round(score, 3),
                          "box": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)], "area_frac": round(area, 4),
                          "truncated": _touches_edge((x0, y0, x1, y1), w, h)})
        return resolve_overlaps(_per_category_nms(found))


def _touches_edge(box: tuple[float, float, float, float], w: int, h: int, margin: float = 0.01) -> bool:
    """True when a box is cut off by the frame border, so its size is only a lower bound."""
    x0, y0, x1, y1 = box
    return x0 <= margin * w or y0 <= margin * h or x1 >= (1 - margin) * w or y1 >= (1 - margin) * h


def _iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _inside_fraction(inner: list[float], outer: list[float]) -> float:
    ix = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return ix * iy / area if area > 0 else 0.0


SHELF_BOARD_LIKE = {"table", "desk"}


def resolve_overlaps(found: list[dict], *, same_box_iou: float = 0.7, inside: float = 0.8) -> list[dict]:
    """Remove contradictory labels within one frame.

    - One object, two labels (sofa and chair, desk and table): boxes overlapping by
      same_box_iou or more keep only the higher score. Books are exempt; they sit inside
      shelves by nature.
    - A table or desk box lying almost entirely inside a bookshelf box is a shelf board.
    """
    ranked = sorted(found, key=lambda d: -d["score"])
    shelves = [d["box"] for d in ranked if d["category"] == "bookshelf"]
    kept: list[dict] = []
    for det in ranked:
        if det["category"] in SHELF_BOARD_LIKE and any(_inside_fraction(det["box"], s) >= inside for s in shelves):
            continue
        if det["category"] != "book" and any(k["category"] not in {"book", det["category"]} and _iou(det["box"], k["box"]) >= same_box_iou
                                             for k in kept):
            continue
        kept.append(det)
    return kept


def _per_category_nms(found: list[dict], iou: float = 0.5) -> list[dict]:
    kept = []
    for category in sorted({d["category"] for d in found}):
        group = [d for d in found if d["category"] == category]
        keep = _nms(np.array([d["box"] for d in group]), np.array([d["score"] for d in group]), iou)
        kept.extend(group[i] for i in keep)
    return sorted(kept, key=lambda d: -d["score"])


