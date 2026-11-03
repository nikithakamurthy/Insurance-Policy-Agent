"""Stage 1: extract the sharpest frames from a sweep video and save them as frame_refs."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def sharpness(image_bgr: np.ndarray, width: int = 640) -> float:
    """Variance of the Laplacian on a downscaled grey copy. Higher is sharper."""
    h, w = image_bgr.shape[:2]
    small = cv2.resize(image_bgr, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def _thumb(image_bgr: np.ndarray, width: int = 160) -> np.ndarray:
    grey = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(grey, (width, round(grey.shape[0] * width / grey.shape[1])), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(small, (5, 5), 0).astype(np.float32)


def similarity(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(overlap, difference) between two thumbnails.

    Overlap is the shared area after the best whole-image shift (phase correlation).
    Difference is the mean absolute grey-level difference (0-255) over that shared area,
    which catches what a shift cannot: stepping closer, rotating, or a blank wall where
    phase correlation finds no shift at all.
    """
    (dx, dy), _ = cv2.phaseCorrelate(a, b)
    h, w = a.shape
    overlap = max(0.0, 1 - abs(dx) / w) * max(0.0, 1 - abs(dy) / h)
    if overlap < 0.5:
        return overlap, 255.0
    aligned = cv2.warpAffine(b, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h))
    mx, my = int(abs(dx)) + 2, int(abs(dy)) + 2
    return overlap, float(np.abs(aligned[my:h - my, mx:w - mx] - a[my:h - my, mx:w - mx]).mean())


def is_duplicate(a: np.ndarray, b: np.ndarray, *, min_overlap: float = 0.95, max_diff: float = 10.0) -> bool:
    overlap, diff = similarity(a, b)
    return overlap >= min_overlap and diff < max_diff


def extract_frames(
    video: Path,
    out_dir: Path,
    *,
    sample_fps: float = 6.0,
    keep_fps: float = 2.0,
    max_side: int = 1920,
    blur_ratio: float = 0.35,
    skip_duplicates: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Sample the video at sample_fps and keep the sharpest frame in each 1/keep_fps window.

    A frame that shows almost the same view as the last kept frame (the camera paused) is
    skipped; if it is clearly sharper, it replaces that frame instead. Comparing with the
    last kept frame, not the previous one, stops a slow pan from being skipped step by step.

    Frames are saved at up to max_side pixels on the long side, which keeps spine text
    readable for OCR. A frame is flagged blurry when its sharpness is below blur_ratio
    times the median of the kept frames. Returns (kept frames, skipped duplicates).
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps / sample_fps))
    window = 1.0 / keep_fps
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    best: dict[int, tuple[float, float, np.ndarray]] = {}
    index = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if index % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                t = index / fps
                bucket = int(t // window)
                score = sharpness(frame)
                if bucket not in best or score > best[bucket][0]:
                    best[bucket] = (score, t, frame)
        index += 1
    cap.release()
    if not best:
        raise ValueError(f"No frames decoded from {video}")

    kept: list[tuple[int, float, float, np.ndarray, np.ndarray]] = []
    skipped = []
    for bucket in sorted(best):
        score, t, frame = best[bucket]
        thumb = _thumb(frame)
        if skip_duplicates and kept and is_duplicate(kept[-1][4], thumb):
            last = kept[-1]
            if score > 1.2 * last[1]:
                kept[-1] = (bucket, score, t, frame, thumb)
                skipped.append({"id": f"f{last[0]:05d}", "t": round(last[2], 3), "duplicate_of": f"f{bucket:05d}", "reason": "replaced by sharper"})
            else:
                skipped.append({"id": f"f{bucket:05d}", "t": round(t, 3), "duplicate_of": f"f{last[0]:05d}", "reason": "near-identical view"})
            continue
        kept.append((bucket, score, t, frame, thumb))

    records = []
    for bucket, score, t, frame, _ in kept:
        h, w = frame.shape[:2]
        scale = min(1.0, max_side / max(h, w))
        if scale < 1.0:
            frame = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        frame_id = f"f{bucket:05d}"
        path = frames_dir / f"{frame_id}.jpg"
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        records.append({
            "id": frame_id, "t": round(t, 3), "path": str(path.relative_to(out_dir)),
            "width": frame.shape[1], "height": frame.shape[0], "sharpness": round(score, 1),
        })
    median = float(np.median([r["sharpness"] for r in records]))
    for r in records:
        r["blurry"] = r["sharpness"] < blur_ratio * median
    return records, skipped


def select_spread(frames: list[dict], count: int) -> list[dict]:
    """Pick up to count non-blurry frames evenly spread across the sweep."""
    usable = [f for f in frames if not f["blurry"]] or frames
    if len(usable) <= count:
        return usable
    picks = np.linspace(0, len(usable) - 1, count).round().astype(int)
    return [usable[i] for i in sorted(set(picks))]
