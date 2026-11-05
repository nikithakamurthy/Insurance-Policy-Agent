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


