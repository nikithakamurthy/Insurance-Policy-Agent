"""Planes in a 3D point cloud: voxel downsampling and RANSAC plane fitting, one plane after another."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Plane:
    normal: np.ndarray
    offset: float          # plane: normal . p = offset
    inliers: np.ndarray    # points belonging to the plane


def voxel_downsample(points: np.ndarray, voxel: float) -> np.ndarray:
    keys = np.floor(points / voxel).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    out = np.zeros((len(counts), 3))
    for axis in range(3):
        out[:, axis] = np.bincount(inverse, weights=points[:, axis]) / counts
    return out


def fit_plane(points: np.ndarray) -> tuple[np.ndarray, float]:
    centre = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centre, full_matrices=False)
    normal = vt[-1]
    return normal, float(normal @ centre)


def ransac_plane(points: np.ndarray, threshold: float, rng: np.random.Generator, iterations: int = 400) -> np.ndarray:
    """Return a boolean inlier mask for the best plane."""
    n = len(points)
    best = np.zeros(n, dtype=bool)
    for start in range(0, iterations, 50):
        idx = rng.integers(0, n, size=(min(50, iterations - start), 3))
        a, b, c = points[idx[:, 0]], points[idx[:, 1]], points[idx[:, 2]]
        normals = np.cross(b - a, c - a)
        norms = np.linalg.norm(normals, axis=1)
        valid = norms > 1e-9
        normals, a = normals[valid] / norms[valid, None], a[valid]
        offsets = np.einsum("ij,ij->i", normals, a)
        distances = np.abs(points @ normals.T - offsets)
        counts = (distances < threshold).sum(axis=0)
        if len(counts) and counts.max() > best.sum():
            best = distances[:, counts.argmax()] < threshold
    if best.sum() >= 3:
        normal, offset = fit_plane(points[best])
        best = np.abs(points @ normal - offset) < threshold
    return best


def extract_planes(points: np.ndarray, *, threshold: float = 0.03, max_planes: int = 14,
                   min_fraction: float = 0.015, seed: int = 0) -> list[Plane]:
    rng = np.random.default_rng(seed)
    remaining = points
    planes = []
    min_inliers = max(50, int(min_fraction * len(points)))
    for _ in range(max_planes):
        if len(remaining) < min_inliers:
            break
        mask = ransac_plane(remaining, threshold, rng)
        if mask.sum() < min_inliers:
            break
        normal, offset = fit_plane(remaining[mask])
        planes.append(Plane(normal, offset, remaining[mask]))
        remaining = remaining[~mask]
    return planes
