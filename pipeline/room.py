"""Stage 5: room dimensions and areas from a metric point cloud.

Method
1. Downsample the cloud and fit planes with RANSAC, one after another.
2. "Up" is the normal of the largest roughly-horizontal plane, signed to agree with the
   cameras' average up direction (phones are held roughly upright during a sweep).
3. Floor is the lowest well-supported horizontal plane, ceiling the highest.
4. Wall directions come from the vertical planes (Manhattan assumption: walls meet at
   right angles). Walls are the outermost tall vertical planes along each axis; bookshelf
   fronts and furniture sides sit inside them and are ignored.
5. Length and width are wall-to-wall distances. A top-down occupancy outline checks for a
   non-rectangular footprint (an L shape, for example); if clearly found, its polygon area and
   perimeter are used instead of the rectangle.
6. Wall area is gross: perimeter times height, with windows and doors not subtracted.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

M2_TO_FT2 = 10.7639


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


def remove_outliers(points: np.ndarray, voxel: float, min_neighbours: int = 3) -> np.ndarray:
    """Drop isolated points: keep those whose 3x3x3 voxel neighbourhood holds enough points."""
    keys = np.floor(points / (voxel * 2)).astype(np.int64)
    uniq, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    lookup = {tuple(k): c for k, c in zip(uniq.tolist(), counts.tolist())}
    offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    neighbourhood = np.array([sum(lookup.get((k[0] + a, k[1] + b, k[2] + c), 0) for a, b, c in offsets) for k in uniq.tolist()])
    return points[neighbourhood[inverse.ravel()] >= min_neighbours]


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


def _orient(plane: Plane, direction: np.ndarray) -> Plane:
    if plane.normal @ direction < 0:
        return Plane(-plane.normal, -plane.offset, plane.inliers)
    return plane


def measure_room(points: np.ndarray, up_hint: np.ndarray, *, voxel: float = 0.04) -> dict:
    """Measure a room from a metric point cloud. up_hint is the cameras' mean up direction."""
    pts = remove_outliers(voxel_downsample(points, voxel), voxel)
    planes = extract_planes(pts, threshold=max(0.02, voxel * 0.75))
    flags: list[str] = []
    up_hint = up_hint / np.linalg.norm(up_hint)

    horizontal_like = [p for p in planes if abs(p.normal @ up_hint) > np.cos(np.radians(30))]
    if horizontal_like:
        up = _orient(max(horizontal_like, key=lambda p: len(p.inliers)), up_hint).normal
    else:
        up = up_hint
        flags.append("no horizontal plane found; up taken from camera orientation")

    heights = pts @ up
    support = max(50, int(0.01 * len(pts)))
    horizontals = sorted((_orient(p, up) for p in planes if abs(p.normal @ up) > np.cos(np.radians(12)) and len(p.inliers) >= support),
                         key=lambda p: p.offset)
    floor_h = horizontals[0].offset if horizontals else float(np.quantile(heights, 0.01))
    ceiling = [p for p in horizontals if p.offset - floor_h > 1.5]
    ceiling_h = ceiling[-1].offset if ceiling else float(np.quantile(heights, 0.99))
    if not horizontals:
        flags.append("floor plane not found; used 1st percentile of heights")
    if not ceiling:
        flags.append("ceiling plane not found; used 99th percentile of heights")
    height = ceiling_h - floor_h

    # Horizontal basis aligned to the dominant wall direction.
    e1 = np.cross(up, [1.0, 0, 0] if abs(up[0]) < 0.9 else [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    verticals = [p for p in planes if abs(p.normal @ up) < np.sin(np.radians(12))]
    if verticals:
        angles = np.array([np.arctan2(p.normal @ e2, p.normal @ e1) for p in verticals])
        weights = np.array([len(p.inliers) for p in verticals], dtype=float)
        theta = float(np.angle((weights * np.exp(4j * angles)).sum()) / 4)
    else:
        theta = 0.0
        flags.append("no wall planes found; footprint from point extents")
    a1 = np.cos(theta) * e1 + np.sin(theta) * e2
    a2 = np.cross(up, a1)

    def walls_along(axis: np.ndarray) -> list[float]:
        offsets = []
        for p in verticals:
            if abs(p.normal @ axis) < np.cos(np.radians(20)):
                continue
            span = np.ptp(p.inliers @ up)
            if span >= 0.5 * height:   # tall enough to be a wall, not a desk side
                offsets.append(float(p.inliers.mean(axis=0) @ axis))
        return sorted(offsets)

    footprint = np.stack([pts @ a1, pts @ a2], axis=1)
    inside = (heights > floor_h + 0.05) & (heights < ceiling_h - 0.05)
    dims, walls_found = [], 0
    for i, axis in enumerate((a1, a2)):
        offsets = walls_along(axis)
        lo, hi = np.quantile(footprint[inside, i], [0.01, 0.99]) if inside.any() else (0.0, 0.0)
        if len(offsets) >= 2 and offsets[-1] - offsets[0] > 1.0:
            lo, hi = offsets[0], offsets[-1]
            walls_found += 2
        else:
            walls_found += len(offsets[:1])
            flags.append(f"axis {i + 1}: {len(offsets)} wall plane(s); extent partly from point percentiles")
        dims.append((float(lo), float(hi)))
    length, width = sorted((d[1] - d[0] for d in dims), reverse=True)

    shape, floor_area, perimeter, polygon = _footprint_shape(footprint[inside], dims, length * width)
    if shape == "rectangle":
        floor_area, perimeter = length * width, 2 * (length + width)
    wall_area = perimeter * height

    confidence = 1.0 - 0.15 * (4 - walls_found) - 0.15 * (not horizontals) - 0.15 * (not ceiling)
    return {
        "length_m": round(length, 2), "width_m": round(width, 2), "height_m": round(height, 2),
        "floor_area_m2": round(floor_area, 2), "floor_area_ft2": round(floor_area * M2_TO_FT2, 1),
        "wall_area_m2": round(wall_area, 2), "wall_area_ft2": round(wall_area * M2_TO_FT2, 1),
        "perimeter_m": round(perimeter, 2), "shape": shape,
        "shape_method": "Manhattan walls from RANSAC planes; top-down occupancy outline for non-rectangular rooms",
        "wall_area_method": "gross: perimeter x height, openings not subtracted",
        "walls_found": walls_found, "floor_plane_found": bool(horizontals), "ceiling_plane_found": bool(ceiling),
        "room_type": "enclosed" if walls_found == 4 else "open_or_partially_seen",
        "room_type_note": ("all four walls measured wall to wall" if walls_found == 4 else
                           f"{walls_found} of 4 walls seen as planes (open plan, or walls not swept); the missing "
                           "sides use the extent of what was seen, so length, width and areas cover the observed space"),
        "geometry_confidence": round(max(0.0, confidence), 2), "flags": flags,
        "frame": {"up": up.tolist(), "axis1": a1.tolist(), "axis2": a2.tolist(),
                  "floor_h": floor_h, "ceiling_h": ceiling_h, "extents": dims, "footprint_polygon": polygon},
        "planes": [{"normal": p.normal.round(3).tolist(), "offset": round(p.offset, 3), "points": len(p.inliers)} for p in planes],
    }


def _footprint_shape(xy: np.ndarray, dims: list[tuple[float, float]], rect_area: float, cell: float = 0.05):
    """Outline the occupied footprint; call it a polygon only if it is clearly not a rectangle."""
    if len(xy) < 100:
        return "rectangle", rect_area, 0.0, []
    (x0, x1), (y0, y1) = dims
    grid = np.zeros((int((y1 - y0) / cell) + 3, int((x1 - x0) / cell) + 3), dtype=np.uint8)
    ix = np.clip(((xy[:, 0] - x0) / cell).astype(int) + 1, 0, grid.shape[1] - 1)
    iy = np.clip(((xy[:, 1] - y0) / cell).astype(int) + 1, 0, grid.shape[0] - 1)
    grid[iy, ix] = 255
    grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "rectangle", rect_area, 0.0, []
    outline = max(contours, key=cv2.contourArea)
    filled = np.zeros_like(grid)
    cv2.drawContours(filled, [outline], -1, 255, thickness=cv2.FILLED)
    approx = cv2.approxPolyDP(outline, 0.3 / cell, True)
    area = float(cv2.contourArea(approx)) * cell * cell
    perimeter = float(cv2.arcLength(approx, True)) * cell
    polygon = [[round(float(p[0][0]) * cell + x0, 2), round(float(p[0][1]) * cell + y0, 2)] for p in approx]
    rect_perimeter = 2 * ((x1 - x0) + (y1 - y0))
    # Only a clean outline counts as a non-rectangular room. A ragged one (many corners, long
    # perimeter) usually means parts of the room were not seen, so keep the wall rectangle.
    clean = 6 <= len(approx) <= 12 and perimeter <= 1.6 * rect_perimeter
    if clean and area < 0.85 * rect_area:
        return "polygon", area, perimeter, polygon
    return "rectangle", rect_area, 0.0, polygon


def up_hint_from_cameras(extrinsic: np.ndarray) -> np.ndarray:
    """Mean camera up direction in world coordinates (OpenCV cameras: +y points down)."""
    return -extrinsic[:, 1, :3].mean(axis=0)


def floorplan_image(points: np.ndarray, room: dict, cam_centres: np.ndarray, path, cell: float = 0.02) -> None:
    """Top-down debug view: points, measured rectangle or polygon, camera path."""
    f = room["frame"]
    a1, a2, up = map(np.array, (f["axis1"], f["axis2"], f["up"]))
    h = points @ up
    keep = (h > f["floor_h"] + 0.05) & (h < f["ceiling_h"] - 0.05)
    xy = np.stack([points[keep] @ a1, points[keep] @ a2], axis=1)
    (x0, x1), (y0, y1) = f["extents"]
    pad = 0.5
    W, H = int((x1 - x0 + 2 * pad) / cell), int((y1 - y0 + 2 * pad) / cell)
    img = np.full((H, W, 3), 255, np.uint8)

    def px(x, y):
        return int((x - x0 + pad) / cell), int((y - y0 + pad) / cell)

    for x, y in xy[:: max(1, len(xy) // 200000)]:
        u, v = px(x, y)
        if 0 <= u < W and 0 <= v < H:
            img[v, u] = (150, 150, 150)
    cv2.rectangle(img, px(x0, y0), px(x1, y1), (0, 0, 220), 2)
    if room["shape"] == "polygon" and f["footprint_polygon"]:
        cv2.polylines(img, [np.array([px(x, y) for x, y in f["footprint_polygon"]])], True, (220, 0, 0), 2)
    path_px = np.array([px(c @ a1, c @ a2) for c in cam_centres])
    cv2.polylines(img, [path_px], False, (0, 160, 0), 2)
    label = f"{room['length_m']} x {room['width_m']} x {room['height_m']} m  floor {room['floor_area_m2']} m2"
    cv2.putText(img, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), img)
