"""Stage 6: place detections in the room, merge repeats across frames, estimate sizes.

Only frames used in the 3D reconstruction carry a point map, so only their detections can be
located. Each detection takes the reconstructed points inside its box, keeps the nearest depth
layer (the object, not the wall behind it), and becomes a 3D box in the room frame
(axis1, axis2, up). Same-category boxes that overlap in 3D are one object.

Sizes prefer views where the object is not cut off by the frame edge. If every view is cut off
(typical for shelf close-ups) the union of the partial views is used and the item is flagged
partial_view. Furniture standing on the floor is measured floor-to-top, because only its top
surface may be visible.
"""

from __future__ import annotations

import numpy as np

from .detect import STRUCTURAL

FLOOR_STANDING = {"table", "desk", "chair", "sofa", "bookshelf", "cabinet", "lamp"}


def _object_points(points: np.ndarray, conf: np.ndarray, depth: np.ndarray, box: list[float],
                   scale_xy: tuple[float, float], conf_min: float) -> np.ndarray:
    sx, sy = scale_xy
    x0, y0, x1, y1 = box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy
    mx, my = 0.12 * (x1 - x0), 0.12 * (y1 - y0)   # shrink to avoid edges bleeding into background
    xs = slice(int(x0 + mx), max(int(x0 + mx) + 1, int(x1 - mx)))
    ys = slice(int(y0 + my), max(int(y0 + my) + 1, int(y1 - my)))
    p, c, d = points[ys, xs].reshape(-1, 3), conf[ys, xs].ravel(), depth[ys, xs].ravel()
    ok = c >= conf_min
    if ok.sum() < 30:
        return np.empty((0, 3))
    p, d = p[ok], d[ok]
    near = np.quantile(d, 0.3)
    return p[(d > 0.85 * near) & (d < 1.15 * near + 0.15)]


def _overlap(a: dict, b: dict, pad: float) -> bool:
    """3D boxes touch or overlap once each is padded by pad metres."""
    return bool(np.all(a["lo"] - pad <= b["hi"]) and np.all(b["lo"] - pad <= a["hi"]))


def _clusters(dets: list[dict], pad: float) -> list[list[dict]]:
    parent = list(range(len(dets)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(dets)):
        for j in range(i + 1, len(dets)):
            if _overlap(dets[i], dets[j], pad):
                parent[find(i)] = find(j)
    groups: dict[int, list[dict]] = {}
    for i, d in enumerate(dets):
        groups.setdefault(find(i), []).append(d)
    return [sorted(g, key=lambda d: (d["truncated"], -d["score"])) for g in groups.values()]


def _size(cluster: list[dict], floor_h: float, category: str) -> tuple[dict, bool]:
    whole = [d for d in cluster if not d["truncated"]]
    if whole:
        ext = np.median([d["hi"] - d["lo"] for d in whole], axis=0)
        lo = np.median([d["lo"] for d in whole], axis=0)
        hi = np.median([d["hi"] for d in whole], axis=0)
    else:
        lo = np.min([d["lo"] for d in cluster], axis=0)
        hi = np.max([d["hi"] for d in cluster], axis=0)
        ext = hi - lo
    w, d = sorted(ext[:2].tolist(), reverse=True)
    h = float(ext[2])
    bottom = float(lo[2] - floor_h)
    if category in FLOOR_STANDING and bottom < 0.25:
        h = float(hi[2] - floor_h)
    return {"w": round(w * 100), "h": round(h * 100), "d": round(d * 100), "bottom_above_floor_m": round(bottom, 2),
            "centre_m": [round(v, 2) for v in ((lo + hi) / 2).tolist()]}, not whole


def locate_items(recon: dict, detections: dict, room: dict, *, pad_m: float = 0.1) -> dict:
    f = room["frame"]
    a1, a2, up = map(np.array, (f["axis1"], f["axis2"], f["up"]))
    w, h = recon["size"]
    ow, oh = recon["orig_size"]
    conf_min = float(np.quantile(recon["conf"], 0.3))
    located = []
    for i, frame_id in enumerate(recon["frame_ids"]):
        for det in detections["frames"].get(str(frame_id), []):
            if det["category"] == "book":
                continue
            pts = _object_points(recon["points"][i], recon["conf"][i], recon["depth"][i], det["box"], (w / ow, h / oh), conf_min)
            if len(pts) < 20:
                continue
            local = np.stack([pts @ a1, pts @ a2, pts @ up], axis=1)
            lo, hi = np.quantile(local, [0.05, 0.95], axis=0)
            located.append({"category": det["category"], "score": det["score"], "frame_ref": str(frame_id),
                            "box": det["box"], "truncated": bool(det.get("truncated", False)), "lo": lo, "hi": hi})

    items = []
    for category in sorted({d["category"] for d in located}):
        group = [d for d in located if d["category"] == category]
        for n, cluster in enumerate(sorted(_clusters(group, pad_m), key=lambda c: -len(c)), 1):
            size, partial = _size(cluster, f["floor_h"], category)
            items.append({
                "id": f"{category}-{n}", "category": category, "structural": category in STRUCTURAL,
                "frame_ref": cluster[0]["frame_ref"], "frame_refs": sorted({d["frame_ref"] for d in cluster}),
                "detections": len(cluster), "best_score": max(d["score"] for d in cluster),
                "dimensions_cm": {k: size[k] for k in ("w", "h", "d")},
                "partial_view": partial,
                "dimension_note": ("cut off in every view: size is the union of partial views, a lower bound" if partial
                                   else "w and h from whole views; d is a single-view estimate and least reliable"),
                "position_m": size["centre_m"], "bottom_above_floor_m": size["bottom_above_floor_m"],
            })

    shelves = [i for i in items if i["category"] == "bookshelf"]
    shelved = sum(i["dimensions_cm"]["w"] * i["dimensions_cm"]["h"] for i in shelves) / 1e4
    return {"items": items, "shelved_wall_area_m2": round(shelved, 2), "bookshelf_units": len(shelves),
            "located_detections": len(located),
            "note": "Only detections in reconstruction frames are located; increase --recon-frames for more coverage."}
