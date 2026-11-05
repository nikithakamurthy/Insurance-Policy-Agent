"""Room length and width from a corner view: Depth Pro scaled by paper sheets, no VGGT (experimental).

The policyholder tapes a sheet of printer paper (A4 or US Letter) on each of two adjacent walls,
stands in the corner diagonally opposite them and holds the phone still on each sheet in turn.
Nothing here depends on the room: it measures the two walls the sheets are on, from where the
phone is.

1. Every sampled frame: the sheet is found (Grounding DINO + SAM, the paper's proportions, plain
   white) and Depth Pro gives depth and a focal length.
2. One focal length for the clip: the median of the sheet frames (one phone, no zoom). Depth Pro's
   per-frame focal varied by 20% on one clip, and every distance scales with it.
3. The sheet's length in pixels from its four corners refined to sub-pixel on the image gradient
   (the segmentation outline includes edge blur, about 6% on a 50 px sheet).
4. The frame's scale: the sheet's true distance (focal x sheet length / pixels, pinhole model)
   over Depth Pro's depth at the sheet. Depth Pro's depths times that factor are metric.
5. Planes in the scaled 3D points: the floor gives "up" (no floor in view: frame not used, a
   tilted phone would tilt every wall); vertical planes are walls; the sheet must lie flat on one.
6. The camera's horizontal perpendicular distance to the sheet's wall.

Each wall's distance is the median of its frames. Length and width are the two distances plus the
phone's distance from the corner walls behind it (corner_offset_m, stated in the result). In the
live app the voice agent labels each hold (wall 1, wall 2: CornerRoomJob); on a recorded clip the
frames are grouped into walls by time (a gap means the camera turned to the other sheet).

    python -m pipeline.room_corner --video clip.mp4 [--paper A4] [--corner-offset 0.5] [--start S --end E]
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .planes import extract_planes, voxel_downsample

PAPER_M = {"A4": (0.297, 0.210), "Letter": (0.2794, 0.2159)}   # ISO 216 A4; US Letter
SHEET_PHRASES = ["sheet of paper", "white paper", "paper"]
SHEET_ASPECT_TOLERANCE = 0.08     # A4 (1.41) and Letter (1.29) differ by 9%
CORNER_OFFSET_M = 0.5      # default: phone held in front of the body, standing in the corner
WALL_GAP_S = 1.5           # sheet frames further apart than this belong to different walls
MAX_DEPTH_M = 15.0
MIN_SHEET_PX = 40          # a shorter sheet (too far away) makes the scale too coarse: ~2.5% per pixel
SHEET_TO_WALL_M = 0.30     # the sheet's wall: the vertical plane nearest the sheet, within this (window recesses)
MAX_SHEET_SATURATION = 60  # 0-255: plain paper is close to grey; a coloured print or book cover is not
MAX_SHEET_TILT_DEG = 25    # the sheet is taped on a wall: its own plane is vertical (a bag on the floor is not)


def find_sheet(detector: Any, segmenter: Any, image: Image.Image, paper: str) -> dict | None:
    """The paper sheet in a frame, or None. The outline must have the sheet's proportions; the
    segmenter runs only when the detector found something."""
    from .measure import fit_rectangle, touches_edge

    hits = detector.find(image, SHEET_PHRASES, threshold=0.3)[:3]
    if not hits:
        return None
    long_m, short_m = PAPER_M[paper]
    segmenter.set_image(image)
    for hit in hits:
        for mask, score in sorted(segmenter.masks(box=hit["box"]), key=lambda m: -m[1]):
            if mask.sum() < 400 or touches_edge(mask):
                continue
            long_px, short_px, _, _ = fit_rectangle(mask)
            if short_px > 0 and abs(long_px / short_px / (long_m / short_m) - 1) <= SHEET_ASPECT_TOLERANCE:
                return {"long_px": float(long_px), "short_px": float(short_px), "mask": mask,
                        "detector_score": hit["score"], "sam_score": round(float(score), 3)}
    return None


def plain_paper(image: Image.Image, mask: np.ndarray) -> bool:
    """A printer sheet is white: low colour saturation under any lighting."""
    hsv = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2HSV)
    return float(np.median(hsv[..., 1][mask])) <= MAX_SHEET_SATURATION


def sheet_length_px(image: Image.Image, mask: np.ndarray, fallback: float) -> tuple[float, str]:
    """The sheet's long side in pixels from its four corners, refined to sub-pixel on the image.
    Falls back to the outline's rectangle when the outline is not a clean quadrilateral."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return fallback, "outline rectangle"
    contour = max(contours, key=cv2.contourArea)
    quad = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
    if len(quad) != 4:
        return fallback, "outline rectangle"
    grey = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY).astype(np.float32)
    corners = quad.reshape(4, 1, 2).astype(np.float32)
    cv2.cornerSubPix(grey, corners, (3, 3), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01))
    c = corners.reshape(4, 2)
    sides = [float(np.linalg.norm(c[i] - c[(i + 1) % 4])) for i in range(4)]
    long_px = max((sides[0] + sides[2]) / 2, (sides[1] + sides[3]) / 2)     # opposite sides averaged
    if not 0.8 * fallback <= long_px <= 1.1 * fallback:                  # refinement ran off: keep the outline
        return fallback, "outline rectangle"
    return long_px, "sub-pixel corners"


def sample_frames(video: Path, step_s: float) -> list[tuple[float, Image.Image]]:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    every = max(1, round(fps * step_s))
    out, index = [], 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if index % every == 0:
            ok, frame = cap.retrieve()
            if ok:
                out.append((index / fps, Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))))
        index += 1
    cap.release()
    return out


def points_from_depth(depth: np.ndarray, focal: float, stride: int = 3) -> np.ndarray:
    """Camera-frame 3D points: x right, y down, z forward."""
    h, w = depth.shape
    v, u = np.mgrid[0:h:stride, 0:w:stride]
    z = depth[v, u]
    keep = (z > 0.2) & (z < MAX_DEPTH_M)
    u, v, z = u[keep], v[keep], z[keep]
    return np.stack([(u - w / 2) * z / focal, (v - h / 2) * z / focal, z], axis=1)


def measure_frame(image: Image.Image, mask: np.ndarray, long_px: float, depth: np.ndarray, focal: float,
                  sheet_long_m: float) -> dict:
    """Scale the frame by its sheet, then the camera's horizontal perpendicular distance to the sheet's wall."""
    if mask.sum() < 20 or long_px <= 0:
        return {"wall_distance_m": None, "reason": "sheet outline empty", "sheet_long_px": round(long_px, 1)}
    inner = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    sheet_depth = float(np.median(depth[inner if inner.sum() > 20 else mask]))
    true_distance = focal * sheet_long_m / long_px
    factor = true_distance / sheet_depth
    scaled = depth * factor
    result = {"scale_factor": round(factor, 3), "sheet_long_px": round(long_px, 1), "sheet_distance_m": round(true_distance, 3)}
    if long_px < MIN_SHEET_PX:
        return {**result, "wall_distance_m": None, "reason": f"sheet only {long_px:.0f} px long; move closer or film at a higher resolution"}

    planes = extract_planes(voxel_downsample(points_from_depth(scaled, focal), 0.03), threshold=0.03, max_planes=10)
    camera_up = np.array([0.0, -1.0, 0.0])          # a hint only: which horizontal plane is below the camera
    horizontal = [p for p in planes if abs(p.normal @ camera_up) > np.cos(np.radians(30))]
    floor = max((p for p in horizontal if p.offset * (p.normal @ camera_up) < 0), key=lambda p: len(p.inliers), default=None)
    if floor is None:
        return {**result, "wall_distance_m": None, "reason": "floor not in view: cannot tell which way is up"}
    up = floor.normal * np.sign(floor.normal @ camera_up)
    walls = [p for p in planes if abs(p.normal @ up) < np.sin(np.radians(15))]
    result.update(camera_height_m=round(abs(floor.offset), 3), walls_found=len(walls))

    ys, xs = np.nonzero(mask)
    z = scaled[ys, xs]
    sheet_pts = np.stack([(xs - image.width / 2) * z / focal, (ys - image.height / 2) * z / focal, z], axis=1)
    sheet_normal = np.linalg.svd(sheet_pts - sheet_pts.mean(axis=0), full_matrices=False)[2][-1]
    tilt = 90 - np.degrees(np.arccos(min(1.0, abs(float(sheet_normal @ up)))))
    if tilt > MAX_SHEET_TILT_DEG:
        return {**result, "wall_distance_m": None, "reason": f"sheet not on a vertical surface ({tilt:.0f} deg from vertical)"}

    sheet_point = np.array([(xs.mean() - image.width / 2) * true_distance / focal,
                            (ys.mean() - image.height / 2) * true_distance / focal, true_distance])
    wall = min(walls, key=lambda p: abs(p.normal @ sheet_point - p.offset), default=None)
    gap = abs(wall.normal @ sheet_point - wall.offset) if wall is not None else None
    if wall is None or gap > SHEET_TO_WALL_M:
        return {**result, "wall_distance_m": None, "reason": "no wall plane at the sheet"}
    horizontal_normal = wall.normal - (wall.normal @ up) * up
    horizontal_normal /= np.linalg.norm(horizontal_normal)
    distance = abs(wall.offset) / abs(wall.normal @ horizontal_normal)
    return {**result, "sheet_to_wall_m": round(gap, 3), "wall_distance_m": round(distance, 3)}


def observe(image: Image.Image, paper: str, detector: Any, segmenter: Any, depth_model: Any) -> dict | None:
    """One frame: the sheet (paper-shaped, plain white) and Depth Pro's depth and focal length; None
    when no sheet is in view. Measuring waits until all frames are in, for one focal length."""
    sheet = find_sheet(detector, segmenter, image, paper)
    if not sheet or not plain_paper(image, sheet["mask"]):
        return None
    depth, focal = depth_model(image)
    long_px, how = sheet_length_px(image, sheet["mask"], sheet["long_px"])
    return {"image": image, "mask": sheet["mask"], "long_px": long_px, "how": how, "depth": depth, "focal": focal}


def measure_observations(observations: list[dict], paper: str, focal: float) -> list[dict]:
    return [{**measure_frame(o["image"], o["mask"], o["long_px"], o["depth"], focal, PAPER_M[paper][0]),
             "sheet_length_from": o["how"], "frame_focal_px": round(o["focal"], 1),
             **{k: o[k] for k in ("t", "frame_ref") if k in o}} for o in observations]


def summarise_walls(walls: list[tuple[str, list[dict]]], corner_offset_m: float) -> dict:
    """walls: (label, that wall's frame measurements). Length and width from the two walls measured
    with the most frames, plus the corner offset."""
    out_walls, flags = [], []
    for label, frames in walls:
        d = [f["wall_distance_m"] for f in frames if f.get("wall_distance_m") is not None]
        reasons = sorted({f["reason"] for f in frames if f.get("reason")})
        out_walls.append({"wall": label, "frames": len(frames), "measured": len(d),
                          "distance_m": round(statistics.median(d), 3) if d else None,
                          "spread_m": round(max(d) - min(d), 3) if len(d) > 1 else None, "reasons": reasons})
    measured = [w for w in out_walls if w["distance_m"] is not None]
    out = {"corner_offset_m": corner_offset_m, "walls": out_walls, "flags": flags}
    if len(measured) > 2:
        flags.append(f"{len(measured)} sheet views: the two with most frames are used; label each wall when filming")
    if len(measured) < 2:
        flags.append("fewer than two walls measured: length and width left empty")
        return out
    two = sorted(measured, key=lambda w: -w["measured"])[:2]
    a, b = sorted((w["distance_m"] + corner_offset_m for w in two), reverse=True)
    return {**out, "length_m": round(a, 2), "width_m": round(b, 2)}


def group_walls(found: list[dict]) -> list[list[dict]]:
    groups: list[list[dict]] = []
    for f in sorted(found, key=lambda f: f["t"]):
        if groups and f["t"] - groups[-1][-1]["t"] <= WALL_GAP_S:
            groups[-1].append(f)
        else:
            groups.append([f])
    return groups


def summarise(found: list[dict], corner_offset_m: float) -> dict:
    """A recorded clip: walls are the sheet views separated in time."""
    return summarise_walls([(f"{g[0]['t']}-{g[-1]['t']} s", g) for g in group_walls(found)], corner_offset_m)


class CornerRoomJob:
    """Room length and width for one live sweep. The voice agent asks the policyholder to hold the
    phone still on each wall's sheet; each hold's frames arrive as one call (add_wall). A second
    hold on the same wall replaces the first. The result is recomputed whenever a wall is added,
    with one focal length (the median) over all frames."""

    def __init__(self, corner_offset_m: float = CORNER_OFFSET_M):
        self.corner_offset_m = corner_offset_m
        self.paper = "A4"
        self.observations: dict[str, list[dict]] = {}
        self.held: dict[str, int] = {}
        self.result: dict | None = None
        self.state, self.error = "not_started", None
        self.measuring = 0                       # walls being measured right now
        self._lock = threading.Lock()

    def add_wall(self, wall: str, paper: str, frames: list[tuple[str, Path]], detector: Any, segmenter: Any,
                 depth_model: Any) -> dict:
        with self._lock:
            self.measuring += 1
            self.state = "measuring"
        found = []
        try:
            for ref, path in frames:
                o = observe(Image.open(path).convert("RGB"), paper, detector, segmenter, depth_model)
                if o:
                    found.append({**o, "frame_ref": ref})
        finally:
            with self._lock:
                self.measuring -= 1
        with self._lock:
            self.paper = paper if paper in PAPER_M else "A4"
            self.observations[wall] = found
            self.held[wall] = len(frames)
            self.result = self._compute()
            self.state = "measuring" if self.measuring else "done"
            mine = next(w for w in self.result["walls"] if w["wall"] == wall)
        return {"wall": wall, "frames": len(frames), "sheet_found_in": len(found), "measured": mine["measured"],
                "reasons": mine["reasons"] or ([] if found else ["sheet not found: is it plain white paper, taped flat and in view?"])}

    def _compute(self) -> dict:
        everything = [o for obs in self.observations.values() for o in obs]
        if not everything:
            return {"walls": [{"wall": w, "frames": self.held.get(w, 0), "measured": 0, "distance_m": None, "spread_m": None,
                               "reasons": ["sheet not found"]} for w in sorted(self.observations)],
                    "flags": ["no sheet found"], "corner_offset_m": self.corner_offset_m}
        focal = statistics.median(o["focal"] for o in everything)
        walls = [(w, measure_observations(self.observations[w], self.paper, focal)) for w in sorted(self.observations)]
        long_m, short_m = PAPER_M[self.paper]
        return {**summarise_walls(walls, self.corner_offset_m), "paper": self.paper, "focal_px": round(focal, 1),
                "method": f"Depth Pro depth scaled by a {self.paper} sheet ({long_m * 1000:.0f} x {short_m * 1000:.0f} mm) on each "
                          f"of two adjacent walls, from the opposite corner; distance to each wall plus {self.corner_offset_m} m "
                          "for the phone's distance from the corner",
                "frames_used": sorted(f"room_frames/{o['frame_ref']}" for o in everything)}

    def status(self) -> dict:
        with self._lock:
            return {"state": self.state, "room": self.result, "error": self.error}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--paper", choices=sorted(PAPER_M), default="A4")
    parser.add_argument("--corner-offset", type=float, default=CORNER_OFFSET_M, help="phone's distance from each corner wall, m")
    parser.add_argument("--step", type=float, default=0.5, help="seconds between sampled frames")
    parser.add_argument("--start", type=float, default=0.0, help="use frames from this second (the still corner view)")
    parser.add_argument("--end", type=float, default=None, help="... up to this second")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    from .common import pick_device
    from .detect import Detector
    from .measure import DepthModel, Segmenter

    device = pick_device()
    detector, segmenter, depth_model = Detector(device), Segmenter(device), DepthModel(device)
    frames = [(t, im) for t, im in sample_frames(args.video, args.step) if t >= args.start and (args.end is None or t <= args.end)]
    print(f"{len(frames)} frames sampled from {args.video.name}", flush=True)

    # Pass 1: the sheet and Depth Pro on every frame that shows one.
    seen = [{**o, "t": round(t, 2)} for t, image in frames if (o := observe(image, args.paper, detector, segmenter, depth_model))]
    if not seen:
        print("no sheet found")
        return
    # Pass 2: one focal length for the clip, then each frame's measurement.
    focal = statistics.median(o["focal"] for o in seen)
    found = measure_observations(seen, args.paper, focal)
    for m in found:
        print(f"  t={m['t']:5.1f}s sheet {m['sheet_long_px']:5.1f}px ({m['sheet_length_from']}), Depth Pro x{m.get('scale_factor')}, "
              f"wall {m['wall_distance_m']} m {m.get('reason', '')}", flush=True)
    report = {"video": str(args.video), "paper": args.paper, "focal_px": round(focal, 1), **summarise(found, args.corner_offset),
              "frames": found}
    print(json.dumps({k: v for k, v in report.items() if k != "frames"}, indent=2))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
