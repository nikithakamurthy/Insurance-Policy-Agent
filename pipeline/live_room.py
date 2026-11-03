"""Room dimensions and areas for a live sweep, after it ends.

The live app saves one frame per second of the whole session (room_frames/), so walls, floor
and ceiling are in view, not only the books. When the sweep ends:

1. About 24 room frames spread over the session, plus a frame where the bank card was found,
   go to VGGT (reconstruct.py): one shared 3D scene, unitless, put into metres with Depth Pro.
2. Bank card scale: the card's width in pixels and the camera's focal length give its true
   distance (85.60 mm card, pinhole model); VGGT's distance to the card in the Depth Pro-scaled
   scene is compared with it, and the whole scene is rescaled by that one factor. Several card
   frames are combined by the median and their spread is reported. Without a card, the Depth Pro
   scale stays and the result is labelled estimated.
3. room.py fits floor, ceiling and walls (RANSAC planes, Manhattan walls); items.py places the
   bookcases for the shelved wall area.

Results: room.json, reconstruction.npz and floorplan.png in the sweep folder.
"""

from __future__ import annotations

import logging
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image

from .common import read_json, write_json
from .telemetry import stage

logger = logging.getLogger(__name__)
RECON_FRAMES = 24
CARD_FRAMES = 2
CARD_LONG_M = 0.0856


def card_scale_factor(card_long_px: float, orig_width: int, recon: dict, index: int, mask: np.ndarray) -> float | None:
    """How much the reconstruction must be scaled so the card is at its true distance.

    In the reconstruction's pixels the card's long side is card_long_px * (w / orig_width); with
    the reconstruction's focal length fx, a card 85.6 mm long at that size is fx * 0.0856 / px
    metres away. The reconstruction's own depth at the card, divided into that, is the factor."""
    w, h = recon["size"]
    long_px = card_long_px * w / orig_width
    if long_px < 8:            # a few pixels across at VGGT's resolution: too coarse to trust
        return None
    fx = float(recon["intrinsic"][index][0, 0])
    small = cv2.resize(mask.astype(np.uint8), (int(w), int(h)), interpolation=cv2.INTER_NEAREST).astype(bool)
    depths = recon["depth"][index][small]
    if depths.size < 10:
        return None
    true_distance = fx * CARD_LONG_M / long_px
    return float(true_distance / np.median(depths))


def rescale(recon: dict, factor: float) -> dict:
    out = dict(recon)
    out["points"] = recon["points"] * factor
    out["depth"] = recon["depth"] * factor
    extrinsic = recon["extrinsic"].copy()
    extrinsic[:, :, 3] *= factor
    out["extrinsic"] = extrinsic
    return out


def measure_live_room(sweep_dir: Path, card_frames: list[str], measurer: Any, detector: Any, device: str) -> dict:
    """Room from a live sweep's room frames, scaled by the bank card where one was seen."""
    from .frames import select_spread
    from .items import locate_items
    from .reconstruct import reconstruct
    from .room import floorplan_image, measure_room, up_hint_from_cameras

    room_frames = read_json(sweep_dir / "room_frames.json")["frames"] if (sweep_dir / "room_frames.json").exists() else []
    if len(room_frames) < 8:
        raise ValueError(f"only {len(room_frames)} room frames saved; at least 8 are needed")
    chosen = select_spread(room_frames, RECON_FRAMES - min(CARD_FRAMES, len(card_frames)))
    chosen += [{"id": ref, "path": f"frames/{ref}.jpg", "blurry": False} for ref in card_frames[:CARD_FRAMES]]

    with stage("room: reconstruction (VGGT + Depth Pro)"):
        scale = reconstruct(sweep_dir, chosen, device)
    data = np.load(sweep_dir / "reconstruction.npz")
    recon = {k: data[k] for k in data.files}

    factors = []
    with stage("room: bank card scale"):
        for i, frame_id in enumerate(recon["frame_ids"]):
            if str(frame_id) not in card_frames:
                continue
            image = Image.open(sweep_dir / "frames" / f"{frame_id}.jpg").convert("RGB")
            from .measure import find_card
            measurer.segmenter.set_image(image)
            card = find_card(detector, measurer.segmenter, image, [])
            if card:
                factor = card_scale_factor(card["long_px"], image.width, recon, i, card["mask"])
                if factor:
                    factors.append({"frame_ref": str(frame_id), "factor": round(factor, 4)})
    if factors:
        factor = statistics.median(f["factor"] for f in factors)
        recon = rescale(recon, factor)
        spread = (max(f["factor"] for f in factors) - min(f["factor"] for f in factors)) / factor if len(factors) > 1 else None
        scale_method = (f"bank card (85.60 x 53.98 mm) seen in {len(factors)} frame(s); Depth Pro's scale was off by "
                        f"{(factor - 1) * 100:+.0f}% and was corrected")
        estimated = False
    else:
        factor, spread = 1.0, None
        scale_method = "estimated: Depth Pro metric depth (no bank card found in the reconstructed frames)"
        estimated = True

    with stage("room: planes and walls"):
        conf = recon["conf"]
        mask = conf >= np.quantile(conf, 0.4)
        points = recon["points"][mask]
        room = measure_room(points, up_hint_from_cameras(recon["extrinsic"]))
    with stage("room: bookcases (Grounding DINO)"):
        detections = {"frames": {}}
        for frame in chosen:
            image = Image.open(sweep_dir / frame["path"]).convert("RGB")
            detections["frames"][frame["id"]] = detector.detect(image)
        items = locate_items(recon, detections, room)
    floorplan_image(points, room, -np.einsum("sij,si->sj", recon["extrinsic"][:, :, :3], recon["extrinsic"][:, :, 3]),
                    sweep_dir / "floorplan.png")

    card_quality = 1.0 if spread is None else max(0.0, 1 - 2 * spread)
    room.update(
        shelved_wall_area_m2=items["shelved_wall_area_m2"], bookshelf_units=items.get("bookshelf_units"),
        scale_method=scale_method, scale_estimated=estimated, card_scale_factors=factors,
        depth_pro_relative_iqr=scale["relative_iqr"], frames_used=[str(f) for f in recon["frame_ids"]],
        confidence=round(room["geometry_confidence"] * (card_quality if not estimated else max(0.0, 1 - 2 * scale["relative_iqr"])), 2),
    )
    write_json(sweep_dir / "room.json", room)
    return room


class RoomJob:
    """One room measurement per sweep, run in the background after the sweep ends."""

    def __init__(self, sweep_dir: Path):
        self.sweep_dir = Path(sweep_dir)
        self.state, self.result, self.error, self.seconds = "not_started", None, None, None

    def status(self) -> dict:
        return {"state": self.state, "room": self.result, "error": self.error, "seconds": self.seconds}

    def run(self, card_frames: list[str], measurer_factory: Callable[[], Any], detector: Any, device: str) -> None:
        self.state, started = "running", time.time()
        try:
            self.result = measure_live_room(self.sweep_dir, card_frames, measurer_factory(), detector, device)
            self.state = "done"
        except Exception as exc:     # a failed room measurement leaves the room empty in the packet, with the reason
            logger.exception("Room measurement failed for %s", self.sweep_dir.name)
            self.state, self.error = "failed", f"{type(exc).__name__}: {exc}"[:300]
        finally:
            self.seconds = round(time.time() - started, 1)
