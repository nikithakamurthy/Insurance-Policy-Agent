"""Stage 4: multi-frame 3D reconstruction (VGGT) put into metres with a metric depth model.

VGGT returns cameras and depth for all frames in one shared, but unitless, coordinate frame.
A monocular metric depth model (Depth Pro by default) predicts depth in metres per frame.
The ratio between the two on confident pixels gives one global scale factor, which turns the
whole reconstruction into metres. The spread of per-frame ratios is reported as a confidence
signal: a consistent ratio across frames means the metric model agrees with itself.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

VGGT_ID = "facebook/VGGT-1B"
DEPTH_ID = "apple/DepthPro-hf"
PATCH = 14


def _vggt_size(width: int, height: int, target: int = 518) -> tuple[int, int]:
    """Largest side becomes target; the other side is rounded to a multiple of the patch size."""
    if width >= height:
        return target, max(PATCH, round(height * target / width / PATCH) * PATCH)
    return max(PATCH, round(width * target / height / PATCH) * PATCH), target


def _load_images(paths: list[Path]) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    with Image.open(paths[0]) as first:
        orig = first.size
    size = _vggt_size(*orig)
    images = []
    for path in paths:
        with Image.open(path) as img:
            img = img.convert("RGB")
            if img.size != orig:
                raise ValueError(f"All frames must share one size; {path} is {img.size}, expected {orig}")
            images.append(np.asarray(img.resize(size, Image.Resampling.BICUBIC), dtype=np.float32) / 255.0)
    return np.stack(images), orig, size


def run_vggt(paths: list[Path], device: str) -> dict:
    import torch
    from vggt.models.vggt import VGGT
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    images, orig, (w, h) = _load_images(paths)
    model = VGGT.from_pretrained(VGGT_ID).to(device).eval()
    tensor = torch.from_numpy(images).permute(0, 3, 1, 2).to(device)
    with torch.no_grad():
        predictions = model(tensor)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], (h, w))
    depth = predictions["depth"][0].float().cpu().numpy()            # (S, H, W, 1)
    depth_conf = predictions["depth_conf"][0].float().cpu().numpy()  # (S, H, W)
    extrinsic = extrinsic[0].float().cpu().numpy()                   # (S, 3, 4) camera-from-world
    intrinsic = intrinsic[0].float().cpu().numpy()                   # (S, 3, 3)
    points = unproject_depth_map_to_point_map(depth, extrinsic, intrinsic)
    del model, predictions
    if device == "mps":
        torch.mps.empty_cache()
    return {"images": images, "orig_size": orig, "size": (w, h), "depth": depth[..., 0], "conf": depth_conf,
            "extrinsic": extrinsic, "intrinsic": intrinsic, "points": points}


def metric_depths(paths: list[Path], size: tuple[int, int], device: str, model_id: str = DEPTH_ID) -> np.ndarray:
    """Metric depth in metres for each frame, resized to the VGGT resolution."""
    import torch
    from transformers import pipeline

    estimator = pipeline("depth-estimation", model=model_id, device=device)
    out = []
    for path in paths:
        with Image.open(path) as img:
            result = estimator(img.convert("RGB"))
        depth = result["predicted_depth"]
        depth = depth.squeeze().float().cpu().numpy() if isinstance(depth, torch.Tensor) else np.asarray(depth, dtype=np.float32)
        out.append(np.asarray(Image.fromarray(depth).resize(size, Image.Resampling.BILINEAR)))
    del estimator
    if device == "mps":
        torch.mps.empty_cache()
    return np.stack(out)


def estimate_scale(vggt_depth: np.ndarray, vggt_conf: np.ndarray, metric: np.ndarray,
                   *, conf_quantile: float = 0.5, min_m: float = 0.2, max_m: float = 15.0,
                   min_scene_m: float = 1.0, min_frames: int = 5) -> dict:
    """Global metres-per-VGGT-unit from per-frame median depth ratios on confident pixels.

    Close-up frames (median metric depth under min_scene_m, e.g. a shelf filling the view)
    are left out: monocular metric depth is least reliable at short range and those frames
    made the per-frame ratios scatter. They are kept only if too few frames would remain.
    """
    candidates = []
    for i, (d, c, m) in enumerate(zip(vggt_depth, vggt_conf, metric)):
        mask = (c >= np.quantile(c, conf_quantile)) & (d > 1e-6) & (m > min_m) & (m < max_m)
        if mask.sum() > 500:
            candidates.append((i, float(np.median(m[mask] / d[mask])), float(np.median(m[mask]))))
    if not candidates:
        raise ValueError("Metric depth and VGGT depth share no confident pixels; cannot set scale.")
    far = [c for c in candidates if c[2] >= min_scene_m]
    used = far if len(far) >= min_frames else candidates
    ratios = np.array([c[1] for c in used])
    scale = float(np.median(ratios))
    q1, q3 = np.quantile(ratios, [0.25, 0.75])
    spread = float((q3 - q1) / scale)
    return {"scale": scale, "per_frame": [round(c[1], 5) for c in used], "relative_iqr": round(spread, 4),
            "frames_used": len(used), "frames_excluded_close_up": [c[0] for c in candidates if c not in used],
            "min_scene_depth_m": min_scene_m}


def reconstruct(sweep_dir: Path, frames: list[dict], device: str, depth_model: str = DEPTH_ID) -> dict:
    paths = [sweep_dir / f["path"] for f in frames]
    print(f"  VGGT on {len(paths)} frames", flush=True)
    recon = run_vggt(paths, device)
    print("  metric depth for scale", flush=True)
    metric = metric_depths(paths, recon["size"], device, depth_model)
    scale = estimate_scale(recon["depth"], recon["conf"], metric)
    s = scale["scale"]
    extrinsic = recon["extrinsic"].copy()
    extrinsic[:, :, 3] *= s
    np.savez_compressed(
        sweep_dir / "reconstruction.npz",
        frame_ids=np.array([f["id"] for f in frames]), points=(recon["points"] * s).astype(np.float32),
        conf=recon["conf"].astype(np.float32), depth=(recon["depth"] * s).astype(np.float32),
        metric_depth=metric.astype(np.float32), extrinsic=extrinsic.astype(np.float32),
        intrinsic=recon["intrinsic"].astype(np.float32), images=(recon["images"] * 255).astype(np.uint8),
        orig_size=np.array(recon["orig_size"]), size=np.array(recon["size"]),
    )
    return {"geometry_model": VGGT_ID, "depth_model": depth_model, "scale_method": "metric_depth_alignment",
            "frames": [f["id"] for f in frames], **scale}
