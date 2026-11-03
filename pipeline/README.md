# Sweep pipeline (offline)

Turns one sweep video into room measurements, located non-book objects, and the list of frames
that show book spines (input for the OCR stage, not built yet).

```
video ─► frames ─► detect ─► spines                      (frames.json, detections.json, spine_frames.json, previews/)
            └────► reconstruct ─► room ─► items          (scale.json, reconstruction.npz, room.json, floorplan.png,
                                                          room_points.ply, items.json)
```

## Setup (macOS, Apple Silicon or CUDA)

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r pipeline/requirements.txt
.venv/bin/pip install --no-deps git+https://github.com/facebookresearch/vggt.git
```

Models download from Hugging Face on first run: VGGT-1B (~5 GB), Depth Pro (~2 GB), Grounding DINO base (~1 GB).

## Run

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -m pipeline.run --video my_sweep.mp4 --out sweeps/my-room
.venv/bin/python -m pipeline.run --out sweeps/my-room --force room,items   # recompute later stages only
.venv/bin/python -m unittest discover -s pipeline/tests -t .                # unit tests
```

Useful options: `--recon-frames` (frames given to VGGT, default 24; memory grows quickly),
`--keep-fps` (frames kept per second, default 2), `--depth-model`
(e.g. `depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf`), `--device`.

## Stages

| Stage | Model / method | Output |
|---|---|---|
| frames | sample 6 fps, keep the sharpest frame per 0.5 s (variance of Laplacian), skip near-duplicates of the last kept frame (≥ 95% overlap after shift and grey difference < 10), save up to 1920 px | `frames/*.jpg` (the `frame_ref`s), `frames.json` with sharpness, blurry flag, skipped duplicates |
| detect | Grounding DINO base, one prompt with the whole vocabulary, run at the frame's own resolution (0.42 s per 480×854 frame on an M5). Per-category non-max suppression; same box with two labels keeps the higher score; table or desk boxes inside a bookshelf are shelf boards and dropped; boxes touching the frame edge are marked truncated | `detections.json` |
| spines | rule: not blurry and (≥ 3 books, or a bookshelf filling ≥ 15% of the frame with at least one book) | `spine_frames.json`, annotated `previews/` |
| reconstruct | VGGT-1B gives cameras and depth for all frames in one unitless space; Depth Pro gives metric depth per frame; scale = median ratio on confident pixels, leaving out close-up frames (scene under 1 m) | `reconstruction.npz`, `scale.json` (scale, per-frame ratios, relative spread, excluded close-ups) |
| room | RANSAC planes; up from the largest horizontal plane; floor and ceiling = lowest and highest horizontal planes; walls = outermost tall vertical planes on two right-angled axes; top-down outline for non-rectangular rooms | `room.json`, `floorplan.png`, `room_points.ply` |
| items | detection boxes → nearest-depth points in the reconstruction → 3D boxes in the room frame; same-category boxes overlapping in 3D (10 cm padding) are one object; sizes from views not cut off by the frame edge, else the union of partial views (flagged); floor-standing furniture measured floor to top | `items.json`, `shelved_wall_area_m2` in `room.json` |

## Book agent (read → identify → price)

```bash
.venv/bin/python -m pipeline.run --out sweeps/my-room --stages ocr,identify,price \
    --country DE --currency EUR --second-country GB --second-currency GBP
```

| Stage | Method | Output |
|---|---|---|
| ocr | docTR (db_resnet50 + parseq) on every spine frame, three times: upright, rotated 90° each way (spines read top-to-bottom or bottom-to-top). Raw lines are stored | `spine_text.json` |
| identify | lines on one spine strip are joined; each strip and each neighbouring pair is a candidate. Garbled readings are filtered, near-identical readings across frames share one Open Library query. A work is accepted only if most title words were read **and** the author was read, or the title has ≥ 3 significant words and the reading is mostly that title. `id_confidence` is this match score. Edition is left empty | `books_identified.json` (identified books with frame_refs and evidence, plus readable spines that matched nothing) |
| price | Google Books sale info per country: home country → same-currency countries (eurozone for EUR) → US, GB, CA, AU, IN. A foreign price is converted with the ECB reference rate and labelled `converted` with rate, date and source. Pre-1900 works and prices above `--appraisal-threshold` (default 150) become `needs_appraisal`, unpriced. No price → empty amount and a review entry | `books_priced.json`, `books_priced_<CC>.json` |

The price stage needs `GOOGLE_BOOKS_API_KEY` in the environment or in `.env` at the repo root
(a Google Cloud API key with the Books API enabled; an AI Studio Gemini key does not work).
Google Books prices are Google Play eBook prices, so replacement cost is stated as a new eBook
edition and used value stays empty until a used-market source is added. Network responses
are cached in `.cache/http/` with their original retrieval time.

## Metric scale, and its limits

Scale comes only from the metric depth model (a deliberate choice: no reference object).
Monocular metric depth is typically 5–15% off indoors, and that error passes straight into
every length (areas get roughly twice the percentage error). `scale.json` reports the
per-frame ratios; a large `relative_iqr` means the depth model disagreed with itself across
frames, and `room.json → confidence` is reduced accordingly. If the room bars (floor within 10%,
walls within 15%) are missed, the first fix to try is a known-size anchor.

## Known limits
- Wall area is gross: windows and doors are detected but not yet subtracted.
- Only detections in the reconstruction frames are placed in 3D; raise `--recon-frames` for more coverage.
- Depth (`d`) of an object is a single-view estimate.
- The Manhattan assumption (walls at right angles) is built in; angled walls are flagged, not measured.
