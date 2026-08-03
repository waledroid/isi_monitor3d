# Zone background-crop capture tool — design

**Date:** 2026-08-03
**Goal:** stop yolo26l/yolo26n from detecting the empty flat wooden pallet support as
`palette` by adding hard-negative **background images** (and optionally in-domain
positives) to `trainer/isidet/data/pallet3_yolo_seg`.

## Problem

The dataset (4998 train / 1005 val, classes `palette, carton, polybag`) contains
**zero background images**, and none of its images look like what the detector
actually sees at inference: gray polygon-filled zone crops
(`zone_crop_polygon_fill` default on). The empty wooden support is therefore
out-of-distribution and gets hallucinated as a palette.

## Decisions (brainstormed 2026-08-03)

1. **Frame source:** live cameras — `/dev/shm` frame bus (`FrameShmReader`) when
   isistream runs, RTSP fallback via the `rtsp` FrameSource plugin otherwise.
   CPU-only; never builds a detector; safe beside the live stack.
2. **Crop style:** gray polygon-filled, byte-matching inference
   (`zone_fill_polygons` + the `_fill_outside` recipe, `_FILL_GRAY`).
3. Backgrounds are **class-agnostic**: a saved background must contain zero
   instances of ALL three classes (human review before merge).
4. **Counts:** ~250 backgrounds to start (5% of train), ceiling ~500 (10%);
   diversity (lighting, shadows, passers-by) beats volume. Optional second
   session with palettes present → ~100–200 in-domain positives, labeled via
   the existing LabelMe → `labelme_to_yolo.py` flow.

## Tool: `tools/capture_zone_bg.py`

Standalone script, no changes to isistream/Backbone. Usage:

```bash
conda activate monitor3d
python tools/capture_zone_bg.py --config config/backbone.yaml \
    [--out trainer/isidet/data/bg_captures] [--prefix bg] \
    [--interval 2.0] [--count 300] [--min-diff 4.0] [--cams cam_a,cam_b]
```

### Flow

1. Load `backbone.yaml` → `CameraRig.from_file(cfg["calibration_path"])`,
   `ZoneRegistry.load(cfg["zones_path"])`. **Refuse to run** (exit 2) if no
   zones are configured.
2. Compute inference-identical geometry once:
   `zone_crop_boxes(rig, zones, crop_height_m=det.zone_crop_height_m)` and
   `zone_fill_polygons(rig, zones, crop_height_m=…)` — same calls, same
   defaults as `isistream.core._build_object_detector`.
3. Per camera, open a frame provider:
   - `FrameShmReader(cam_id).latest()` — preferred; retries transparently.
   - If the bus is absent/stale for > 5 s, fall back to
     `frame_source_registry.create("rtsp", …)` from the camera's config entry.
   - A camera with neither logs a warning and is skipped (others continue).
4. Every `--interval` seconds, per camera × visible zone:
   - Scale the calibration-frame box to the actual frame size
     (`sx = fw/calib_w, sy = fh/calib_h` — same as `ZoneScopedDetector.detect`).
   - Crop, apply the gray outside-polygon fill (poly scaled by `(sx, sy)`,
     shifted by crop origin, dilated by `round(dilate_px*(sx+sy)/2)`,
     `cv2.fillPoly` + `cv2.dilate`, outside = `_FILL_GRAY`).
   - **Dedup:** save only if mean absolute difference vs the last *saved* crop
     for that (cam, zone) ≥ `--min-diff` (grayscale, resized to 64×64; first
     crop always saves).
   - Write `{prefix}_{cam}_{zone}_{NNNN}.jpg` (quality 95) to `--out`.
5. Stop at `--count` total saved images or Ctrl-C; print a per-zone tally.

### Merge procedure (manual, documented in the tool's docstring)

1. Eyeball `bg_captures/` — delete any crop containing any object of any class.
2. Copy ~90% into `images/train/`, ~10% into `images/val/` — **no label
   files** (that is YOLO's background convention; `labels/` untouched).
3. Retrain; if the FP persists, capture up to the 10% ceiling and retrain.

## Error handling

- No zones → hard exit with message (mirrors isistream's pose-only warning).
- Stale bus + failed RTSP → skip camera, keep running with the rest; exit 1 if
  *no* camera delivers frames within 30 s.
- Zone not visible from a camera (`zone_crop_boxes` skipped it) → naturally absent.

## Testing (`tests/test_capture_zone_bg.py`, hermetic)

- Geometry parity: for a synthetic rig + square zone, the tool's crop equals
  slicing the frame with `zone_crop_boxes` output; filled corners equal
  `_FILL_GRAY`.
- Dedup: two identical frames → 1 file; a changed frame → 2 files.
- No-zones refusal: exits non-zero with empty `ZoneRegistry`.
- Downscale parity: 720p frame vs 1080p calibration → box scaled exactly as
  `ZoneScopedDetector.detect` does.

## Out of scope

- Auto-labeling of occupied-zone positives (existing LabelMe flow covers it).
- Any dashboard/UI integration; any change to `zone_scope.py` (read-only reuse).
- Retraining itself (isidet trainer, `isi-train` env — separate step).
