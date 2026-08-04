# Object-centric crop dataset (pallet3 → crop384) — design

**Date:** 2026-08-04
**Goal:** derive an object-centric cropped dataset from `pallet3_yolo_seg`
optimized for the YOLO nano model used by isimonitor3d zone-scoped inference
(deployed at `zone_imgsz: 320`, dataset supports 384), while preserving the
source dataset untouched.

## Problem

`pallet3_yolo_seg` (4998 train / 1005 val, classes `palette, carton,
polybag`, YOLO-seg polygon labels) is full photos where objects are small.
At inference the model sees tight zone crops letterboxed to `zone_imgsz` —
a scale + framing domain gap. Object-centric crops of the training set close
it (same logic as the zone-crop background captures).

## Decisions (brainstormed 2026-08-04)

1. **Crop from ground-truth labels, NOT a prior model's detections.** GT
   boxes are exact, and GT polygons must be remapped anyway. Model-driven
   crops risk unlabeled half-visible objects (training poison).
2. **Crop size 384×384** (`--size`, default 384). Deployed inference stays
   `zone_imgsz: 320` and the nano trains at `imgsz=320` — a pure downscale
   that mirrors the app's own letterboxing; 384 inference remains available
   with no upscaling. Regenerating other sizes = re-run the script (source
   dataset is preserved).
3. **Cluster crops** (Approach A): one square crop per cluster of
   overlapping/nearby GT objects, not per object — avoids near-duplicate
   crops for adjacent pallets.
4. **Scope:** transform all of pallet3 AND fold `grouped_backgrounds/` in as
   label-free backgrounds (90/10 train/val). `grouped_with_pallets/` joins
   later once LabelMe-labeled — out of scope here.

## Tool: `tools/make_crop_dataset.py`

```bash
conda activate monitor3d
python tools/make_crop_dataset.py \
    --src trainer/isidet/data/pallet3_yolo_seg \
    --out trainer/isidet/data/pallet3_yolo_seg_crop384 \
    [--size 384] [--backgrounds trainer/isidet/data/grouped_backgrounds] \
    [--margin 0.10 0.25] [--keep-frac 0.30] [--seed 0] [--preview N]
```

### Pipeline (per source image, split-preserving: train→train, val→val)

1. **Parse labels:** YOLO-seg lines `cls x1 y1 x2 y2 ...` (normalized) →
   pixel polygons → per-object bbox.
2. **Cluster:** expand each bbox by its margin; group boxes whose expanded
   rects intersect (union-find). Each cluster → one crop.
3. **Crop window:** union bbox of the cluster + randomized margin
   (uniform in `--margin` range, per side, seeded RNG for reproducibility)
   → squared (grow the short side) → clamped to the image → grown to at
   least `size`×`size` source pixels when the image allows (window may
   shift to stay inside). Never upscale: if the image itself is smaller
   than `size`, the crop is letterbox-padded with gray 114.
4. **Label remap:** every GT polygon intersecting the window is shifted to
   crop coordinates and clipped to the crop rectangle
   (Sutherland–Hodgman). Objects with < `--keep-frac` (default 0.30) of
   their original polygon area inside the window are NOT labeled — instead
   their in-crop pixels are painted gray 114 (matches the inference-time
   polygon fill the model already sees). Objects ≥ keep-frac keep their
   clipped polygon as a label.
5. **Resize:** window → `size`×`size` (downscale or identity; pad if
   needed), polygons scaled accordingly, renormalized, written as YOLO-seg
   lines. Degenerate clipped polygons (< 3 points or ~zero area) are
   dropped (and their pixels gray-filled, same rule).
6. **Output:** `images/{train,val}/<stem>_c<k>.jpg` (quality 95) +
   matching `labels/...txt`; `data.yaml` with the same 3 class names and
   the new root.
7. **Backgrounds:** each image from `--backgrounds` is letterboxed to
   `size`×`size` and written with NO label file — 90/10 into train/val
   (deterministic by filename hash). Skipped when the flag is omitted.
8. **Summary:** counts printed at exit — source images, crops, labels
   kept, labels gray-filled, backgrounds, per-split totals.

### `--preview N`

Writes N random annotated crops (polygons drawn) to `<out>/_preview/` and
generates nothing else — the human check that remapping is correct before
committing to a full run. Preview files are excluded from the dataset
(leading underscore dir, no entries in data.yaml splits).

## Error handling

- Refuses to run if `--out` already exists (no silent merge/overwrite).
- A label line that fails to parse: warn with file/line, skip that object.
- Image file unreadable: warn, skip image, count reported in the summary.
- An image with no labels in the source (shouldn't exist in pallet3) passes
  through as a full-frame letterboxed background (no label file).

## Testing (`tests/test_make_crop_dataset.py`, hermetic)

- Polygon clip: square polygon clipped at a crop edge keeps the inside
  region; fully-outside polygon yields nothing.
- Renormalize round-trip: synthetic polygon → crop coords → normalized →
  back ≈ original within 1 px.
- Cluster: two overlapping boxes → one cluster; two far boxes → two.
- Keep-frac: object 20% inside → no label + gray pixels at its location;
  object 80% inside → labeled.
- No-upscale: 200×200 source image → 384 crop is letterbox-padded, object
  scale unchanged.
- Split preservation: val source image produces only val crops.
- Refusal: existing `--out` → non-zero exit.

## Out of scope

- Labeling `grouped_with_pallets/` (LabelMe flow, separate task).
- Training itself (`isi-train` env; nano at imgsz=320).
- Any change to `backbone/` / the app config (`zone_imgsz: 320` set at
  deploy time).
