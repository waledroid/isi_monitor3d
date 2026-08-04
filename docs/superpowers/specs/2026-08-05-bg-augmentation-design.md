# Background photometric augmentation — design

**Date:** 2026-08-05
**Goal:** grow the crop384 dataset's background negatives from 89 to ~445
(≈8% of train) with offline LIGHTING-variation augmentation, since the
platform's position is stable and Ultralytics' online HSV jitter already
covers plain brightness.

## Decisions (brainstormed 2026-08-05)

1. **Photometric only, no geometry** (except horizontal flip p=0.5 — free
   viewpoint variety, safe because backgrounds carry no labels). The
   valuable effects are the ones YOLO's train pipeline does NOT do:
   synthetic shadows, directional gradients, color temperature, gamma
   crush, sensor noise.
2. **4 variants per original** → 89 + 356 = 445 backgrounds ≈ 8% of the
   5397-crop train split (inside the 5–10% band).
3. **Split cooperation:** variant filenames must preserve the capture
   series prefix that `make_crop_dataset.bg_split` hashes, so every
   variant lands in the SAME train/val split as its source series (no
   synthetic-only series drifting into val).

## Tool: `tools/augment_backgrounds.py`

```bash
conda activate monitor3d
python tools/augment_backgrounds.py \
    --src trainer/isidet/data/grouped_backgrounds \
    --out trainer/isidet/data/grouped_backgrounds_aug \
    [--variants 4] [--seed 0]
```

### Behaviour

1. Refuse (exit 2) if `--out` exists. Copy every readable image from
   `--src` into `--out` byte-identical; unreadable files warn + skip.
2. For each original, write `--variants` augmented JPEGs (quality 95).
   Per variant, compose 2–4 effects drawn by a seeded
   `np.random.default_rng(seed)` (deterministic across runs):
   - **gamma** uniform 0.55–1.6 (LUT);
   - **contrast/brightness** alpha 0.7–1.3, beta −25..+25;
   - **color temperature** ±12%: scale B up & R down (cool) or R up & B
     down (warm);
   - **directional gradient**: multiply by a linear ramp in a random
     direction, far side falling to 0.6–1.0;
   - **shadow bands**: 1–2 random quadrilaterals multiplied by
     0.45–0.75, edges softened with Gaussian blur (kernel ~31 px);
   - **sensor noise**: additive Gaussian sigma 2–6;
   - **horizontal flip** at p=0.5 (independent of the 2–4 effect draw).
   Result clipped to uint8; dimensions/dtype unchanged.
3. **Variant naming:** original `<series>_<NNNN>.jpg` (the capture tool's
   shape, NNNN < 1000) → variant v (1-based) is
   `<series>_<v*1000 + NNNN:04d>.jpg` — e.g. `..._Sortie_1_0007.jpg` →
   `..._Sortie_1_1007.jpg`, `_2007.jpg`. The `_<digits>.jpg` suffix shape
   is preserved, so `bg_split`'s prefix-strip yields the same series
   group ⇒ same split as the original; collisions impossible.
   Fallback for names without a `_<digits>` suffix: append `_<v>000.jpg`
   style indices (`<stem>_1000.jpg`, `_2000.jpg`, …) — same guarantee.
4. Exit summary: originals copied, variants written, skipped files.

## Rebuild step (operator, after the tool runs)

Delete `trainer/isidet/data/pallet3_yolo_seg_crop384/` and re-run
`make_crop_dataset` with `--backgrounds trainer/isidet/data/grouped_backgrounds_aug`.
Expected: ~5397 labeled train crops + ~445 backgrounds.

## Testing (`tests/test_augment_backgrounds.py`, hermetic)

- Count & naming: 2 originals × 3 variants → 8 files; variant of
  `foo_bar_0007.jpg` named `foo_bar_1007.jpg` / `foo_bar_2007.jpg` /
  `foo_bar_3007.jpg`.
- Split cooperation: `make_crop_dataset.bg_split(variant_name) ==
  bg_split(original_name)` for every variant (import via the same
  importlib-by-path pattern the other tool tests use).
- Determinism: two runs with the same seed produce byte-identical
  variant files; different seeds differ.
- Variants differ from source: mean abs pixel diff > 2.0.
- Originals in `--out` byte-identical to `--src`.
- Dims/dtype unchanged (384×384×3 uint8 in, same out).
- Refusal: existing `--out` → exit 2.

## Out of scope

- Augmenting labeled positives (polygon-aware augmentation — separate
  problem, online HSV covers most of it).
- The rebuild itself and retraining (operator steps).
- Any change to `make_crop_dataset.py`.
