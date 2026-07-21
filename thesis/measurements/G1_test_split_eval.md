# G1 — Held-out test-split evaluation

**Date:** 2026-07-20 · **Env:** `isi-train` (ultralytics 8.4.22, torch 2.10.0+cu128,
RTX 5070) · **Planned per PLAN.md §5:** evaluate the best detectors on the
pallet3_coco *test* split (1,050 images) "that no reported metric has used".

## Outcome: the planned experiment is impossible — no held-out test split exists

Provenance check (full evidence in `G0_data_provenance.md` §5): `pallet3_coco/test`
is a **byte-identical duplicate** of `pallet3_coco/valid` — all 1,049 filenames
identical (`comm -12` → 1,049), image bytes identical (md5 spot checks), and
`_annotations.coco.json` md5-identical. This is by design: `scripts/yolo_seg_to_coco.py`
(lines 8–9, 133) duplicates val into test because "RF-DETR requires the folder".
`pallet3_yolo_seg` has only train/val. **Any metric computed on "test" is a val
metric.** The thesis must not present a test-split number; the Limitations section
should state that all detector accuracy is validation-split accuracy.

## What was run instead: independent re-evaluation (reproduction) of the headline model on val

This verifies that the reported T1 numbers trace to an executable artifact
(same 1,049 val images; not new held-out evidence).

### (a) YOLO26l-seg best.pt — command

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate isi-train
cd /home/aatanda/isi_monitor3d/trainer/isidet
yolo segment val \
  model=runs/segment/models/yolo/yolo26l-seg_e200_640px_09-06-2026_00-24-57/weights/best.pt \
  data=data/pallet3_yolo_seg/data.yaml imgsz=640 batch=8 workers=4 split=val device=0
```

**Class order verified:** `data/pallet3_yolo_seg/data.yaml` → `nc: 3`,
`names: ['palette', 'carton', 'polybag']` (the eval uses this file directly, so
label↔class mapping is authoritative by construction).

### Results (1,049 images, 1,436 instances)

| Class | Imgs | Inst | Box P | Box R | Box mAP@0.5 | Box mAP@0.5:0.95 | Mask P | Mask R | Mask mAP@0.5 | Mask mAP@0.5:0.95 |
|---|---|---|---|---|---|---|---|---|---|---|
| **all** | 1049 | 1436 | 0.950 | 0.951 | **0.977** | **0.948** | 0.948 | 0.949 | **0.972** | **0.921** |
| palette | 808 | 1047 | 0.973 | 0.945 | 0.989 | 0.933 | 0.969 | 0.940 | 0.974 | 0.895 |
| carton | 74 | 174 | 0.909 | 0.948 | 0.960 | 0.943 | 0.909 | 0.948 | 0.960 | 0.900 |
| polybag | 186 | 215 | 0.967 | 0.958 | 0.983 | 0.969 | 0.967 | 0.958 | 0.980 | 0.967 |

Speed: 1.2 ms preprocess / 10.3 ms inference / 1.2 ms postprocess per image (batch 8).
Raw outputs (PR curves, confusion matrices): `thesis/measurements/raw/g1_val_yolo26l/`.

**Cross-check vs reported:** run's `report.md` states best epoch 154/172 →
mAP@50 0.977, mAP@50-95 0.947, P 0.960 / R 0.939. Re-eval reproduces box mAP@50
0.977 and mAP@50-95 0.948 (0.001 diff — TTA/EMA rounding); P/R differ slightly
(0.950/0.951 vs 0.960/0.939) as ultralytics reports P/R at a different
F1-optimal confidence point per run. Mask mAPs (0.972/0.921) were not in the
original report headline and are newly recorded here.

**GPU context:** the dashboard process (`monitor_web`, untouched per campaign
rules) held ≈4.5 GB VRAM throughout; the eval fit within the remaining budget.

### (b) RF-DETR medium-seg — SKIPPED (nothing to gain)

Its trainer's metric machinery (`models/rfdetr/rfdetr-medium-seg_e41_432px/metrics.csv`)
already reports val-split COCO metrics, and the "test" split is the same data.
Running its harness on test would re-produce the val numbers under a misleading
label. No new information exists to extract; skipped per the honesty rule.
