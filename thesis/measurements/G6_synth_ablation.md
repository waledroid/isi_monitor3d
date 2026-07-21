# G6 — Synthetic-vs-real training ablation (polybag)

**Date:** 2026-07-20 · **Env:** `isi-train` (ultralytics 8.4.22, torch 2.10.0+cu128,
RTX 5070) · **Goal:** quantify what the isiGen synthetic data contributes for the
polybag class, evaluated on real imagery only.

## Design (adapted per G0 provenance findings)

Single-class (`polybag`) instance segmentation, three training arms with
**identical** model/hyperparameters, all evaluated on one common **real** test
set never seen by any arm.

- **Common real test set** (`G6_ablation/test_real/`): all 186 polybag-containing
  `colis-*` images of the **pallet3 val split** (215 instances), labels filtered
  to class `polybag` (pallet3 class id 2 → 0). Real photos (G0 §2); pallet3's
  colis train/val split is leak-free by construction (`preserve_colis_split: true`).
- **Arm S — synthetic-only** (238 train / 29 val): the `syn*.png` images of
  `trainer/isiGen/data/black_polybag/export/yolo_seg` (SDXL+ControlNet+LoRA
  generations, CLIP-filtered export). *Deviation from the raw export:* the
  export also contains the 53 real LoRA-source photos; these were **excluded**
  so the arm is purely synthetic and cannot overlap real data.
- **Arm R — real-only** (238 train / 29 val): random subsample (seed 42) of the
  737 polybag-containing `colis-*` images of the **pallet3 train split**,
  matched in count to Arm S; labels filtered/remapped identically.
- **Arm R+S — merged** (476 train / 58 val): union of the two.

Leakage checks: test images come only from pallet3 *val*; Arm R only from
pallet3 *train* (disjoint, split preserved from the source dataset); Arm S
synthetic images share zero bytes with pallet3 (G0: 0 size/hash collisions).

## Commands (run strictly sequentially on the GPU)

Dataset build: `thesis/measurements/raw/g6_build_datasets.py` (inlined python,
reproduced there) → `thesis/measurements/G6_ablation/{test_real,arm_S,arm_R,arm_RS}/`.

```bash
conda activate isi-train && cd /home/aatanda/isi_monitor3d/trainer/isidet
G6=/home/aatanda/isi_monitor3d/thesis/measurements/G6_ablation
# per arm A in {arm_S, arm_R, arm_RS}:
yolo segment train model=yolo26n-seg.pt data=$G6/$A/data.yaml \
  imgsz=640 epochs=80 batch=8 workers=4 seed=0 device=0 \
  project=$G6/runs name=$A exist_ok=True plots=False
yolo segment val model=$G6/runs/$A/weights/best.pt data=$G6/test_real/data.yaml \
  imgsz=640 batch=8 workers=4 device=0 project=$G6/runs name=eval_${A}_on_real
```

## Results — common real test set (186 images, 215 polybag instances)

| Arm | Train imgs (real/syn) | Box P | Box R | Box mAP@0.5 | Box mAP@0.5:0.95 | Mask mAP@0.5 | Mask mAP@0.5:0.95 |
|---|---|---|---|---|---|---|---|
| S (synthetic-only) | 0 / 238 | 0.319 | 0.391 | 0.223 | 0.185 | 0.219 | 0.179 |
| R (real-only) | 238 / 0 | 0.922 | 0.885 | 0.941 | 0.927 | 0.946 | 0.930 |
| R+S (merged) | 238 / 238 | 0.906 | 0.935 | 0.962 | 0.950 | 0.965 | 0.950 |

(Arm S sanity: on its own synthetic val split it reached box mAP@0.5 ≈ 0.936 —
the drop to 0.223 on real frames is the sim-to-real gap, not a failed run.
Training times: arm_S 80 epochs ok; arm_R 0.262 h; arm_RS 0.497 h; all exit 0.
Raw per-run outputs: `G6_ablation/runs/{arm_S,arm_R,arm_RS}/results.csv` and
`G6_ablation/runs/eval_{S,R,RS}_on_real/`; training logs `G6_ablation/runs_arm_*.log`.)

## Observation

On the 186-image real test set, the synthetic-only model reaches box mAP@0.5
0.223 (vs 0.936 on synthetic val), the real-only model 0.941, and the merged
model 0.962. Adding the 238 synthetic images to the 238 real ones raises box
mAP@0.5 by +0.021 (0.941 → 0.962), box mAP@0.5:0.95 by +0.023 (0.927 → 0.950),
mask mAP@0.5:0.95 by +0.020 (0.930 → 0.950), and recall by +0.050 (0.885 →
0.935), with precision moving from 0.922 to 0.906. Interpretation is deferred to
the Discussion section.

## Caveats

- The test set contains only polybag-positive real images from one site/camera
  family (the pallet3 colis source); false positives on object-free frames are
  not measured.
- One seed per arm (seed=0); differences of ~0.02 mAP are within single-seed
  training noise for 238-image datasets — flag as such in the thesis or re-run
  with 2 more seeds if the margin becomes a headline claim.
- The 53 real LoRA-source photos were excluded from Arm S; their potential
  overlap with the pallet3 colis pool was therefore irrelevant to leakage.

## Post-review leakage check (2026-07-20, main session)

FULL.REVIEW-1 flagged one unchecked channel: the 53 real LoRA-source photos vs
the 186-image real test set. md5 over all image bytes: **0 overlaps** (53 vs
186 files). The generator never saw any test image; the ablation's leakage
guarantees are complete.
