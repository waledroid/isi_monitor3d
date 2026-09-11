# G0 — Data provenance investigation (Task 0)

**Date:** 2026-07-20 · **Question:** were the reported detectors trained on a
"merged real+synthetic corpus", as claimed in `thesis/draft/01_introduction.md`
(contribution 4) and `thesis/draft/02_methods.md` (§2.5)?

## Verdict (definitive)

**The claim is FALSE as written.** Every detector whose accuracy is reported in
the thesis (yolo26l-seg, yolo26m-seg, yolo26n-seg@320, RF-DETR medium-seg) was
trained on `pallet3` — a corpus that contains **zero synthetic images**. The
synthetic data (500 isiGen images) exists and a genuinely merged corpus exists
(`dataset_v2`), but **no recorded training run ever consumed the merged corpus**,
and no reported metric traces to it.

**Bonus finding (kills G1 as planned):** `pallet3_coco/test` is a **byte-identical
duplicate of `pallet3_coco/valid`** — there is no independent held-out test split
anywhere in the repo (see §5).

## 1. What the reported detectors actually trained on

Every `args.yaml` under `trainer/isidet/runs/`:

```
$ grep -rh "^data:" trainer/isidet/runs --include=args.yaml | sort | uniq -c
      4 data: data/pallet3_yolo_seg/data.yaml     ← incl. yolo26l-seg_e200 (T1 headline), yolo26m-seg_e200, yolo26n-seg_e100_320px
      2 data: data/pallet_seg_yolo_seg/data.yaml
      3 data: data/pallet_v2_yolo/data.yaml
```

No run references `dataset_v2` (grep over runs/ and configs/: zero hits in runs;
configs never mention it). The RF-DETR configs (`train_pallet3_rfdetr.yaml`,
`train_pallet3_rfdetr_seg_m560.yaml`) both set `dataset_path: "data/pallet3_coco"`.
The reported RF-DETR medium-seg run (`models/rfdetr/rfdetr-medium-seg_e41_432px/`)
has tfevents epoch 1780816195 = **2026-06-07**, eleven days *before* the first
synthetic image was generated (see timeline) — it cannot have seen synthetic data.
(Two later RF-DETR runs, `22-06-2026_2226` and `24-06-2026_0905 rfdetr_nano_312`,
have empty `hparams.yaml` `{}` so their dataset is not recorded on disk; the only
RF-DETR configs on disk point at `pallet3_coco`. Neither run's metrics are the
ones cited in PLAN T1.)

## 2. pallet3 is 100 % real — four independent proofs

`trainer/isidet/data/pallet3_yolo_seg` (5,540 train / 1,049 val; classes
`[palette, carton, polybag]`) and its COCO mirror `pallet3_coco`.

1. **Filenames/format.** All 6,589 images are `pallets-NNNNN.jpg` (5,386, class
   palette) or `colis-NNNNN.jpg` (1,203, classes carton/polybag) per its own
   `dataset.json`. Zero `.png`, zero `syn*` names, zero 12-hex-char names.
   isiGen synthetic images are `syn000000.png … syn000499.png` (PNG).
2. **Hash/size cross-check.** File-size intersection of all 6,589 pallet3 images
   with all 500 isiGen generated PNGs: **0 collisions** (script below), so no
   renamed/converted copies either at the byte level.
3. **Timeline.** `pallet3` was built **2026-06-04** (dir mtimes); the *only*
   synthetic generation that ever ran (isiGen project `black_polybag`) produced
   its images **2026-06-18/19** (`generated/syn000000.png` mtime 2026-06-19 01:30).
   The other two isiGen projects (`dataset_v1`, `polybag_seg`) have **empty
   `generated/` dirs** (0 files) — 500 synthetic images is the repo-wide total
   (`find isiGen/data -path "*generated*" -name "*.png"` → 500, all black_polybag;
   `export_noclip` holds re-exports of the same 553 items, not new generations).
4. **Visual inspection.** `colis-00300.jpg` = real overhead conveyor photo
   (white polybag on belt); `pallets-00002.jpg` = real wooden pallet on asphalt;
   contrast `generated/syn000010.png` = clearly diffusion-rendered conveyor scene.
   colis-* originals date to 2026-03-20 (camera capture era). The builder
   (`scripts/build_pallet3_seg.py`, invoked with `--colis-originals-only
   --preserve-colis-split` per `dataset.json`) merged `pallet_universal_labelme`
   (real LabelMe pallet photos) + `colis_universal_dataset` (real YOLO-seg
   conveyor images; Roboflow-style `_augN` offline-augmented *copies of real
   photos* were explicitly **excluded**, `colis_originals_only: true`).

## 3. What the synthetic data actually is (isiGen `black_polybag`)

`trainer/isiGen/data/black_polybag/manifest.jsonl` (573 entries) is explicit:
hex-named files (`03c25c821f23.jpg`, …) are the **real raw photos** — each entry
carries `source_path` → `trainer/isidet/data/black/{img,bg}/…jpg` real captures —
while `syn*.png` are the SDXL+ControlNet+LoRA generations.

- Real polybag photos: **53** (49 train + 4 val in the export) — matches the
  "53 reals" LoRA claim. Plus 20 background frames.
- Generated: **500** (`generated/`, 2026-06-18/19) — matches "500 generated".
- Export `black_polybag/export/yolo_seg` (1 class `polybag`): train = 238 syn
  + 49 real = 287; val = 29 syn + 4 real = 33. **This export is itself a small
  merged real+synthetic set** — but only 267 of the 500 generations passed the
  CLIP filter into it.

## 4. The one genuinely merged corpus: `dataset_v2` — never trained on

`trainer/isidet/data/dataset_v2` (built **2026-06-22**; 2 classes
`[carton, polybag]`; 2,484 train / 259 val = 2,743 images):

| source prefix | train | val | nature |
|---|---|---|---|
| `dataset_v1__*` | 1,984 | 206 | **real** (isiGen project `dataset_v1` export = re-exported real photos; that project generated nothing) |
| `black_polybag__*` hex | 49 | 4 | **real** (the 53 LoRA source photos) |
| `black_polybag__syn*.png` | 451 | 49 | **synthetic** (all 500 generations) |

So dataset_v2 = 2,243 real + 500 synthetic (18.2 % synthetic). **No run in
`trainer/isidet/runs/` used it** and no config on disk references it.

## 5. pallet3_coco "test split" is a copy of valid

```
$ comm -12 <(ls pallet3_coco/valid) <(ls pallet3_coco/test) | wc -l   → 1049 (all)
$ md5sum {valid,test}/_annotations.coco.json                          → identical
$ md5sum {valid,test}/colis-00963.jpg                                 → identical
```

By design — `scripts/yolo_seg_to_coco.py` line 8–9 & 133: *"``test/`` duplicates
``valid/`` (RF-DETR requires the folder; swap in a real held-out set later if you
want an independent test metric)"*; split plan `[("train","train"),
("val","valid"), ("val","test")]`. **The repo contains no independent 3-class
held-out test set.** PLAN.md's G1 premise ("test split … that no reported metric
has used") is wrong: any eval on `test/` is a re-eval of `valid/`.

## 6. Required thesis corrections

- **01_introduction.md, contribution 4** — "detectors trained on the resulting
  merged real+synthetic corpus reach 0.96–0.98 mAP@0.5": the metrics belong to
  models trained on **all-real** pallet3. Either re-train on dataset_v2 and
  report those numbers, or reword: the synthetic pipeline's output was *produced
  and packaged* (dataset_v2) but the reported detector metrics come from real
  data; synthetic contribution is quantified separately (→ G6 ablation).
- **02_methods.md §2.5** — "Per-class synthetic exports are merged with real
  annotated imagery into a single three-class corpus … of 5,540 training and
  1,049 validation images": **false**. The 5,540/1,049 corpus (pallet3) is
  all-real. The actual merged corpus is 2-class dataset_v2 (2,743 images, never
  trained on).
- Any "held-out test" phrasing for pallet3_coco test must go (it is valid, duplicated).

## Commands used (all read-only)

```bash
ls / stat / md5sum / comm on trainer/isidet/data/{pallet3_yolo_seg,pallet3_coco,dataset_v2}
head trainer/isiGen/data/*/manifest.jsonl; ls trainer/isiGen/data/black_polybag/{generated,export/yolo_seg}
grep -rh "^data:" trainer/isidet/runs --include=args.yaml | sort | uniq -c
python3 size/sha256 intersection: 500 generated PNGs × 6,589 pallet3 images → 0 collisions
grep -n "test|valid" scripts/yolo_seg_to_coco.py   # test-duplicates-valid by design
```
