# Datasets & training — isidet + isiGen

**WHY** — the Backbone is **inference-only**; it consumes `.onnx` files. Training lives in `trainer/isidet/` (dedicated `isi-train` conda env — ultralytics pulls `opencv-python`, which would clash with monitor3d's conda OpenCV). The KPI: mAP@0.5 ≥ 0.90, pallet empty/full P/R ≥ 0.95/0.93.

**WHAT** — YOLO instance-**segmentation** datasets, 3 warehouse classes.

## The datasets (`trainer/isidet/data/`, counted on disk)

| Dataset | train | val | nc | classes | format |
|---|---|---|---|---|---|
| **`pallet3_yolo_seg`** (production) | 5 540 | 1 049 | 3 | `palette`, `carton`, `polybag` | YOLO-seg (`class x1 y1 x2 y2 ...` normalized polygons) |
| `pallet3_coco` | 7 638 imgs total | — | 3 | same | COCO JSON (`_annotations.coco.json`) — RF-DETR path |
| `dataset_v2` | 2 484 | 259 | 2 | `carton`, `polybag` | YOLO-seg (colis source set) |
| `pallet_seg_yolo_det` | 500 | 125 | 1 | `palette` | YOLO-seg (pallet source set) |

Images are mixed-resolution JPEGs (mostly 1280×720; some tall 1280×2276 phone shots). Sources are traceable by filename: `pallets-*` (LabelMe pallet set) vs `colis-*` (carton/polybag set). The production set is **built, not hand-assembled**:

```bash
python ../../scripts/build_pallet3_seg.py \
  --labelme data/pallet_universal_labelme --colis data/colis_universal_dataset \
  --out data/pallet3_yolo_seg --force
```

```yaml
# data/pallet3_yolo_seg/data.yaml — verbatim
path: .../trainer/isidet/data/pallet3_yolo_seg
train: images/train
val: images/val
nc: 3
names: ['palette', 'carton', 'polybag']
```

## Training — HOW

```bash
conda activate isi-train           # NEVER install the training stack into monitor3d
cd trainer/isidet
python scripts/run_train.py --config configs/train_pallet.yaml        # yolo26l-seg @ 640
python scripts/run_train.py --config configs/train_pallet3_seg.yaml   # yolo11m-seg @ 1024
```

| Knob (from the configs) | Value | Why |
|---|---|---|
| `weights` | `yolo26l-seg.pt` / `yolo11m-seg.pt` | `-seg` suffix ⇒ mask head, auto task=segment |
| `imgsz` | 640 or **1024** | high imgsz = the main lever for far/small floor objects |
| `batch_size` / `mixed_precision` | 2–4 / AMP on | 12 GB RTX 5070 at 1024px seg |
| `copy_paste: 0.5` | high | mask-aware paste — boosts minority carton/polybag (palette ≈ 9:1 over them) |
| `camera_aug: true` | MotionBlur + JPEG compression + noise + downscale | CCTV realism |
| `workers: 2` | low | 1024px seg batches × workers exhausted WSL `/dev/shm` (bus error) |

**Export (one-shot, at train end)** — raw head so the Backbone runs its own NMS + mask decode:

```yaml
export_model: true
export_formats: ["onnx", "openvino"]
export_nms: false        # raw head — yolo_onnx_seg does NMS + mask decode itself
export_opset: 17
export_dynamic: true     # runtime-selectable inference size (dashboard imgsz slider)
```

**Where results land** — `runs/segment/models/yolo/<model>_e<epochs>_<imgsz>px_<timestamp>/` (e.g. `yolo11m-seg_e200_1024px_04-06-2026_00-59-56/`), each with weights, exports, and an auto-generated `report.md`. Detection runs go to `runs/detect/models/yolo/`; RF-DETR exports to `trainer/isidet/models/rfdetr/<ts>/`. The dashboard's Settings model dropdown scans both roots (`list_trained_onnx()`).

Then: drop the `.onnx` into `models/`, point `config/backbone.yaml`'s `detection.onnx_path` at it, sanity-check with `python tools/onnx_inspect.py <model.onnx>` and `python tools/detection_smoke.py --onnx <model> --image <jpg>`.

## isiGen — synthetic data (`trainer/isiGen/`, Studio :8200)

**WHY** — 50–100 real photos per class is not a dataset. **WHAT** — SDXL + depth-ControlNet generation turns those seed photos into an unlimited, *perfectly-labeled* synthetic set. **HOW** — an 8-phase pipeline (scaffolds → captioning → LoRA → generation → mask import → QA) ending in a **YOLO-seg export** that feeds isidet directly. Fully standalone (`./launch.sh`, `ISIGEN_*` env), no repo coupling.

!!! warning "WSL2 training gotchas"
    Heavy models (yolo11l/yolo26l) can swap-thrash the 12 GB WSL VM → transient EIO/bus-error crash (not disk-full). Levers: smaller model, `batch_size`, `workers`. The `MemoryCleanup` hook (gc + `torch.cuda.empty_cache()` per epoch) exists because a 1024px run was OOM-killed twice around epoch ~90.
