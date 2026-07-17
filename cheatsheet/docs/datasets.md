# Datasets & training — isidet + isiGen

**WHY** — the Backbone is inference-only; training lives in `trainer/isidet/` (`isi-train` conda env).

**WHAT** — YOLO instance-segmentation datasets, 3 warehouse classes.

## Datasets (`trainer/isidet/data/`, counted on disk)

| Dataset | train | val | nc | classes | format |
|---|---|---|---|---|---|
| **`pallet3_yolo_seg`** (production) | 5 540 | 1 049 | 3 | `palette`, `carton`, `polybag` | YOLO-seg polygons |
| `pallet3_coco` | 7 638 total | — | 3 | same | COCO JSON (RF-DETR path) |
| `dataset_v2` | 2 484 | 259 | 2 | `carton`, `polybag` | YOLO-seg |
| `pallet_seg_yolo_det` | 500 | 125 | 1 | `palette` | YOLO-seg |

Mixed-resolution JPEGs, mostly 1280×720. Filenames trace sources: `pallets-*` vs `colis-*`. The production set is built:

```bash
python ../../scripts/build_pallet3_seg.py \
  --labelme data/pallet_universal_labelme --colis data/colis_universal_dataset \
  --out data/pallet3_yolo_seg --force
```

```yaml
# data/pallet3_yolo_seg/data.yaml
train: images/train
val: images/val
nc: 3
names: ['palette', 'carton', 'polybag']
```

## Training

Two configs for `scripts/run_train.py`: `configs/train_pallet.yaml` (yolo26l-seg @ 640) and `configs/train_pallet3_seg.yaml` (yolo11m-seg @ 1024).

| Knob | Value | Why |
|---|---|---|
| `weights` | `yolo26l-seg.pt` / `yolo11m-seg.pt` | `-seg` ⇒ mask head |
| `imgsz` | 640 or **1024** | main lever for far/small objects |
| `batch_size` / AMP | 2–4 / on | 12 GB RTX 5070 |
| `copy_paste: 0.5` | high | boosts minority carton/polybag (palette ≈ 9:1) |
| `camera_aug: true` | blur + compression + noise | CCTV realism |
| `workers: 2` | low | WSL `/dev/shm` limit |

**Export** — raw head; the Backbone runs its own NMS + mask decode:

```yaml
export_formats: ["onnx", "openvino"]
export_nms: false
export_opset: 17
export_dynamic: true     # runtime-selectable inference size
```

**Results** — `runs/segment/models/yolo/<model>_e<epochs>_<imgsz>px_<ts>/` (weights, exports, auto `report.md`). Detect runs: `runs/detect/models/yolo/`. RF-DETR: `trainer/isidet/models/rfdetr/<ts>/`.

Deploy: drop `.onnx` into `models/`, set `detection.onnx_path`, check with `tools/onnx_inspect.py` + `tools/detection_smoke.py --onnx <model> --image <jpg>`.

## isiGen — synthetic data (`trainer/isiGen/`, Studio :8200)

**WHY** — 50–100 real photos per class isn't a dataset. **WHAT** — SDXL + depth-ControlNet → unlimited labeled synthetic images. **HOW** — 8-phase pipeline (scaffolds → captioning → LoRA → generation → masks → QA) → YOLO-seg export feeding isidet.

!!! warning "WSL2"
    Heavy models swap-thrash the 12 GB VM → EIO/bus-error. Levers: smaller model, `batch_size`, `workers`.
