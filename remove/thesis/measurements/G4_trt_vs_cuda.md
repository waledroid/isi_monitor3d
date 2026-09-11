# G4 — TensorRT EP vs CUDA EP benchmark

**Date:** 2026-07-20 · **Env:** `monitor3d` (onnxruntime-gpu 1.23.2, TensorRT
10.16, CUDA 12.9, RTX 5070 12 GB, WSL2) · **Model:** the PRODUCTION detector from
`config/backbone.yaml` (read-only): `trainer/isidet/runs/segment/models/yolo/`
`yolo26n-seg_e100_320px_03-07-2026_15-09-28/weights/best.fp16.onnx`
(`yolo_onnx_seg` plugin, classes palette/carton/polybag, conf 0.15, iou 0.45,
decode_masks=true as in the config). Production `detection.zone_imgsz: 320`;
also measured at 640.

## Method

`tools/detection_smoke.py` reports only a single total `detect()` time (its
per-stage fields are zero-filled — line 228–229), so a small harness
(`thesis/measurements/raw/g4_bench.py`) builds the detector through the same
registry (`detector_registry.create("yolo_onnx_seg", ...)`) and the same
`ISI3D_TRT` EP switch (`backbone/shared/ort_session.py`), then times:

- **detect() end-to-end** — preprocess + inference + NMS + mask decode (what the
  pipeline pays per frame), N=50 after 5 warm-up calls;
- **isolated `session.run`** — pure EP inference on a preprocessed tensor, N=50
  after 5 warm-ups;
- **VRAM** — `nvidia-smi` total-used sampled at 4 Hz (baseline before build,
  peak during).

Input frame: mid-frame (idx 3090) of `trainer/isidet/data/testvid.mp4`
(1080×1920), 2 detections at every config (consistent across EPs/sizes).
One (EP × imgsz) config per process, run **strictly sequentially**:

```bash
conda activate monitor3d && cd /home/aatanda/isi_monitor3d
ISI3D_TRT=0 python g4_bench.py 320 g4_frame.jpg g4_cuda_320.json
ISI3D_TRT=0 python g4_bench.py 640 g4_frame.jpg g4_cuda_640.json
ISI3D_TRT=1 python g4_bench.py 320 g4_frame.jpg g4_trt_320.json
ISI3D_TRT=1 python g4_bench.py 640 g4_frame.jpg g4_trt_640.json
```

## Results (N=50 each; raw JSON in `thesis/measurements/raw/g4_*.json`)

### Isolated inference (session.run)

| EP | imgsz | mean ms | median ms | p95 ms | TRT speedup (median) |
|---|---|---|---|---|---|
| CUDA | 320 | 13.78 | 13.85 | 19.15 | — |
| **TRT** | 320 | 5.40 | 4.55 | 9.16 | **3.0×** (mean 2.6×) |
| CUDA | 640 | 23.02 | 16.39 | 62.62 | — |
| **TRT** | 640 | 6.63 | 5.83 | 10.18 | **2.8×** (mean 3.5×) |

### End-to-end detect() (preprocess + inference + NMS + full mask decode)

| EP | imgsz | mean ms | median ms | p95 ms | TRT speedup (median) |
|---|---|---|---|---|---|
| CUDA | 320 | 24.47 | 22.35 | 38.14 | — |
| **TRT** | 320 | 10.69 | 8.74 | 23.41 | **2.6×** |
| CUDA | 640 | 67.63 | 64.35 | 94.73 | — |
| **TRT** | 640 | 48.97 | 46.68 | 64.19 | **1.4×** |

### VRAM (nvidia-smi total used, MB)

| EP | imgsz | baseline before | peak during | delta (bench footprint) |
|---|---|---|---|---|
| CUDA | 320 | 4,971 | 5,178 | 207 |
| CUDA | 640 | 4,970 | 5,402 | 432 |
| TRT | 320 | 5,131 | 5,440 | 309 |
| TRT | 640 | 5,131 | 5,452 | 321 |

## Notes / caveats

- **TRT engine cache hit** — no rebuild occurred: detector build took 1.9 s
  (TRT 320) and 1.0 s (TRT 640) against the pre-existing engines in
  `models/.trt_cache` (a cold first build per shape takes minutes; not observed
  here, so these numbers are steady-state production behavior).
- The isolated-inference speedups (2.6–3.5×) bracket the previously documented
  "2.1–2.3× over CUDA EP" claim; the end-to-end gain at 640 collapses to 1.4×
  because CPU-side postprocess/mask decode dominates (≈41 ms of the 46.7 ms
  TRT total).
- **Baseline contamination:** the operator dashboard (`monitor_web`, untouched
  per campaign rules) held ≈4.97–5.13 GB VRAM and was actively streaming during
  all runs; occasional p95 spikes (e.g. CUDA-640 infer p95 62.6 ms vs median
  16.4 ms) plausibly reflect GPU contention with it. Medians are the robust
  statistic here. Absolute VRAM columns are machine totals, deltas are the
  bench's own footprint.
- detect() at imgsz 640 exceeds the isolated-inference cost by ~40 ms mostly in
  Python-side letterboxing, fp16 tensor conversion, NMS, and per-detection
  full-frame mask assembly (`decode_masks=True`, matching the deployed config).
