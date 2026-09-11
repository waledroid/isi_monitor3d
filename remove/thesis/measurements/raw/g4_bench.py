"""G4 bench: one (EP, imgsz) config per invocation — strictly sequential GPU use.

Usage: ISI3D_TRT={0|1} python g4_bench.py <imgsz> <frame.jpg> <out.json>

Measures, for the production yolo26n-seg best.fp16.onnx:
  - detector build time (session init; TRT engine build/cache hit shows up here)
  - N=50 timed det.detect(pair) calls (end-to-end: preprocess+inference+postprocess)
  - N=50 timed raw session.run on the preprocessed tensor (inference only)
  - VRAM (nvidia-smi total-used) sampled at 4 Hz during the run: baseline/peak
"""
import json, os, statistics, subprocess, sys, threading, time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "/home/aatanda/isi_monitor3d")
import backbone.detection  # noqa: F401
from backbone.core.interfaces import detector_registry
from backbone.core.types import Frame, FramePair

IMGSZ = int(sys.argv[1]); FRAME = sys.argv[2]; OUT = sys.argv[3]
ONNX = ("/home/aatanda/isi_monitor3d/trainer/isidet/runs/segment/models/yolo/"
        "yolo26n-seg_e100_320px_03-07-2026_15-09-28/weights/best.fp16.onnx")
N = 50; WARMUP = 5

vram, stop = [], threading.Event()
def sampler():
    while not stop.is_set():
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True)
        try: vram.append(int(r.stdout.strip()))
        except ValueError: pass
        time.sleep(0.25)

baseline = None
r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                   capture_output=True, text=True)
baseline = int(r.stdout.strip())
t = threading.Thread(target=sampler, daemon=True); t.start()

img = cv2.imread(FRAME); assert img is not None, FRAME

t0 = time.perf_counter()
det = detector_registry.create(
    "yolo_onnx_seg", onnx_path=ONNX, class_names=["palette", "carton", "polybag"],
    confidence_threshold=0.15, iou_threshold=0.45, input_size=(IMGSZ, IMGSZ),
    decode_masks=True)
build_s = time.perf_counter() - t0
det.warmup()

f = Frame(camera_id="bench", capture_ts=time.time(), frame_idx=0, image=img)
pair = FramePair(capture_ts=f.capture_ts, frame_idx=0, frames={"bench": f})

for _ in range(WARMUP):
    det.detect(pair)

detect_ms = []
for _ in range(N):
    a = time.perf_counter(); res = det.detect(pair); detect_ms.append((time.perf_counter() - a) * 1e3)
n_dets = len(res["bench"])

# isolated inference: replicate the plugin's preprocess once, then time session.run
sess = det._session
inp = sess.get_inputs()[0]
# letterbox to IMGSZ like the plugin (approx: direct resize is fine for timing)
blob = cv2.resize(img, (IMGSZ, IMGSZ)).astype(np.float32)[:, :, ::-1] / 255.0
blob = np.ascontiguousarray(blob.transpose(2, 0, 1)[None])
if "float16" in inp.type: blob = blob.astype(np.float16)
for _ in range(WARMUP):
    sess.run(None, {inp.name: blob})
infer_ms = []
for _ in range(N):
    a = time.perf_counter(); sess.run(None, {inp.name: blob}); infer_ms.append((time.perf_counter() - a) * 1e3)

stop.set(); t.join(timeout=2)
out = {
    "trt_env": os.environ.get("ISI3D_TRT"), "imgsz": IMGSZ, "n": N,
    "active_providers": det.active_providers, "build_s": round(build_s, 2),
    "n_detections": n_dets,
    "detect_ms": {"mean": round(statistics.mean(detect_ms), 2),
                  "median": round(statistics.median(detect_ms), 2),
                  "p95": round(sorted(detect_ms)[int(N * 0.95) - 1], 2)},
    "infer_ms": {"mean": round(statistics.mean(infer_ms), 2),
                 "median": round(statistics.median(infer_ms), 2),
                 "p95": round(sorted(infer_ms)[int(N * 0.95) - 1], 2)},
    "vram_mb": {"baseline_before": baseline, "peak_during": max(vram) if vram else None,
                "delta_peak": (max(vram) - baseline) if vram else None},
}
Path(OUT).write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
