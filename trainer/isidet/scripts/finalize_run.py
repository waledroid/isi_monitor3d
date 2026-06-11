"""One-off finalize pass for an interrupted training run.

Reproduces everything the end-of-training pipeline (run_train.py steps 10-11)
would have generated — WITHOUT any further training — from the existing
`best.pt`:

  1. validation plots into the run dir (confusion matrix, PR/F1 curves, val batches)
  2. results.png — full-history training curves from results.csv
  3. ONNX export (raw head, opset 17, nms off) — Backbone yolo_onnx_seg compatible
  4. OpenVINO IR (via the trainer's export_engine, with a native fallback)
  5. report.md

Run from trainer/isidet with the isi-train env.
"""
import contextlib
import os
import sys
from pathlib import Path

ROOT = Path("/home/aatanda/isi_monitor3d/trainer/isidet")
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

RUN = ROOT / "runs/segment/models/yolo/yolo26l-seg_e200_640px_09-06-2026_00-24-57"
BEST = RUN / "weights/best.pt"
DATA = ROOT / "data/pallet3_yolo_seg/data.yaml"
IMGSZ = 640
OPSET = 17
NMS = False

from ultralytics import YOLO  # noqa: E402


def step(name):
    print("\n" + "=" * 60 + f"\n  {name}\n" + "=" * 60, flush=True)


# 1 + 3/4 share the loaded model
model = YOLO(str(BEST))

# ---- 1. validation plots into the run dir ----
step("1/5  VALIDATION (plots → run dir)")
try:
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        res = model.val(
            data=str(DATA), imgsz=IMGSZ, batch=4, workers=2, plots=True,
            project=str(RUN.parent), name=RUN.name, exist_ok=True, verbose=False,
        )
    print(f"  box  mAP50={res.box.map50:.4f}  mAP50-95={res.box.map:.4f}", flush=True)
    print(f"  mask mAP50={res.seg.map50:.4f}  mAP50-95={res.seg.map:.4f}", flush=True)
except Exception as e:
    print(f"  ⚠️ validation failed: {e}", flush=True)

# ---- 2. training-history curves ----
step("2/5  results.png (training curves)")
try:
    from ultralytics.utils.plotting import plot_results
    try:
        plot_results(file=str(RUN / "results.csv"), segment=True)
    except TypeError:
        plot_results(str(RUN / "results.csv"))
    print(f"  wrote {RUN/'results.png'}", flush=True)
except Exception as e:
    print(f"  ⚠️ plot_results failed: {e}", flush=True)

# ---- 3. ONNX export (raw head) ----
step("3/5  ONNX export (opset 17, nms off, raw head)")
onnx_path = None
try:
    onnx_path = model.export(
        format="onnx", imgsz=IMGSZ, opset=OPSET, nms=NMS, simplify=True, dynamic=False,
    )
    print(f"  ✅ {onnx_path}", flush=True)
except Exception as e:
    print(f"  ⚠️ ONNX export failed: {e}", flush=True)

# ---- 4. OpenVINO IR ----
step("4/5  OpenVINO export")
try:
    from src.inference.export_engine import run_pipeline
    run_pipeline(model_dir=BEST.parent, formats={"openvino"}, imgsz=IMGSZ)
    print("  ✅ OpenVINO via export_engine", flush=True)
except Exception as e:
    print(f"  ⚠️ export_engine openvino failed ({e}); trying native ultralytics export", flush=True)
    try:
        ov = model.export(format="openvino", imgsz=IMGSZ)
        print(f"  ✅ {ov}", flush=True)
    except Exception as e2:
        print(f"  ⚠️ native openvino export also failed: {e2}", flush=True)

# ---- 5. report.md ----
step("5/5  report.md")
try:
    from make_train_report import generate_report
    generate_report(RUN)
    print(f"  ✅ {RUN/'report.md'}", flush=True)
except Exception as e:
    print(f"  ⚠️ report generation failed: {e}", flush=True)

print("\n=== FINALIZE COMPLETE ===", flush=True)
print("run dir:", RUN, flush=True)
