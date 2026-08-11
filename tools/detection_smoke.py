#!/usr/bin/env python
"""Single-image OpenVINO IR detection smoke test (CPU branch).

Runs the registered ``yolo_openvino``/``yolo_openvino_seg`` plugin (picked by
IR output arity) on one image and prints the detections + per-stage timing:

    python tools/detection_smoke.py --xml models/pallet_seg_openvino/model.xml \\
        --image shot.jpg --input-size 320 [--annotate out.jpg]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xml", required=True, help="OpenVINO IR model.xml (bin beside it)")
    ap.add_argument("--image", required=True, help="input image")
    ap.add_argument("--input-size", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--classes", nargs="+", default=["palette", "carton", "polybag"])
    ap.add_argument("--annotate", help="write annotated copy here")
    args = ap.parse_args()

    import openvino as ov

    import backbone.detection  # noqa: F401 — registers the openvino plugins
    from backbone.core.interfaces import detector_registry
    from backbone.core.types import Frame, FramePair

    n_outputs = len(ov.Core().read_model(args.xml).outputs)
    plugin = "yolo_openvino_seg" if n_outputs == 2 else "yolo_openvino"
    print(f"IR outputs: {n_outputs} → plugin {plugin}")

    t0 = time.perf_counter()
    det = detector_registry.create(
        plugin, model_xml=args.xml, class_names=args.classes,
        confidence_threshold=args.conf,
        input_size=(args.input_size, args.input_size), device="CPU")
    t_load = time.perf_counter() - t0

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"cannot read {args.image}")
    pair = FramePair(capture_ts=0.0, frame_idx=0, frames={
        "cam": Frame(camera_id="cam", capture_ts=0.0, frame_idx=0, image=img)})

    det.detect(pair)                                   # warmup
    t1 = time.perf_counter()
    out = det.detect(pair)["cam"]
    t_inf = time.perf_counter() - t1

    print(f"load {t_load * 1000:.0f} ms | infer {t_inf * 1000:.1f} ms | {len(out)} detection(s)")
    for d in out:
        x0, y0, x1, y1 = (round(v) for v in d.bbox_xyxy)
        print(f"  {d.cls:10s} conf={d.confidence:.2f} bbox=({x0},{y0},{x1},{y1})"
              f" mask={'yes' if d.mask is not None else 'no'}")

    if args.annotate:
        for d in out:
            x0, y0, x1, y1 = (int(v) for v in d.bbox_xyxy)
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 220, 0), 2)
            cv2.putText(img, f"{d.cls} {d.confidence:.2f}", (x0, max(12, y0 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1, cv2.LINE_AA)
        cv2.imwrite(args.annotate, img)
        print(f"annotated → {args.annotate}")


if __name__ == "__main__":
    main()
