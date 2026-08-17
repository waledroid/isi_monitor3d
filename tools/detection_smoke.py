"""Manual smoke test for the YOLO / RF-DETR ONNX detectors against a real ONNX.

The plugin is auto-selected from the model's ONNX output names — same rule as the
dashboard overlay: 3 outputs named ``dets``/``labels``/``masks`` ⇒ RF-DETR
(``rfdetr_onnx_seg``); 2 outputs ⇒ ``yolo_onnx_seg``; 1 output ⇒ ``yolo_onnx``.
The detector is always built via ``detector_registry.create(...)``.

Usage:
    python tools/detection_smoke.py --onnx yolo11n.onnx \\
        --mp4 fixtures/cam_a.mp4 --timestamps fixtures/cam_a.timestamps.json \\
        --class-names person,bicycle,car,...,toothbrush  \\
        --keep person --frames 30

Or with a single image file:
    python tools/detection_smoke.py --onnx yolo11n.onnx --image fixtures/test.jpg \\
        --class-names person --keep person

RF-DETR (its own fixed input size; NMS-free; --iou/--keep are ignored for it):
    python tools/detection_smoke.py \\
        --onnx trainer/isidet/models/rfdetr/07-06-2026_0909/inference_model.sim.onnx \\
        --image some.jpg --class-names palette,carton,polybag --annotate out.jpg

Prints per-frame: detection count, per-stage latency (preprocess / inference /
postprocess / total), and the top-3 detections by confidence.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

import backbone.detection  # noqa: F401 — registers yolo_onnx{,_seg}, yolo_openvino{,_seg}, rfdetr_onnx_seg
from backbone.core.interfaces import detector_registry
from backbone.core.types import Frame, FramePair
from backbone.ingestion.replay import ReplayFrameSource

# RF-DETR ONNX exports name their outputs dets / labels / masks. dets+labels are
# mandatory (matching RfdetrOnnxSegDetector's own check); masks is the seg head.
_RFDETR_REQUIRED_NAMES = frozenset({"dets", "labels"})
_RFDETR_DEFAULT_CLASS_NAMES = ("palette", "carton", "polybag")


def _onnx_output_names(onnx_path: Path) -> list[str]:
    """The model's output tensor names (used to auto-select the plugin).
    Native ``.engine`` files answer from their conversion sidecar."""
    if str(onnx_path).endswith(".engine"):
        from backbone.shared.trt_session import read_sidecar

        return list((read_sidecar(onnx_path) or {}).get("outputs") or [])
    import onnx

    return [o.name for o in onnx.load(str(onnx_path)).graph.output]


def _select_plugin(output_names: list[str]) -> str:
    """yolo_onnx / yolo_onnx_seg / rfdetr_onnx_seg from ONNX output names. Same
    rule as the dashboard overlay's ``select_plugin``."""
    name_set = {str(n) for n in output_names}
    if _RFDETR_REQUIRED_NAMES.issubset(name_set):
        return "rfdetr_onnx_seg"
    if len(output_names) == 2:
        return "yolo_onnx_seg"
    return "yolo_onnx"


COCO_DEFAULTS = (
    "person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,"
    "fire hydrant,stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,"
    "elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,"
    "skis,snowboard,sports ball,kite,baseball bat,baseball glove,skateboard,"
    "surfboard,tennis racket,bottle,wine glass,cup,fork,knife,spoon,bowl,banana,"
    "apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,"
    "potted plant,bed,dining table,toilet,tv,laptop,mouse,remote,keyboard,"
    "cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,"
    "scissors,teddy bear,hair drier,toothbrush"
)


def _build_detector(args: argparse.Namespace):
    """Build the detector via the registry, auto-selecting the plugin from the
    ONNX output names. RF-DETR ignores the YOLO-only iou/keep kwargs."""
    providers = None if args.providers == "auto" else args.providers.split(",")
    plugin = _select_plugin(_onnx_output_names(Path(args.onnx)))

    if plugin == "rfdetr_onnx_seg":
        # RF-DETR: NMS-free, fixed input — pass only its kwargs. Default class
        # names to the trained palette/carton/polybag triplet unless overridden.
        class_names = (
            [c.strip() for c in args.class_names.split(",")]
            if args.class_names_explicit
            else list(_RFDETR_DEFAULT_CLASS_NAMES)
        )
        det = detector_registry.create(
            plugin,
            onnx_path=str(args.onnx),
            class_names=class_names,
            confidence_threshold=args.conf,
            providers=providers,
        )
        return plugin, det

    class_names = [c.strip() for c in args.class_names.split(",")]
    keep = [c.strip() for c in args.keep.split(",")] if args.keep else None
    det = detector_registry.create(
        plugin,
        onnx_path=str(args.onnx),
        class_names=class_names,
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
        keep_classes=keep,
        input_size=(args.input_size, args.input_size),
        providers=providers,
    )
    return plugin, det


def _write_annotated(image, dets: list, out_path: Path) -> None:
    """Draw boxes + class/conf labels (and seg-mask underlay if present) and save."""
    canvas = image.copy()
    for d in dets:
        if getattr(d, "mask", None) is not None and d.mask.shape == canvas.shape[:2]:
            overlay = canvas.copy()
            overlay[d.mask] = (80, 220, 80)
            cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, dst=canvas)
        x1, y1, x2, y2 = (int(v) for v in d.bbox_xyxy)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (80, 220, 80), 2)
        cv2.putText(canvas, f"{d.cls} {d.confidence:.2f}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 220, 80), 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def _report(idx: int, dets: list, timings_ms: dict[str, float]) -> None:
    total = sum(timings_ms.values())
    print(
        f"frame {idx:>4d}  count={len(dets):>2d}  "
        f"pre={timings_ms['preprocess']:.1f}  inf={timings_ms['inference']:.1f}  "
        f"post={timings_ms['postprocess']:.1f}  total={total:.1f} ms"
    )
    for d in dets[:3]:
        x1, y1, x2, y2 = d.bbox_xyxy
        print(f"    {d.cls:<12s} conf={d.confidence:.2f}  "
              f"box=({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})  foot={d.foot_uv}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument(
        "--class-names",
        default=COCO_DEFAULTS,
        help="comma-separated names matching the model's output channels "
        "(RF-DETR defaults to palette,carton,polybag if omitted)",
    )
    parser.add_argument("--keep", default=None, help="comma-separated names to emit (YOLO only)")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--input-size", type=int, default=640,
                        help="letterbox size for dynamic-input YOLO exports (e.g. 320 for a "
                             "320-trained model); fixed-input exports override it (YOLO only)")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU (YOLO only)")
    parser.add_argument(
        "--annotate",
        type=Path,
        default=None,
        help="write an annotated JPG of the --image result (boxes + class + conf)",
    )
    parser.add_argument(
        "--providers",
        default="auto",
        help="comma-separated ORT providers; 'auto' = CUDA→CPU fallback",
    )
    parser.add_argument("--camera-id", default="smoke")
    parser.add_argument("--frames", type=int, default=10)

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", type=Path, help="single image file")
    src.add_argument("--mp4", type=Path, help="recorded MP4")
    parser.add_argument(
        "--timestamps",
        type=Path,
        default=None,
        help="sidecar JSON of capture_ts (required with --mp4)",
    )

    args = parser.parse_args(argv)
    # Did the user pass --class-names explicitly? (so RF-DETR can default to its
    # own triplet otherwise). argparse can't tell, so compare against the default.
    args.class_names_explicit = args.class_names != COCO_DEFAULTS
    plugin, det = _build_detector(args)
    print(f"[smoke] plugin: {plugin}  (auto-selected from ONNX outputs)")
    print(f"[smoke] providers active: {det.active_providers}")
    det.warmup()

    if args.image is not None:
        img = cv2.imread(str(args.image))
        if img is None:
            raise RuntimeError(f"could not read image: {args.image}")
        f = Frame(camera_id=args.camera_id, capture_ts=time.time(), frame_idx=0, image=img)
        pair = FramePair(capture_ts=f.capture_ts, frame_idx=0, frames={args.camera_id: f})
        t0 = time.perf_counter()
        result = det.detect(pair)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        dets = result.get(args.camera_id, [])
        print(f"[smoke] image: {args.image}  detections: {len(dets)}  "
              f"detect(): {elapsed_ms:.1f} ms")
        for d in sorted(dets, key=lambda d: d.confidence, reverse=True):
            x1, y1, x2, y2 = d.bbox_xyxy
            has_mask = getattr(d, "mask", None) is not None
            print(f"  {d.cls:<12s} conf={d.confidence:.2f}  "
                  f"box=({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})  "
                  f"foot={d.foot_uv}  mask={'yes' if has_mask else 'no'}")
        if args.annotate is not None:
            _write_annotated(img, dets, args.annotate)
            print(f"[smoke] annotated image written to {args.annotate}")
        return 0

    # MP4 + sidecar
    if args.timestamps is None:
        parser.error("--mp4 requires --timestamps")
    rep = ReplayFrameSource(
        camera_id=args.camera_id,
        mp4_path=args.mp4,
        timestamps_json=args.timestamps,
    )
    for i, frame in enumerate(rep.frames()):
        if i >= args.frames:
            break
        pair = FramePair(capture_ts=frame.capture_ts, frame_idx=i, frames={args.camera_id: frame})
        t0 = time.perf_counter()
        result = det.detect(pair)
        t1 = time.perf_counter()
        dets = result.get(args.camera_id, [])
        timings = {"preprocess": 0.0, "inference": (t1 - t0) * 1000.0, "postprocess": 0.0}
        # Stage-level timings would require instrumenting yolo_onnx.detect.
        # For S3 v1, the aggregate detect() latency is what matters.
        _report(i, dets, timings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
