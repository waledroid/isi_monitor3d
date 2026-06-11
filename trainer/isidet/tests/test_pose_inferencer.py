"""Smoke test for the YOLO-pose ONNX path: export → infer → foot node.

Proves the whole pose pipeline works end-to-end using the STOCK pretrained
person-pose model (no custom dataset needed — in the IsiMonitor3D design person
is handled by the pretrained COCO pose model). It exports ``yolo11n-pose.pt`` to
the raw-head ONNX the backbone consumes, runs ``PoseOnnxInferencer`` on a bundled
person image, and asserts we get person detections with 17 keypoints, visible
ankles in-frame, and a sane foot node.

Runs in the ``isi-train`` env. Works two ways:
    /home/aatanda/miniforge3/envs/isi-train/bin/python tests/test_pose_inferencer.py
    (or, if pytest is installed)  python -m pytest tests/test_pose_inferencer.py
Run from ``trainer/isidet/``. First run downloads yolo11n-pose.pt (~6 MB).
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

# Make ``import src...`` work regardless of cwd (trainer/isidet on the path).
_ISIDET = Path(__file__).resolve().parents[1]
if str(_ISIDET) not in sys.path:
    sys.path.insert(0, str(_ISIDET))

from src.inference.pose_onnx_inferencer import (  # noqa: E402
    LEFT_ANKLE,
    RIGHT_ANKLE,
    PoseOnnxInferencer,
)

_WEIGHTS = "yolo11n-pose.pt"
_EXPORT_DIR = _ISIDET / "tests" / ".cache"


def _export_pose_onnx() -> Path:
    """Export the stock pose model to ONNX once (cached). opset/nms/dynamic match
    the backbone's raw-head contract."""
    onnx_path = _EXPORT_DIR / "yolo11n-pose.onnx"
    if onnx_path.exists():
        return onnx_path
    _EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO

    model = YOLO(_WEIGHTS)   # auto-downloads if missing; task=pose from suffix
    out = model.export(format="onnx", imgsz=640, opset=17, nms=False,
                       simplify=True, dynamic=False)
    src = Path(out)
    if src.resolve() != onnx_path.resolve():
        onnx_path.write_bytes(src.read_bytes())
    return onnx_path


def _person_image() -> np.ndarray:
    # bus.jpg has full-body people (feet visible) — needed to exercise the ankle
    # foot node. (zidane.jpg is a waist-up crop, so its ankles are out of frame.)
    import ultralytics

    img_path = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"
    frame = cv2.imread(str(img_path))
    assert frame is not None, f"could not read test image {img_path}"
    return frame


def test_pose_onnx_detects_person_with_foot_nodes() -> None:
    onnx_path = _export_pose_onnx()
    frame = _person_image()
    h, w = frame.shape[:2]

    eng = PoseOnnxInferencer(str(onnx_path), conf_threshold=0.25, device="cpu")
    # Head layout: single person class, 17 COCO keypoints.
    assert eng.num_keypoints == 17, f"expected 17 keypoints, got {eng.num_keypoints}"

    poses = eng.predict(frame)
    assert len(poses) >= 1, "expected at least one person detection on zidane.jpg"

    for p in poses:
        assert p.keypoints.shape == (17, 3), p.keypoints.shape
        assert p.score >= 0.25
        # Box within image bounds.
        x1, y1, x2, y2 = p.box_xyxy
        assert 0 <= x1 < x2 <= w + 1 and 0 <= y1 < y2 <= h + 1
        # Foot node finite + inside the frame.
        fx, fy = p.foot_uv
        assert np.isfinite(fx) and np.isfinite(fy)
        assert 0 <= fx <= w and 0 <= fy <= h

    # At least one person must have a visible ankle (the foot node source).
    any_ankle = any(
        p.keypoints[LEFT_ANKLE, 2] >= 0.3 or p.keypoints[RIGHT_ANKLE, 2] >= 0.3
        for p in poses
    )
    assert any_ankle, "no visible ankle keypoint found — foot node would fall back to bbox"


def _main() -> int:
    """Standalone runner (no pytest needed) — prints a readable report + saves an
    annotated image so you can eyeball that ankles land at the feet."""
    onnx_path = _export_pose_onnx()
    frame = _person_image()
    eng = PoseOnnxInferencer(str(onnx_path), conf_threshold=0.25, device="cpu")
    poses = eng.predict(frame)
    print(f"\nONNX: {onnx_path}")
    print(f"input={eng.model_w}x{eng.model_h}  nc={eng.nc}  K={eng.num_keypoints}")
    print(f"persons detected: {len(poses)}")
    for i, p in enumerate(poses):
        la, ra = p.keypoints[LEFT_ANKLE], p.keypoints[RIGHT_ANKLE]
        print(f"  person {i}: score={p.score:.2f}  foot_uv=({p.foot_uv[0]:.0f},{p.foot_uv[1]:.0f})  "
              f"L-ankle conf={la[2]:.2f}  R-ankle conf={ra[2]:.2f}")
    out_path = _EXPORT_DIR / "pose_smoke_annotated.jpg"
    cv2.imwrite(str(out_path), eng.draw(frame, poses))
    print(f"annotated: {out_path}")

    try:
        test_pose_onnx_detects_person_with_foot_nodes()
    except AssertionError as exc:
        print(f"\nFAIL: {exc}")
        return 1
    print("\nPASS: pose ONNX detects persons with foot nodes (ankles).")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
