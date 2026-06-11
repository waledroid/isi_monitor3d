"""``YoloOnnxDetector`` — registration + constructor validation + end-to-end
inference on a tiny hand-built ONNX model (no real weights required).

The integration test builds an ONNX graph that mimics a YOLO11 detect head's
output shape `(N, 4+nc, A)` so we can verify the whole letterbox → ORT → decode
pipeline without depending on a downloaded model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from backbone.core.interfaces import detector_registry
from backbone.core.types import Frame, FramePair
from backbone.detection.yolo_onnx import YoloOnnxDetector

CLASS_NAMES = ["person", "forklift", "pallet"]
NC = len(CLASS_NAMES)
NUM_ANCHORS = 64  # tiny — keeps the test fast


def _build_constant_yolo_onnx(out_tensor: np.ndarray, path: Path) -> None:
    """Write a minimal ONNX model that ignores its input and returns a constant.

    The model declares the input shape YOLO11-detect expects (N, 3, 640, 640)
    and the output shape (N, 4+nc, A). We use a Constant node to return the
    same tensor regardless of input — sufficient to exercise the pre/post path.
    """
    input_tv = helper.make_tensor_value_info(
        "images", TensorProto.FLOAT, ["N", 3, 640, 640]
    )
    output_tv = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [out_tensor.shape[0], out_tensor.shape[1], out_tensor.shape[2]]
    )
    const_tensor = numpy_helper.from_array(out_tensor.astype(np.float32), name="const_out")
    const_node = helper.make_node("Constant", inputs=[], outputs=["output"], value=const_tensor)
    graph = helper.make_graph(
        nodes=[const_node],
        name="constant_yolo_stub",
        inputs=[input_tv],
        outputs=[output_tv],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=10,
        producer_name="isi-monitor3d-tests",
    )
    onnx.save(model, str(path))


def _make_pair_with_one_image() -> FramePair:
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    f = Frame(camera_id="cam_a", capture_ts=10.0, frame_idx=0, image=img)
    return FramePair(capture_ts=10.0, frame_idx=0, frames={"cam_a": f})


# ---------- registry + constructor ----------


def test_plugin_registered_under_yolo_onnx() -> None:
    import backbone.detection  # noqa: F401  — trigger registration

    assert "yolo_onnx" in detector_registry


def test_missing_onnx_raises_filenotfound(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="ONNX file not found"):
        YoloOnnxDetector(
            onnx_path=tmp_path / "missing.onnx",
            class_names=CLASS_NAMES,
        )


def test_no_class_names_anywhere_rejected(tmp_path: Path) -> None:
    """The detector prefers names embedded in the ONNX, falling back to the
    configured ones. If the model embeds none AND none are configured, it must
    fail clearly (the stub here carries no embedded names)."""
    out = np.zeros((1, 4 + NC, NUM_ANCHORS), dtype=np.float32)
    model_path = tmp_path / "stub.onnx"
    _build_constant_yolo_onnx(out, model_path)
    with pytest.raises(ValueError, match="no class names"):
        YoloOnnxDetector(onnx_path=model_path, class_names=[])


def test_unknown_keep_classes_dropped_not_fatal(tmp_path: Path) -> None:
    """Stale/unknown keep_classes are ignored (dropped), not fatal — so config
    drift can't break a detector when the model's classes change. When nothing
    valid remains, the filter is disabled (keep all)."""
    out = np.zeros((1, 4 + NC, NUM_ANCHORS), dtype=np.float32)
    model_path = tmp_path / "stub.onnx"
    _build_constant_yolo_onnx(out, model_path)
    det = YoloOnnxDetector(
        onnx_path=model_path,
        class_names=CLASS_NAMES,
        keep_classes=["ghost"],
    )
    assert det._keep_classes is None   # 'ghost' dropped → no filter


# ---------- end-to-end inference ----------


def test_detect_returns_detections_for_strong_anchor(tmp_path: Path) -> None:
    """Build an ONNX that always emits one strong 'person' anchor; verify
    the detector decodes it into a single Detection in source-frame coords."""
    out = np.zeros((1, 4 + NC, NUM_ANCHORS), dtype=np.float32)
    # Anchor 0: centre (320, 320) in 640x640, size 100x200, class=person, conf=0.9
    out[0, 0, 0] = 320.0  # cx
    out[0, 1, 0] = 320.0  # cy
    out[0, 2, 0] = 100.0  # w
    out[0, 3, 0] = 200.0  # h
    out[0, 4, 0] = 0.9    # person score

    model_path = tmp_path / "stub.onnx"
    _build_constant_yolo_onnx(out, model_path)

    det = YoloOnnxDetector(
        onnx_path=model_path,
        class_names=CLASS_NAMES,
        providers=["CPUExecutionProvider"],  # hermetic — no GPU dep
    )
    result = det.detect(_make_pair_with_one_image())
    assert set(result) == {"cam_a"}
    assert len(result["cam_a"]) == 1
    d = result["cam_a"][0]
    assert d.cls == "person"
    assert d.confidence == pytest.approx(0.9, abs=1e-5)
    # bbox should be mapped back to source-frame (1080x1920). The exact pixels
    # depend on letterbox scale; just verify it's plausible (centre near image
    # centre, bottom-foot inside the frame).
    x1, y1, x2, y2 = d.bbox_xyxy
    assert x1 < x2 and y1 < y2
    assert 0 <= x1 and x2 <= 1919
    assert 0 <= y1 and y2 <= 1079
    foot_u, foot_v = d.foot_uv
    assert foot_u == pytest.approx((x1 + x2) / 2.0)
    assert foot_v == y2


def test_detect_empty_output_yields_no_detections(tmp_path: Path) -> None:
    out = np.zeros((1, 4 + NC, NUM_ANCHORS), dtype=np.float32)  # zero confidence everywhere
    model_path = tmp_path / "stub.onnx"
    _build_constant_yolo_onnx(out, model_path)
    det = YoloOnnxDetector(
        onnx_path=model_path,
        class_names=CLASS_NAMES,
        providers=["CPUExecutionProvider"],
    )
    result = det.detect(_make_pair_with_one_image())
    assert result == {"cam_a": []}


def test_active_providers_exposed(tmp_path: Path) -> None:
    out = np.zeros((1, 4 + NC, NUM_ANCHORS), dtype=np.float32)
    model_path = tmp_path / "stub.onnx"
    _build_constant_yolo_onnx(out, model_path)
    det = YoloOnnxDetector(
        onnx_path=model_path,
        class_names=CLASS_NAMES,
        providers=["CPUExecutionProvider"],
    )
    assert "CPUExecutionProvider" in det.active_providers


def test_warmup_does_not_raise(tmp_path: Path) -> None:
    out = np.zeros((1, 4 + NC, NUM_ANCHORS), dtype=np.float32)
    model_path = tmp_path / "stub.onnx"
    _build_constant_yolo_onnx(out, model_path)
    det = YoloOnnxDetector(
        onnx_path=model_path,
        class_names=CLASS_NAMES,
        providers=["CPUExecutionProvider"],
    )
    det.warmup()


