"""``YoloOpenvinoDetector`` — registration + constructor validation + (when
OpenVINO is installed) end-to-end inference on a tiny IR converted from a
hand-built constant ONNX.

The validation guards (missing file / empty class_names / keep_classes subset)
run *before* the lazy ``import openvino`` in the constructor, so those tests pass
even without OpenVINO installed. The inference test ``importorskip``s openvino.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backbone.core.interfaces import detector_registry
from backbone.core.types import Frame, FramePair
from backbone.detection.yolo_openvino import YoloOpenvinoDetector

CLASS_NAMES = ["person", "forklift", "pallet"]
NC = len(CLASS_NAMES)
NUM_ANCHORS = 64


def _make_pair_with_one_image() -> FramePair:
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    f = Frame(camera_id="cam_a", capture_ts=10.0, frame_idx=0, image=img)
    return FramePair(capture_ts=10.0, frame_idx=0, frames={"cam_a": f})


def _build_constant_yolo_onnx(out_tensor: np.ndarray, path: Path) -> None:
    """Minimal ONNX returning a constant (N, 4+nc, A) — same trick as test_yolo_onnx."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    input_tv = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])
    output_tv = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, list(out_tensor.shape)
    )
    const_tensor = numpy_helper.from_array(out_tensor.astype(np.float32), name="const_out")
    const_node = helper.make_node("Constant", inputs=[], outputs=["output"], value=const_tensor)
    graph = helper.make_graph([const_node], "constant_yolo_stub", [input_tv], [output_tv])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10,
        producer_name="isi-monitor3d-tests",
    )
    onnx.save(model, str(path))


# ---------- registry + constructor validation (no OpenVINO needed) ----------


def test_plugin_registered_under_yolo_openvino() -> None:
    import backbone.detection  # noqa: F401  — trigger registration

    assert "yolo_openvino" in detector_registry


def test_missing_xml_raises_filenotfound(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="OpenVINO IR not found"):
        YoloOpenvinoDetector(model_xml=tmp_path / "missing.xml", class_names=CLASS_NAMES)


def test_empty_class_names_rejected(tmp_path: Path) -> None:
    fake = tmp_path / "fake.xml"
    fake.write_bytes(b"")
    with pytest.raises(ValueError, match="class_names"):
        YoloOpenvinoDetector(model_xml=fake, class_names=[])


def test_keep_classes_must_be_subset(tmp_path: Path) -> None:
    fake = tmp_path / "fake.xml"
    fake.write_bytes(b"")
    with pytest.raises(ValueError, match="not in class_names"):
        YoloOpenvinoDetector(model_xml=fake, class_names=CLASS_NAMES, keep_classes=["ghost"])


# ---------- end-to-end inference (requires openvino) ----------


def _ir_from_constant(out: np.ndarray, tmp_path: Path) -> Path:
    ov = pytest.importorskip("openvino")
    onnx_path = tmp_path / "stub.onnx"
    _build_constant_yolo_onnx(out, onnx_path)
    xml_path = tmp_path / "stub.xml"
    ov.save_model(ov.convert_model(str(onnx_path)), str(xml_path))
    return xml_path


def test_detect_returns_detection_for_strong_anchor(tmp_path: Path) -> None:
    out = np.zeros((1, 4 + NC, NUM_ANCHORS), dtype=np.float32)
    out[0, 0, 0] = 320.0  # cx
    out[0, 1, 0] = 320.0  # cy
    out[0, 2, 0] = 100.0  # w
    out[0, 3, 0] = 200.0  # h
    out[0, 4, 0] = 0.9    # person score
    xml_path = _ir_from_constant(out, tmp_path)

    det = YoloOpenvinoDetector(model_xml=xml_path, class_names=CLASS_NAMES, device="CPU")
    result = det.detect(_make_pair_with_one_image())
    assert set(result) == {"cam_a"}
    assert len(result["cam_a"]) == 1
    d = result["cam_a"][0]
    assert d.cls == "person"
    assert d.confidence == pytest.approx(0.9, abs=1e-4)
    x1, y1, x2, y2 = d.bbox_xyxy
    assert x1 < x2 and y1 < y2


def test_detect_empty_output_yields_no_detections(tmp_path: Path) -> None:
    out = np.zeros((1, 4 + NC, NUM_ANCHORS), dtype=np.float32)
    xml_path = _ir_from_constant(out, tmp_path)
    det = YoloOpenvinoDetector(model_xml=xml_path, class_names=CLASS_NAMES, device="CPU")
    assert det.detect(_make_pair_with_one_image()) == {"cam_a": []}


def test_warmup_does_not_raise(tmp_path: Path) -> None:
    out = np.zeros((1, 4 + NC, NUM_ANCHORS), dtype=np.float32)
    xml_path = _ir_from_constant(out, tmp_path)
    YoloOpenvinoDetector(model_xml=xml_path, class_names=CLASS_NAMES, device="CPU").warmup()
