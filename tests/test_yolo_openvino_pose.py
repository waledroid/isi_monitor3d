"""``yolo_openvino_pose`` plugin — registration, ctor validation, decode.

Mirrors :mod:`tests.test_yolo_openvino` for the pose head. The end-to-end
tests fabricate a constant-output stub ONNX (OpenVINO converts/reads it
natively) whose head is ``(1, 4 + 1 + 17*3, A)`` — one strong person anchor
with ankle keypoints, so the ankle-midpoint ``foot_uv`` rule is exercised.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backbone.core.interfaces import detector_registry
from backbone.core.types import Frame, FramePair
from backbone.detection.yolo_openvino_pose import YoloOpenvinoPoseDetector

NUM_ANCHORS = 64
K = 17                      # COCO keypoints
NC = 1                      # person


def _make_pair_with_one_image() -> FramePair:
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    f = Frame(camera_id="cam_a", capture_ts=10.0, frame_idx=0, image=img)
    return FramePair(capture_ts=10.0, frame_idx=0, frames={"cam_a": f})


def _build_constant_pose_onnx(out_tensor: np.ndarray, path: Path) -> None:
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    input_tv = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])
    output_tv = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, list(out_tensor.shape))
    const_tensor = numpy_helper.from_array(out_tensor.astype(np.float32), name="const_out")
    const_node = helper.make_node("Constant", inputs=[], outputs=["output"],
                                  value=const_tensor)
    graph = helper.make_graph([const_node], "constant_pose_stub", [input_tv], [output_tv])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10,
        producer_name="isi-monitor3d-tests")
    onnx.save(model, str(path))


def _ir_from_constant(out: np.ndarray, tmp_path: Path) -> Path:
    ov = pytest.importorskip("openvino")
    onnx_path = tmp_path / "pose_stub.onnx"
    _build_constant_pose_onnx(out, onnx_path)
    xml_path = tmp_path / "pose_stub.xml"
    ov.save_model(ov.convert_model(str(onnx_path)), str(xml_path))
    return xml_path


def test_plugin_registered_under_yolo_openvino_pose() -> None:
    import backbone.detection  # noqa: F401  — trigger registration

    assert "yolo_openvino_pose" in detector_registry


def test_missing_xml_raises_filenotfound(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="OpenVINO IR not found"):
        YoloOpenvinoPoseDetector(model_xml=tmp_path / "missing.xml")


def test_detect_returns_person_with_keypoints_and_ankle_foot(tmp_path: Path) -> None:
    out = np.zeros((1, 4 + NC + K * 3, NUM_ANCHORS), dtype=np.float32)
    out[0, 0, 0] = 320.0    # cx
    out[0, 1, 0] = 300.0    # cy
    out[0, 2, 0] = 100.0    # w
    out[0, 3, 0] = 200.0    # h
    out[0, 4, 0] = 0.9      # person score
    # ankles (kpt 15/16) visible at (300, 395) and (340, 405) → foot (320, 400)
    base = 4 + NC
    out[0, base + 15 * 3 + 0, 0] = 300.0
    out[0, base + 15 * 3 + 1, 0] = 395.0
    out[0, base + 15 * 3 + 2, 0] = 0.9
    out[0, base + 16 * 3 + 0, 0] = 340.0
    out[0, base + 16 * 3 + 1, 0] = 405.0
    out[0, base + 16 * 3 + 2, 0] = 0.9
    xml_path = _ir_from_constant(out, tmp_path)

    det = YoloOpenvinoPoseDetector(model_xml=xml_path, device="CPU")
    result = det.detect(_make_pair_with_one_image())
    assert set(result) == {"cam_a"}
    assert len(result["cam_a"]) == 1
    d = result["cam_a"][0]
    assert d.cls == "person"
    assert d.confidence == pytest.approx(0.9, abs=1e-4)
    kpts = np.asarray(d.keypoints_uv).reshape(-1, 3)
    assert kpts.shape == (K, 3)
    # 640x640 source into a 640x640 model frame: letterbox is identity.
    assert d.foot_uv[0] == pytest.approx(320.0, abs=1.0)
    assert d.foot_uv[1] == pytest.approx(400.0, abs=1.0)


def test_detect_empty_output_yields_no_persons(tmp_path: Path) -> None:
    out = np.zeros((1, 4 + NC + K * 3, NUM_ANCHORS), dtype=np.float32)
    xml_path = _ir_from_constant(out, tmp_path)
    det = YoloOpenvinoPoseDetector(model_xml=xml_path, device="CPU")
    result = det.detect(_make_pair_with_one_image())
    assert result["cam_a"] == []


def test_warmup_does_not_raise(tmp_path: Path) -> None:
    out = np.zeros((1, 4 + NC + K * 3, NUM_ANCHORS), dtype=np.float32)
    xml_path = _ir_from_constant(out, tmp_path)
    YoloOpenvinoPoseDetector(model_xml=xml_path, device="CPU").warmup()
