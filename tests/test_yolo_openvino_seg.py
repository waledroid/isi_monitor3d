"""``YoloOpenvinoSegDetector`` — registration + constructor validation + end-to-end
inference on a tiny hand-built 2-output OpenVINO IR (converted from ONNX in test).

Mirrors :mod:`tests.test_yolo_onnx_seg` for the OpenVINO backend. Skipped cleanly
when the ``openvino`` package isn't installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from backbone.core.interfaces import detector_registry
from backbone.core.types import Frame, FramePair

# Skip the entire module when OpenVINO isn't installed in the env.
ov = pytest.importorskip("openvino")

from backbone.detection.yolo_openvino_seg import YoloOpenvinoSegDetector  # noqa: E402

CLASS_NAMES = ["palette"]
NC = len(CLASS_NAMES)
NM = 32                  # YOLO11-seg's mask coefficient count
NUM_ANCHORS = 64         # tiny — fast tests
MH, MW = 160, 160        # YOLO11-seg proto map size at 640 input (stride 4)


def _build_constant_seg_onnx(head: np.ndarray, protos: np.ndarray, path: Path) -> None:
    """Minimal 2-output ONNX returning constant head + protos regardless of input."""
    input_tv = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])
    head_tv = helper.make_tensor_value_info("head", TensorProto.FLOAT, list(head.shape))
    protos_tv = helper.make_tensor_value_info("protos", TensorProto.FLOAT, list(protos.shape))
    head_const = numpy_helper.from_array(head.astype(np.float32), name="head_const")
    protos_const = numpy_helper.from_array(protos.astype(np.float32), name="protos_const")
    head_node = helper.make_node("Constant", inputs=[], outputs=["head"], value=head_const)
    protos_node = helper.make_node("Constant", inputs=[], outputs=["protos"], value=protos_const)
    graph = helper.make_graph(
        nodes=[head_node, protos_node], name="constant_yolo_seg_stub",
        inputs=[input_tv], outputs=[head_tv, protos_tv],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10,
        producer_name="isi-monitor3d-tests",
    )
    onnx.save(model, str(path))


def _onnx_to_ir(onnx_path: Path, xml_path: Path) -> None:
    """Convert ONNX → OpenVINO IR (writes ``xml_path`` + matching .bin)."""
    ir_model = ov.convert_model(str(onnx_path))
    ov.save_model(ir_model, str(xml_path), compress_to_fp16=False)


def _make_pair_with_one_image() -> FramePair:
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    f = Frame(camera_id="cam_a", capture_ts=10.0, frame_idx=0, image=img)
    return FramePair(capture_ts=10.0, frame_idx=0, frames={"cam_a": f})


# ---------- registry + constructor ----------


def test_plugin_registered_under_yolo_openvino_seg() -> None:
    import backbone.detection  # noqa: F401  — trigger registration

    assert "yolo_openvino_seg" in detector_registry


def test_missing_xml_raises_filenotfound(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="OpenVINO IR not found"):
        YoloOpenvinoSegDetector(model_xml=tmp_path / "missing.xml", class_names=CLASS_NAMES)


def test_empty_class_names_rejected(tmp_path: Path) -> None:
    # Build a real IR so we get past the file-exists check before the class_names guard.
    head = np.zeros((1, 4 + NC + NM, NUM_ANCHORS), dtype=np.float32)
    protos = np.zeros((1, NM, MH, MW), dtype=np.float32)
    onnx_path = tmp_path / "seg.onnx"
    xml_path = tmp_path / "seg.xml"
    _build_constant_seg_onnx(head, protos, onnx_path)
    _onnx_to_ir(onnx_path, xml_path)
    with pytest.raises(ValueError, match="class_names"):
        YoloOpenvinoSegDetector(model_xml=xml_path, class_names=[])


def test_one_output_rejected(tmp_path: Path) -> None:
    """A single-output IR (i.e. a DETECT model) should be rejected by the seg plugin."""
    out = np.zeros((1, 4 + NC, NUM_ANCHORS), dtype=np.float32)
    input_tv = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])
    output_tv = helper.make_tensor_value_info("output", TensorProto.FLOAT, list(out.shape))
    const = numpy_helper.from_array(out, name="const_out")
    node = helper.make_node("Constant", inputs=[], outputs=["output"], value=const)
    graph = helper.make_graph([node], "detect_stub", [input_tv], [output_tv])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10,
    )
    onnx_path = tmp_path / "detect_only.onnx"
    xml_path = tmp_path / "detect_only.xml"
    onnx.save(model, str(onnx_path))
    _onnx_to_ir(onnx_path, xml_path)
    with pytest.raises(ValueError, match="expected 2"):
        YoloOpenvinoSegDetector(model_xml=xml_path, class_names=CLASS_NAMES, device="CPU")


# ---------- end-to-end inference ----------


def test_detect_returns_detection_with_mask(tmp_path: Path) -> None:
    """Build a seg-shaped IR with one strong anchor + a single hot proto map; verify
    a Detection comes out with a non-empty boolean mask inside the bbox."""
    head = np.zeros((1, 4 + NC + NM, NUM_ANCHORS), dtype=np.float32)
    head[0, 0, 0] = 320.0          # cx (target-frame)
    head[0, 1, 0] = 320.0          # cy
    head[0, 2, 0] = 100.0          # w
    head[0, 3, 0] = 200.0          # h
    head[0, 4, 0] = 0.9            # class score
    head[0, 4 + NC + 0, 0] = 10.0  # mask coefficient for proto 0

    protos = np.zeros((1, NM, MH, MW), dtype=np.float32)
    protos[0, 0, :, :] = 5.0       # proto 0 uniformly "on" → mask ≈ 1 everywhere

    onnx_path = tmp_path / "seg.onnx"
    xml_path = tmp_path / "seg.xml"
    _build_constant_seg_onnx(head, protos, onnx_path)
    _onnx_to_ir(onnx_path, xml_path)

    det = YoloOpenvinoSegDetector(
        model_xml=xml_path, class_names=CLASS_NAMES, device="CPU",
    )
    result = det.detect(_make_pair_with_one_image())
    assert set(result) == {"cam_a"}
    assert len(result["cam_a"]) == 1
    d = result["cam_a"][0]
    assert d.cls == "palette"
    assert d.confidence == pytest.approx(0.9, abs=1e-4)
    x1, y1, x2, y2 = d.bbox_xyxy
    assert x1 < x2 and y1 < y2
    assert d.mask is not None
    assert d.mask.shape == (1080, 1920)
    assert d.mask.dtype == np.bool_
    inside_pixels = d.mask[int(y1):int(y2), int(x1):int(x2)].sum()
    total_inside = max(1, (int(y2) - int(y1)) * (int(x2) - int(x1)))
    assert inside_pixels / total_inside > 0.9
    outside_mask = d.mask.copy()
    outside_mask[int(y1):int(y2), int(x1):int(x2)] = False
    assert outside_mask.sum() == 0


def test_detect_empty_yields_no_detections(tmp_path: Path) -> None:
    head = np.zeros((1, 4 + NC + NM, NUM_ANCHORS), dtype=np.float32)
    protos = np.zeros((1, NM, MH, MW), dtype=np.float32)
    onnx_path = tmp_path / "seg_empty.onnx"
    xml_path = tmp_path / "seg_empty.xml"
    _build_constant_seg_onnx(head, protos, onnx_path)
    _onnx_to_ir(onnx_path, xml_path)
    det = YoloOpenvinoSegDetector(
        model_xml=xml_path, class_names=CLASS_NAMES, device="CPU",
    )
    assert det.detect(_make_pair_with_one_image()) == {"cam_a": []}
