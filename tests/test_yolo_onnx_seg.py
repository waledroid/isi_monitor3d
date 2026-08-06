"""``YoloOnnxSegDetector`` — registration + constructor validation + end-to-end
inference on a tiny hand-built 2-output ONNX (no real weights required).

Builds a constant ONNX returning ``(1, 4+nc+nm, A)`` head + ``(1, nm, mh, mw)``
protos so the whole letterbox → ORT → decode_yolo11_seg → mask pipeline is
exercised end-to-end without a trained model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from backbone.core.interfaces import detector_registry
from backbone.core.types import Frame, FramePair
from backbone.detection.yolo_onnx_seg import YoloOnnxSegDetector

CLASS_NAMES = ["palette"]
NC = len(CLASS_NAMES)
NM = 32                  # YOLO11-seg's mask coefficient count
NUM_ANCHORS = 64         # tiny — fast tests
MH, MW = 160, 160        # YOLO11-seg proto map size at 640 input (stride 4)


def _build_constant_seg_onnx(head: np.ndarray, protos: np.ndarray, path: Path,
                             batch_dim=1) -> None:
    """Minimal 2-output ONNX returning constant head + protos regardless of input.

    ``batch_dim`` declares the input batch dim — an int (fixed) or a string like
    "N" (dynamic), to exercise ``supports_batch``."""
    input_tv = helper.make_tensor_value_info(
        "images", TensorProto.FLOAT, [batch_dim, 3, 640, 640])
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


def _make_pair_with_one_image() -> FramePair:
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    f = Frame(camera_id="cam_a", capture_ts=10.0, frame_idx=0, image=img)
    return FramePair(capture_ts=10.0, frame_idx=0, frames={"cam_a": f})


# ---------- registry + constructor ----------


def test_plugin_registered_under_yolo_onnx_seg() -> None:
    import backbone.detection  # noqa: F401  — trigger registration

    assert "yolo_onnx_seg" in detector_registry


def test_missing_onnx_raises_filenotfound(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="ONNX file not found"):
        YoloOnnxSegDetector(onnx_path=tmp_path / "missing.onnx", class_names=CLASS_NAMES)


def test_empty_class_names_rejected(tmp_path: Path) -> None:
    head = np.zeros((1, 4 + NC + NM, NUM_ANCHORS), dtype=np.float32)
    protos = np.zeros((1, NM, MH, MW), dtype=np.float32)
    model_path = tmp_path / "seg_stub.onnx"
    _build_constant_seg_onnx(head, protos, model_path)
    with pytest.raises(ValueError, match="no class names"):
        YoloOnnxSegDetector(onnx_path=model_path, class_names=[])


def test_one_output_rejected(tmp_path: Path) -> None:
    """A single-output ONNX (i.e. a DETECT model) should be rejected by the seg plugin."""
    # Reuse the detect test's helper inline (single output).
    out = np.zeros((1, 4 + NC, NUM_ANCHORS), dtype=np.float32)
    input_tv = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])
    output_tv = helper.make_tensor_value_info("output", TensorProto.FLOAT, list(out.shape))
    const = numpy_helper.from_array(out, name="const_out")
    node = helper.make_node("Constant", inputs=[], outputs=["output"], value=const)
    graph = helper.make_graph([node], "detect_stub", [input_tv], [output_tv])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10,
    )
    p = tmp_path / "detect_only.onnx"
    onnx.save(model, str(p))
    with pytest.raises(ValueError, match="expected 2"):
        YoloOnnxSegDetector(onnx_path=p, class_names=CLASS_NAMES,
                            providers=["CPUExecutionProvider"])


# ---------- supports_batch ----------


def test_supports_batch_true_for_dynamic_batch_dim(tmp_path: Path) -> None:
    head = np.zeros((1, 4 + NC + NM, NUM_ANCHORS), dtype=np.float32)
    protos = np.zeros((1, NM, MH, MW), dtype=np.float32)
    model_path = tmp_path / "seg_dyn.onnx"
    _build_constant_seg_onnx(head, protos, model_path, batch_dim="N")
    det = YoloOnnxSegDetector(onnx_path=model_path, class_names=CLASS_NAMES,
                              providers=["CPUExecutionProvider"])
    assert det.supports_batch is True


def test_supports_batch_false_for_fixed_batch_dim(tmp_path: Path) -> None:
    head = np.zeros((1, 4 + NC + NM, NUM_ANCHORS), dtype=np.float32)
    protos = np.zeros((1, NM, MH, MW), dtype=np.float32)
    model_path = tmp_path / "seg_fixed.onnx"
    _build_constant_seg_onnx(head, protos, model_path, batch_dim=1)
    det = YoloOnnxSegDetector(onnx_path=model_path, class_names=CLASS_NAMES,
                              providers=["CPUExecutionProvider"])
    assert det.supports_batch is False


# ---------- end-to-end inference ----------


def test_detect_returns_detection_with_mask(tmp_path: Path) -> None:
    """Build a seg-shaped ONNX with one strong anchor + a single hot proto map; verify
    a Detection comes out with a non-empty boolean mask inside the bbox."""
    head = np.zeros((1, 4 + NC + NM, NUM_ANCHORS), dtype=np.float32)
    # Anchor 0: cx, cy, w, h, class_score, mask_coeffs[0]=1 (others 0).
    head[0, 0, 0] = 320.0          # cx (target-frame)
    head[0, 1, 0] = 320.0          # cy
    head[0, 2, 0] = 100.0          # w
    head[0, 3, 0] = 200.0          # h
    head[0, 4, 0] = 0.9            # class score (single class)
    head[0, 4 + NC + 0, 0] = 10.0  # mask coefficient for proto 0 (post-sigmoid ≈ 1)

    protos = np.zeros((1, NM, MH, MW), dtype=np.float32)
    protos[0, 0, :, :] = 5.0       # proto 0 is uniformly "on" → mask = sigmoid(10*5)=~1

    model_path = tmp_path / "seg_stub.onnx"
    _build_constant_seg_onnx(head, protos, model_path)

    det = YoloOnnxSegDetector(
        onnx_path=model_path, class_names=CLASS_NAMES,
        providers=["CPUExecutionProvider"],
    )
    result = det.detect(_make_pair_with_one_image())
    assert set(result) == {"cam_a"}
    assert len(result["cam_a"]) == 1
    d = result["cam_a"][0]
    assert d.cls == "palette"
    assert d.confidence == pytest.approx(0.9, abs=1e-4)
    # bbox is in source-frame (1080x1920) — just verify plausibility.
    x1, y1, x2, y2 = d.bbox_xyxy
    assert x1 < x2 and y1 < y2
    # Mask: full-frame HxW bool, all True inside the bbox region.
    assert d.mask is not None
    assert d.mask.shape == (1080, 1920)
    assert d.mask.dtype == np.bool_
    # The mask should have pixels set inside the bbox (uniform proto + strong coeff → all True).
    inside_pixels = d.mask[int(y1):int(y2), int(x1):int(x2)].sum()
    total_inside = max(1, (int(y2) - int(y1)) * (int(x2) - int(x1)))
    assert inside_pixels / total_inside > 0.9
    # And nothing outside the bbox.
    outside_mask = d.mask.copy()
    outside_mask[int(y1):int(y2), int(x1):int(x2)] = False
    assert outside_mask.sum() == 0


def test_detect_empty_yields_no_detections(tmp_path: Path) -> None:
    head = np.zeros((1, 4 + NC + NM, NUM_ANCHORS), dtype=np.float32)
    protos = np.zeros((1, NM, MH, MW), dtype=np.float32)
    model_path = tmp_path / "seg_empty.onnx"
    _build_constant_seg_onnx(head, protos, model_path)
    det = YoloOnnxSegDetector(
        onnx_path=model_path, class_names=CLASS_NAMES,
        providers=["CPUExecutionProvider"],
    )
    assert det.detect(_make_pair_with_one_image()) == {"cam_a": []}


def test_static_batch_seg_model_survives_multi_camera_pair(tmp_path: Path) -> None:
    """Regression (2026-08-06): same static-batch fallback as ``yolo_onnx`` —
    the seg plugin is what zone scope actually wraps, so a ``dynamic=False``
    seg export crashed every live tick until re-exported."""
    head = np.zeros((1, 4 + NC + NM, NUM_ANCHORS), dtype=np.float32)
    protos = np.zeros((1, NM, 160, 160), dtype=np.float32)
    model_path = tmp_path / "fixed.onnx"
    _build_constant_seg_onnx(head, protos, model_path, batch_dim=1)
    det = YoloOnnxSegDetector(onnx_path=model_path, class_names=CLASS_NAMES,
                              providers=["CPUExecutionProvider"])
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    pair = FramePair(
        capture_ts=10.0, frame_idx=0,
        frames={cid: Frame(camera_id=cid, capture_ts=10.0, frame_idx=0, image=img)
                for cid in ("cam_a", "cam_b")})
    result = det.detect(pair)
    assert set(result) == {"cam_a", "cam_b"}
