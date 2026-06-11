"""``postprocess.decode_rfdetr_seg`` — synthetic RF-DETR output → ``Detection`` list.

Hermetic: hand-craft ``dets`` / ``labels`` / ``masks`` numpy arrays (one strong
"palette" query + one below-threshold) and verify the decoded ``Detection`` —
class name, box mapped to source pixels, ``foot_uv`` bottom-centre, a full-frame
bool ``mask`` of the right shape, and low-score queries dropped. No real ONNX.

Mirrors the style of ``tests/test_detection_postprocess.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from backbone.detection.postprocess import decode_rfdetr_seg

# class_names[i] ↔ COCO logits column i+1 (column 0 = background).
CLASS_NAMES = ["palette", "carton", "polybag"]
HEAD_NC = 91          # RF-DETR keeps the pretrained COCO head width
NUM_QUERIES = 200
MASK_S = 108          # 432 / 4


def _logit(p: float) -> float:
    """Inverse sigmoid — so we can set a target post-sigmoid score directly."""
    p = min(max(p, 1e-6), 1 - 1e-6)
    return float(np.log(p / (1 - p)))


def _empty_outputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All-background queries: dets zero, labels strongly negative, masks negative."""
    dets = np.zeros((NUM_QUERIES, 4), dtype=np.float32)
    labels = np.full((NUM_QUERIES, HEAD_NC), -10.0, dtype=np.float32)
    masks = np.full((NUM_QUERIES, MASK_S, MASK_S), -10.0, dtype=np.float32)
    return dets, labels, masks


def _set_query(
    dets: np.ndarray,
    labels: np.ndarray,
    masks: np.ndarray | None,
    q: int,
    *,
    cxcywh_norm: tuple[float, float, float, float],
    name_idx: int,        # index into CLASS_NAMES (→ logits column name_idx+1)
    score: float,
    mask_fill: bool = True,
) -> None:
    dets[q] = cxcywh_norm
    labels[q, name_idx + 1] = _logit(score)
    if masks is not None and mask_fill:
        masks[q, :, :] = 10.0      # sigmoid(10) ≈ 1 → mask whole field, box-confined later


# ---------- basic decode + box mapping + foot + mask ----------


def test_single_strong_palette_query_decoded() -> None:
    dets, labels, masks = _empty_outputs()
    orig_w, orig_h = 1920, 1080
    # cxcywh-norm (0.5, 0.5, 0.25, 0.5) → source xyxy:
    #   x: cx=960  w=480 → x1=720  x2=1200
    #   y: cy=540  h=540 → y1=270  y2=810
    _set_query(dets, labels, masks, 0,
               cxcywh_norm=(0.5, 0.5, 0.25, 0.5), name_idx=0, score=0.9)
    dets_out = decode_rfdetr_seg(
        dets, labels, masks,
        camera_id="cam_a", capture_ts=1.0,
        source_wh=(orig_w, orig_h), class_names=CLASS_NAMES,
        confidence_threshold=0.3, mask_threshold=0.5,
    )
    assert len(dets_out) == 1
    d = dets_out[0]
    assert d.camera_id == "cam_a"
    assert d.capture_ts == 1.0
    assert d.cls == "palette"
    assert d.confidence == pytest.approx(0.9, abs=1e-3)
    assert d.bbox_xyxy == pytest.approx((720.0, 270.0, 1200.0, 810.0), abs=1.0)
    # foot = bottom-centre = (cx_px, y2)
    assert d.foot_uv[0] == pytest.approx(960.0, abs=1.0)
    assert d.foot_uv[1] == pytest.approx(810.0, abs=1.0)
    # full-frame bool mask of source shape, non-empty inside the box.
    assert d.mask is not None
    assert d.mask.dtype == bool
    assert d.mask.shape == (orig_h, orig_w)
    assert d.mask.any()
    # mask pixels live inside the bbox (box-confined assembly).
    ys, xs = np.where(d.mask)
    assert xs.min() >= 719 and xs.max() <= 1201
    assert ys.min() >= 269 and ys.max() <= 811


def test_below_threshold_query_dropped() -> None:
    dets, labels, masks = _empty_outputs()
    _set_query(dets, labels, masks, 0,
               cxcywh_norm=(0.5, 0.5, 0.2, 0.2), name_idx=0, score=0.1)
    out = decode_rfdetr_seg(
        dets, labels, masks,
        camera_id="cam_a", capture_ts=0.0,
        source_wh=(1920, 1080), class_names=CLASS_NAMES,
        confidence_threshold=0.3,
    )
    assert out == []


def test_strong_and_weak_query_only_strong_kept() -> None:
    dets, labels, masks = _empty_outputs()
    _set_query(dets, labels, masks, 0,
               cxcywh_norm=(0.5, 0.5, 0.2, 0.2), name_idx=0, score=0.85)   # palette, strong
    _set_query(dets, labels, masks, 1,
               cxcywh_norm=(0.2, 0.2, 0.1, 0.1), name_idx=1, score=0.1)    # carton, weak
    out = decode_rfdetr_seg(
        dets, labels, masks,
        camera_id="cam_a", capture_ts=0.0,
        source_wh=(640, 480), class_names=CLASS_NAMES,
        confidence_threshold=0.3,
    )
    assert [d.cls for d in out] == ["palette"]


# ---------- class index → name mapping (COCO 1=palette,2=carton,3=polybag) ----------


def test_class_column_maps_to_name() -> None:
    dets, labels, masks = _empty_outputs()
    # Put a polybag (name_idx=2 → logits column 3) at high score.
    _set_query(dets, labels, masks, 5,
               cxcywh_norm=(0.4, 0.4, 0.1, 0.1), name_idx=2, score=0.8)
    out = decode_rfdetr_seg(
        dets, labels, masks,
        camera_id="cam_a", capture_ts=0.0,
        source_wh=(640, 480), class_names=CLASS_NAMES,
        confidence_threshold=0.3,
    )
    assert len(out) == 1
    assert out[0].cls == "polybag"


def test_results_sorted_by_descending_confidence() -> None:
    dets, labels, masks = _empty_outputs()
    _set_query(dets, labels, masks, 0,
               cxcywh_norm=(0.2, 0.2, 0.1, 0.1), name_idx=0, score=0.5)
    _set_query(dets, labels, masks, 1,
               cxcywh_norm=(0.6, 0.6, 0.1, 0.1), name_idx=1, score=0.95)
    _set_query(dets, labels, masks, 2,
               cxcywh_norm=(0.8, 0.8, 0.1, 0.1), name_idx=2, score=0.7)
    out = decode_rfdetr_seg(
        dets, labels, masks,
        camera_id="cam_a", capture_ts=0.0,
        source_wh=(640, 480), class_names=CLASS_NAMES,
        confidence_threshold=0.3,
    )
    assert [round(d.confidence, 2) for d in out] == [0.95, 0.70, 0.50]
    assert [d.cls for d in out] == ["carton", "polybag", "palette"]


# ---------- robustness ----------


def test_zero_detections_returns_empty() -> None:
    dets, labels, masks = _empty_outputs()
    out = decode_rfdetr_seg(
        dets, labels, masks,
        camera_id="cam_a", capture_ts=0.0,
        source_wh=(1920, 1080), class_names=CLASS_NAMES,
        confidence_threshold=0.3,
    )
    assert out == []


def test_masks_none_yields_none_mask() -> None:
    dets, labels, _ = _empty_outputs()
    _set_query(dets, labels, None, 0,
               cxcywh_norm=(0.5, 0.5, 0.2, 0.2), name_idx=0, score=0.9)
    out = decode_rfdetr_seg(
        dets, labels, None,
        camera_id="cam_a", capture_ts=0.0,
        source_wh=(640, 480), class_names=CLASS_NAMES,
        confidence_threshold=0.3,
    )
    assert len(out) == 1
    assert out[0].mask is None


def test_wrong_dets_shape_raises() -> None:
    dets = np.zeros((NUM_QUERIES, 5), dtype=np.float32)   # 5 instead of 4
    labels = np.full((NUM_QUERIES, HEAD_NC), -10.0, dtype=np.float32)
    with pytest.raises(ValueError, match="dets"):
        decode_rfdetr_seg(
            dets, labels, None,
            camera_id="cam_a", capture_ts=0.0,
            source_wh=(640, 480), class_names=CLASS_NAMES,
        )
