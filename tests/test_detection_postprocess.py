"""``postprocess.decode_yolo11_detect`` — synthetic-output → ``Detection`` list.

Hermetic: hand-craft a ``(4 + nc, anchors)`` tensor with controlled anchor
values, verify the decoded bboxes / foot points / class assignment / NMS /
class-filter behavior are exactly what we expect — no real model needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from backbone.detection.postprocess import (
    decode_yolo11_detect,
    decode_yolo11_pose,
    decode_yolo11_seg,
)
from backbone.detection.preprocess import LetterboxResult

CLASS_NAMES = ["person", "forklift", "pallet"]
NC = len(CLASS_NAMES)


def _identity_letterbox(src_h: int = 640, src_w: int = 640) -> LetterboxResult:
    """A letterbox whose target == source — no scaling, no padding to undo."""
    return LetterboxResult(
        tensor=np.zeros((3, src_h, src_w), dtype=np.float32),
        scale=1.0,
        pad_xy=(0, 0),
        source_shape_hw=(src_h, src_w),
    )


def _empty_output(num_anchors: int = 100) -> np.ndarray:
    """A YOLO output where every anchor scores ~0 on every class."""
    out = np.zeros((4 + NC, num_anchors), dtype=np.float32)
    # cxcywh stays at zero — boxes are degenerate.
    return out


def _set_anchor(
    out: np.ndarray,
    idx: int,
    *,
    cxcywh: tuple[float, float, float, float],
    cls_idx: int,
    score: float,
) -> None:
    """Inject a single strong anchor into the tensor at index `idx`."""
    out[0, idx] = cxcywh[0]
    out[1, idx] = cxcywh[1]
    out[2, idx] = cxcywh[2]
    out[3, idx] = cxcywh[3]
    out[4 + cls_idx, idx] = score


# ---------- basic decode ----------


def test_single_strong_anchor_decoded() -> None:
    out = _empty_output()
    _set_anchor(out, idx=0, cxcywh=(320.0, 480.0, 100.0, 200.0), cls_idx=0, score=0.9)
    dets = decode_yolo11_detect(
        out,
        camera_id="cam_a",
        capture_ts=1.0,
        letterbox_meta=_identity_letterbox(),
        class_names=CLASS_NAMES,
        confidence_threshold=0.25,
    )
    assert len(dets) == 1
    d = dets[0]
    assert d.camera_id == "cam_a"
    assert d.capture_ts == 1.0
    assert d.cls == "person"
    assert d.confidence == pytest.approx(0.9, abs=1e-6)
    # cxcywh (320, 480, 100, 200) → xyxy (270, 380, 370, 580)
    assert d.bbox_xyxy == pytest.approx((270.0, 380.0, 370.0, 580.0), abs=1e-4)
    # Foot = bottom-centre = (cx, y_bottom) = (320, 580)
    assert d.foot_uv == pytest.approx((320.0, 580.0), abs=1e-4)


def test_below_threshold_anchors_dropped() -> None:
    out = _empty_output()
    _set_anchor(out, idx=0, cxcywh=(100.0, 100.0, 50.0, 50.0), cls_idx=0, score=0.1)
    dets = decode_yolo11_detect(
        out,
        camera_id="cam_a",
        capture_ts=0.0,
        letterbox_meta=_identity_letterbox(),
        class_names=CLASS_NAMES,
        confidence_threshold=0.25,
    )
    assert dets == []


def test_results_sorted_by_descending_confidence() -> None:
    out = _empty_output()
    _set_anchor(out, idx=0, cxcywh=(100.0, 100.0, 40.0, 40.0), cls_idx=0, score=0.50)
    _set_anchor(out, idx=10, cxcywh=(500.0, 500.0, 40.0, 40.0), cls_idx=1, score=0.95)
    _set_anchor(out, idx=20, cxcywh=(300.0, 300.0, 40.0, 40.0), cls_idx=2, score=0.70)
    dets = decode_yolo11_detect(
        out,
        camera_id="cam_a",
        capture_ts=0.0,
        letterbox_meta=_identity_letterbox(),
        class_names=CLASS_NAMES,
        confidence_threshold=0.25,
    )
    assert [round(d.confidence, 2) for d in dets] == [0.95, 0.70, 0.50]
    assert [d.cls for d in dets] == ["forklift", "pallet", "person"]


def test_correct_class_picked_from_argmax() -> None:
    out = _empty_output()
    # One anchor where forklift is the winning class.
    out[0, 5] = 100.0  # cx
    out[1, 5] = 100.0  # cy
    out[2, 5] = 40.0   # w
    out[3, 5] = 40.0   # h
    out[4, 5] = 0.2    # person — below threshold on its own
    out[5, 5] = 0.8    # forklift — wins
    out[6, 5] = 0.3    # pallet
    dets = decode_yolo11_detect(
        out,
        camera_id="cam_a",
        capture_ts=0.0,
        letterbox_meta=_identity_letterbox(),
        class_names=CLASS_NAMES,
        confidence_threshold=0.25,
    )
    assert len(dets) == 1
    assert dets[0].cls == "forklift"
    assert dets[0].confidence == pytest.approx(0.8, abs=1e-6)


# ---------- NMS ----------


def test_overlapping_anchors_deduped_by_nms() -> None:
    out = _empty_output()
    # Two anchors at near-identical boxes, both confident.
    _set_anchor(out, idx=0, cxcywh=(300.0, 300.0, 100.0, 100.0), cls_idx=0, score=0.9)
    _set_anchor(out, idx=1, cxcywh=(305.0, 302.0, 100.0, 100.0), cls_idx=0, score=0.7)
    dets = decode_yolo11_detect(
        out,
        camera_id="cam_a",
        capture_ts=0.0,
        letterbox_meta=_identity_letterbox(),
        class_names=CLASS_NAMES,
        confidence_threshold=0.25,
        iou_threshold=0.45,
    )
    # NMS should keep only the higher-scoring one.
    assert len(dets) == 1
    assert dets[0].confidence == pytest.approx(0.9, abs=1e-6)


def test_distant_anchors_both_kept() -> None:
    out = _empty_output()
    _set_anchor(out, idx=0, cxcywh=(100.0, 100.0, 50.0, 50.0), cls_idx=0, score=0.8)
    _set_anchor(out, idx=1, cxcywh=(500.0, 500.0, 50.0, 50.0), cls_idx=0, score=0.7)
    dets = decode_yolo11_detect(
        out,
        camera_id="cam_a",
        capture_ts=0.0,
        letterbox_meta=_identity_letterbox(),
        class_names=CLASS_NAMES,
        confidence_threshold=0.25,
    )
    assert len(dets) == 2


# ---------- class whitelist ----------


def test_keep_classes_filters_unconfigured() -> None:
    out = _empty_output()
    _set_anchor(out, idx=0, cxcywh=(100.0, 100.0, 50.0, 50.0), cls_idx=0, score=0.9)  # person
    _set_anchor(out, idx=1, cxcywh=(300.0, 300.0, 50.0, 50.0), cls_idx=1, score=0.9)  # forklift
    _set_anchor(out, idx=2, cxcywh=(500.0, 500.0, 50.0, 50.0), cls_idx=2, score=0.9)  # pallet
    dets = decode_yolo11_detect(
        out,
        camera_id="cam_a",
        capture_ts=0.0,
        letterbox_meta=_identity_letterbox(),
        class_names=CLASS_NAMES,
        confidence_threshold=0.25,
        keep_classes=["person", "pallet"],
    )
    assert sorted(d.cls for d in dets) == ["pallet", "person"]


# ---------- letterbox inverse on the decode side ----------


def test_decode_inverts_letterbox_to_source_coords() -> None:
    """When the input was letterboxed, decoded bboxes must be in source pixels."""
    out = _empty_output()
    # Anchor in target (640) frame at the centre with size 100x200.
    _set_anchor(out, idx=0, cxcywh=(320.0, 320.0, 100.0, 200.0), cls_idx=0, score=0.9)

    # A 1920x1080 source → letterboxed into 640x640 → scale ~= 1/3.
    src_h, src_w = 1080, 1920
    scale = 640 / 1920  # = 1/3
    new_h = round(src_h * scale)      # 360
    pad_y = (640 - new_h) // 2             # 140
    lb = LetterboxResult(
        tensor=np.zeros((3, 640, 640), dtype=np.float32),
        scale=scale,
        pad_xy=(0, pad_y),
        source_shape_hw=(src_h, src_w),
    )
    dets = decode_yolo11_detect(
        out,
        camera_id="cam_a",
        capture_ts=0.0,
        letterbox_meta=lb,
        class_names=CLASS_NAMES,
        confidence_threshold=0.25,
    )
    assert len(dets) == 1
    x1, y1, x2, y2 = dets[0].bbox_xyxy
    # Width in target: 100 px; source width = 100/scale = 300 px. ±1 for rounding.
    assert (x2 - x1) == pytest.approx(300.0, abs=1.0)
    # Height in target: 200 px; source height = 200/scale = 600 px.
    assert (y2 - y1) == pytest.approx(600.0, abs=1.0)
    # Foot point in source-frame pixels.
    foot_u, foot_v = dets[0].foot_uv
    assert foot_u == pytest.approx(960.0, abs=1.0)
    assert foot_v == y2


# ---------- error paths ----------


def test_wrong_channel_count_raises() -> None:
    out = np.zeros((3 + NC, 100), dtype=np.float32)  # 3 instead of 4 bbox channels
    with pytest.raises(ValueError, match="channels"):
        decode_yolo11_detect(
            out,
            camera_id="cam_a",
            capture_ts=0.0,
            letterbox_meta=_identity_letterbox(),
            class_names=CLASS_NAMES,
        )


def test_keep_classes_with_no_matches_returns_empty() -> None:
    out = _empty_output()
    _set_anchor(out, idx=0, cxcywh=(100.0, 100.0, 50.0, 50.0), cls_idx=0, score=0.9)
    dets = decode_yolo11_detect(
        out,
        camera_id="cam_a",
        capture_ts=0.0,
        letterbox_meta=_identity_letterbox(),
        class_names=CLASS_NAMES,
        keep_classes=["nonexistent"],
    )
    assert dets == []




# ---------- YOLO26 / end-to-end (NMS-free) seg head ----------

_NM = 32


def _end2end_seg_head(num_det: int = 300) -> np.ndarray:
    """An end-to-end seg head (num_det, 6 + nm): [x1,y1,x2,y2,score,cls, *coeffs],
    all rows zero (score 0 → padding) by default."""
    return np.zeros((num_det, 6 + _NM), dtype=np.float32)


def _protos(mh: int = 16, mw: int = 16) -> np.ndarray:
    return np.zeros((_NM, mh, mw), dtype=np.float32)


def test_end2end_seg_head_routed_and_decoded() -> None:
    """A (num_det, 6+nm) head (YOLO26) must route to the end-to-end branch, not
    raise the raw-head channel error, and decode the class id / box / mask."""
    head = _end2end_seg_head()
    head[0, :6] = [100.0, 50.0, 300.0, 400.0, 0.9, 1]   # cls 1 = forklift
    head[1, :6] = [10.0, 10.0, 40.0, 40.0, 0.05, 0]     # below threshold → dropped
    dets = decode_yolo11_seg(
        head, _protos(),
        camera_id="cam_a", capture_ts=0.0,
        letterbox_meta=_identity_letterbox(),
        target_hw=(640, 640), class_names=CLASS_NAMES,
        confidence_threshold=0.17, iou_threshold=0.45,
        keep_classes=None, mask_threshold=0.5,
    )
    assert len(dets) == 1
    assert dets[0].cls == "forklift"
    assert dets[0].confidence == pytest.approx(0.9)
    assert dets[0].bbox_xyxy == pytest.approx((100.0, 50.0, 300.0, 400.0))
    assert dets[0].mask is not None and dets[0].mask.shape == (640, 640)


def test_end2end_seg_padding_rows_dropped() -> None:
    """All-zero (padding) rows score 0 → no detections, no crash."""
    dets = decode_yolo11_seg(
        _end2end_seg_head(), _protos(),
        camera_id="cam_a", capture_ts=0.0,
        letterbox_meta=_identity_letterbox(),
        target_hw=(640, 640), class_names=CLASS_NAMES,
        confidence_threshold=0.17, iou_threshold=0.45,
    )
    assert dets == []


def test_end2end_seg_keep_classes_filter() -> None:
    head = _end2end_seg_head()
    head[0, :6] = [100.0, 50.0, 300.0, 400.0, 0.9, 0]   # cls 0 = person
    head[1, :6] = [120.0, 60.0, 320.0, 420.0, 0.8, 1]   # cls 1 = forklift
    dets = decode_yolo11_seg(
        head, _protos(),
        camera_id="cam_a", capture_ts=0.0,
        letterbox_meta=_identity_letterbox(),
        target_hw=(640, 640), class_names=CLASS_NAMES,
        confidence_threshold=0.17, iou_threshold=0.45,
        keep_classes=["forklift"],
    )
    assert [d.cls for d in dets] == ["forklift"]


# ---------- YOLO11-pose (person) ----------

_NK = 17  # COCO keypoints


def _pose_head(num_anchors: int = 100) -> np.ndarray:
    """A pose head (4 + nc(1) + K*3, A), all anchors zero."""
    return np.zeros((4 + 1 + _NK * 3, num_anchors), dtype=np.float32)


def _set_pose(out: np.ndarray, idx: int, cxcywh, score: float, kpts: dict) -> None:
    out[0, idx], out[1, idx], out[2, idx], out[3, idx] = cxcywh
    out[4, idx] = score                       # single person class
    for kp_i, (x, y, conf) in kpts.items():   # keypoints start at row 5, 3 rows each
        base = 5 + kp_i * 3
        out[base, idx], out[base + 1, idx], out[base + 2, idx] = x, y, conf


def test_pose_decode_foot_at_ankle_midpoint() -> None:
    out = _pose_head()
    # box centre (320,400) size (40,200) → xyxy (300,300,340,500); ankles at y=490.
    _set_pose(out, 0, (320.0, 400.0, 40.0, 200.0), 0.9,
              {15: (310.0, 490.0, 0.9), 16: (330.0, 490.0, 0.9)})
    dets = decode_yolo11_pose(
        out, camera_id="cam_a", capture_ts=0.0,
        letterbox_meta=_identity_letterbox(), class_names=["person"],
        confidence_threshold=0.25, kpt_conf=0.3,
    )
    assert len(dets) == 1
    d = dets[0]
    assert d.cls == "person"
    assert d.confidence == pytest.approx(0.9)
    assert d.keypoints_uv.shape == (17, 3)
    # foot = midpoint of the two visible ankles.
    assert d.foot_uv[0] == pytest.approx(320.0, abs=1e-3)
    assert d.foot_uv[1] == pytest.approx(490.0, abs=1e-3)


def test_pose_decode_foot_falls_back_to_bbox_bottom_when_ankles_hidden() -> None:
    out = _pose_head()
    _set_pose(out, 0, (320.0, 400.0, 40.0, 200.0), 0.9,
              {15: (310.0, 490.0, 0.0), 16: (330.0, 490.0, 0.0)})   # ankles below kpt_conf
    dets = decode_yolo11_pose(
        out, camera_id="cam_a", capture_ts=0.0,
        letterbox_meta=_identity_letterbox(), class_names=["person"],
        confidence_threshold=0.25, kpt_conf=0.3,
    )
    assert len(dets) == 1
    # falls back to bbox bottom-centre (320, 500).
    assert dets[0].foot_uv == pytest.approx((320.0, 500.0))


def test_pose_decode_below_threshold_dropped() -> None:
    out = _pose_head()
    _set_pose(out, 0, (100.0, 100.0, 40.0, 40.0), 0.1, {15: (90.0, 120.0, 0.9)})
    dets = decode_yolo11_pose(
        out, camera_id="cam_a", capture_ts=0.0,
        letterbox_meta=_identity_letterbox(), class_names=["person"],
        confidence_threshold=0.25,
    )
    assert dets == []


def test_end2end_seg_decode_masks_off_returns_none_masks() -> None:
    """``decode_masks=False`` skips mask assembly (mask=None) while boxes,
    classes and foot points are untouched — the Backbone-pipeline fast path
    (per-detection full-frame masks cost ~12 ms each on a 1080p feed)."""
    head = _end2end_seg_head()
    head[0, :6] = [100.0, 50.0, 300.0, 400.0, 0.9, 1]
    dets = decode_yolo11_seg(
        head, _protos(),
        camera_id="cam_a", capture_ts=0.0,
        letterbox_meta=_identity_letterbox(),
        target_hw=(640, 640), class_names=CLASS_NAMES,
        confidence_threshold=0.17, iou_threshold=0.45,
        keep_classes=None, mask_threshold=0.5,
        decode_masks=False,
    )
    assert len(dets) == 1
    assert dets[0].mask is None
    assert dets[0].cls == "forklift"
    assert dets[0].bbox_xyxy == pytest.approx((100.0, 50.0, 300.0, 400.0))
    assert dets[0].foot_uv == pytest.approx((200.0, 400.0))


def test_raw_seg_decode_masks_off_returns_none_masks() -> None:
    """Same contract on the raw anchor-grid branch (yolo11-style heads)."""
    # (4 + nc + nm, A) raw head with one strong anchor: cxcywh (200,225,200,350).
    head = np.zeros((4 + len(CLASS_NAMES) + _NM, 100), dtype=np.float32)
    head[:4, 0] = [200.0, 225.0, 200.0, 350.0]
    head[4 + 1, 0] = 0.9                       # cls 1 = forklift
    dets = decode_yolo11_seg(
        head, _protos(),
        camera_id="cam_a", capture_ts=0.0,
        letterbox_meta=_identity_letterbox(),
        target_hw=(640, 640), class_names=CLASS_NAMES,
        confidence_threshold=0.5, iou_threshold=0.45,
        keep_classes=None, mask_threshold=0.5,
        decode_masks=False,
    )
    assert dets and all(d.mask is None for d in dets)


# ---------- END-TO-END (NMS-free) detect head — YOLO26 / YOLOv10 ----------

from backbone.detection.postprocess import (  # noqa: E402
    decode_detect_end2end,
    is_end2end_detect_output,
)


def _e2e_rows(num_det: int = 300) -> np.ndarray:
    """(num_det, 6) padded end-to-end output: rows [x1,y1,x2,y2,score,cls]."""
    rows = np.zeros((num_det, 6), dtype=np.float32)
    rows[:, 5] = 0.0  # padding rows: cls 0, score 0
    return rows


def test_e2e_rows_decoded_without_nms_or_argmax() -> None:
    rows = _e2e_rows()
    rows[0] = [10, 20, 110, 220, 0.97, 1]
    rows[1] = [12, 22, 112, 222, 0.30, 0]      # overlapping but a DIFFERENT class:
    dets = decode_detect_end2end(               # E2E must NOT NMS it away
        rows, camera_id="cam", capture_ts=0.0,
        letterbox_meta=_identity_letterbox(), class_names=["empty_box", "filled_box"],
        confidence_threshold=0.25,
    )
    assert [d.cls for d in dets] == ["filled_box", "empty_box"]
    assert dets[0].bbox_xyxy == pytest.approx((10, 20, 110, 220))
    assert dets[0].confidence == pytest.approx(0.97)
    assert dets[0].foot_uv == pytest.approx((60.0, 220.0))


def test_e2e_padding_and_out_of_range_class_dropped() -> None:
    rows = _e2e_rows()
    rows[0] = [0, 0, 50, 50, 0.9, 7]           # class id outside nc=2 -> dropped
    dets = decode_detect_end2end(
        rows, camera_id="cam", capture_ts=0.0,
        letterbox_meta=_identity_letterbox(), class_names=["empty_box", "filled_box"],
    )
    assert dets == []


def test_e2e_detector_distinguished_from_raw_head_at_nc2() -> None:
    """nc == 2 makes 4+nc == 6: the E2E (N, 300, 6) output must NOT be mistaken
    for a transposed raw head (N, A, 4+nc) — the bug that produced conf-1.0
    garbage on the etagere model."""
    e2e = np.zeros((1, 300, 6), dtype=np.float32)
    e2e[0, 0] = [1, 2, 3, 4, 0.9, 1]
    assert is_end2end_detect_output(e2e)
    # raw transposed head at nc=2, 320px (2100 anchors): fractional "class" col
    raw_t = np.random.default_rng(0).random((1, 2100, 6), dtype=np.float32)
    assert not is_end2end_detect_output(raw_t)
    # raw canonical head (N, 4+nc, A) never matches
    raw = np.zeros((1, 6, 2100), dtype=np.float32)
    assert not is_end2end_detect_output(raw)
