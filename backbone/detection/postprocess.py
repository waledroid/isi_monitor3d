"""YOLO11 detector postprocess: raw ONNX output → ``Detection`` list.

YOLO11 detect mode emits a tensor of shape ``(batch, 4 + nc, num_anchors)``
where ``num_anchors`` is the union of stride-{8, 16, 32} grid anchors
(typically 8400 for 640x640 input). Each anchor i carries:

    output[b, 0:4, i]      = (cx, cy, w, h)   in target-frame pixels
    output[b, 4:4+nc, i]   = per-class confidence scores in [0, 1]

This module decodes that into ``Detection`` objects with bbox in
**source-frame** pixels and ``foot_uv`` at the bottom-centre of the bbox —
the exact point the homography layer needs.
"""

from __future__ import annotations

import cv2
import numpy as np

from backbone.core.types import Detection

from .preprocess import LetterboxResult, invert_letterbox_xyxy

END2END_MAX_DET = 300  # Ultralytics end-to-end exports cap num_det at max_det=300


def is_end2end_detect_output(raw: np.ndarray) -> bool:
    """True when a batched detect output is a YOLO26/YOLOv10 END-TO-END head.

    End-to-end (NMS-free) exports emit ``(N, num_det, 6)`` rows of
    ``[x1, y1, x2, y2, score, class_id]``. The raw anchor-grid head is
    ``(N, 4+nc, A)`` (or transposed). The two are ambiguous ONLY when nc == 2
    (4 + nc == 6), so decide on structure, not just width: an E2E head has at
    most END2END_MAX_DET rows and an integral class-id column, whereas a raw
    head at any usable imgsz has hundreds to thousands of anchors and a
    fractional score column.
    """
    if raw.ndim != 3 or raw.shape[2] != 6 or raw.shape[1] > END2END_MAX_DET:
        return False
    cls_col = raw[..., 5]
    return bool(np.all(np.isfinite(cls_col)) and np.allclose(cls_col, np.round(cls_col)))


def decode_detect_end2end(
    rows: np.ndarray,
    *,
    camera_id: str,
    capture_ts: float,
    letterbox_meta: LetterboxResult,
    class_names: list[str],
    confidence_threshold: float = 0.25,
    keep_classes: list[str] | None = None,
) -> list[Detection]:
    """Decode one image's END-TO-END (NMS-free) YOLO detect output.

    ``rows`` is ``(num_det, 6)`` — ``[x1, y1, x2, y2, score, class_id]`` in
    letterbox-target pixels, already NMS-filtered by the model and padded with
    low-score rows. No transpose, no argmax, no NMS: gate on score/class, invert
    the letterbox, build ``Detection`` objects (same shape as
    :func:`decode_yolo11_detect`).
    """
    if rows.ndim != 2 or rows.shape[1] != 6:
        raise ValueError(f"expected (num_det, 6) end-to-end output, got shape {rows.shape}")
    nc = len(class_names)
    xyxy_target = rows[:, :4].astype(np.float32)
    scores = rows[:, 4].astype(np.float32)
    cls_idx = rows[:, 5].astype(np.int64)
    keep = scores >= confidence_threshold
    keep &= (cls_idx >= 0) & (cls_idx < nc)
    if keep_classes is not None:
        keep_set = {class_names.index(c) for c in keep_classes if c in class_names}
        if not keep_set:
            return []
        keep &= np.isin(cls_idx, list(keep_set))
    if not keep.any():
        return []
    xyxy_source = invert_letterbox_xyxy(xyxy_target[keep], letterbox_meta)
    scores, cls_idx = scores[keep], cls_idx[keep]

    detections: list[Detection] = []
    for i in range(xyxy_source.shape[0]):
        x1, y1, x2, y2 = (float(v) for v in xyxy_source[i])
        detections.append(
            Detection(
                camera_id=camera_id,
                capture_ts=capture_ts,
                cls=class_names[int(cls_idx[i])],
                confidence=float(scores[i]),
                bbox_xyxy=(x1, y1, x2, y2),
                foot_uv=((x1 + x2) / 2.0, y2),
                keypoints_uv=None,
            )
        )
    detections.sort(key=lambda d: d.confidence, reverse=True)
    return detections


def decode_yolo11_detect(
    raw_output: np.ndarray,
    *,
    camera_id: str,
    capture_ts: float,
    letterbox_meta: LetterboxResult,
    class_names: list[str],
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    keep_classes: list[str] | None = None,
) -> list[Detection]:
    """Decode one image's YOLO11 detect output into ``Detection`` objects.

    Args:
        raw_output: ``(4 + nc, num_anchors)`` float32 — one image's slice of
            the batched ONNX output. Caller indexes ``onnx_outputs[0][i]``.
        camera_id, capture_ts: propagated into the emitted ``Detection``.
        letterbox_meta: the inverse-transform metadata so we can map boxes
            back to source-frame pixels.
        class_names: list of class names indexed by model output channel.
            ``len(class_names) == nc``.
        confidence_threshold: anchors with max class score below this are
            dropped before NMS.
        iou_threshold: NMS IoU threshold (class-agnostic).
        keep_classes: optional whitelist of class names; anchors whose class
            is not in this set are dropped. ``None`` keeps every class.

    Returns:
        List of ``Detection`` objects in source-frame coordinates, sorted by
        descending confidence.
    """
    if raw_output.ndim != 2:
        raise ValueError(f"expected (C, A) output, got shape {raw_output.shape}")
    nc = len(class_names)
    expected_channels = 4 + nc
    if raw_output.shape[0] != expected_channels:
        raise ValueError(
            f"YOLO output has {raw_output.shape[0]} channels; "
            f"expected {expected_channels} for nc={nc} (4 bbox + nc class scores). "
            f"Did class_names match the ONNX model?"
        )

    # (4 + nc, A) -> (A, 4 + nc)
    pred = raw_output.transpose(1, 0)
    bbox_cxcywh = pred[:, :4]
    class_scores = pred[:, 4:]

    # One score + one label per anchor (max-class).
    class_idx = class_scores.argmax(axis=1)
    class_conf = class_scores.max(axis=1)

    # First filter — confidence threshold + (optional) class whitelist.
    keep_mask = class_conf >= confidence_threshold
    if keep_classes is not None:
        keep_set = {class_names.index(c) for c in keep_classes if c in class_names}
        if not keep_set:
            return []
        keep_mask &= np.isin(class_idx, list(keep_set))

    if not keep_mask.any():
        return []

    bbox_cxcywh = bbox_cxcywh[keep_mask]
    class_idx = class_idx[keep_mask]
    class_conf = class_conf[keep_mask]

    # cxcywh -> xyxy (still in 640-target space).
    xyxy_target = np.empty_like(bbox_cxcywh)
    half_w = bbox_cxcywh[:, 2] / 2.0
    half_h = bbox_cxcywh[:, 3] / 2.0
    xyxy_target[:, 0] = bbox_cxcywh[:, 0] - half_w
    xyxy_target[:, 1] = bbox_cxcywh[:, 1] - half_h
    xyxy_target[:, 2] = bbox_cxcywh[:, 0] + half_w
    xyxy_target[:, 3] = bbox_cxcywh[:, 1] + half_h

    # Class-agnostic NMS via OpenCV (fast, works without torch). NMSBoxes
    # expects xywh in input-pixel space; we use target-frame xywh.
    xywh_for_nms = np.empty_like(bbox_cxcywh)
    xywh_for_nms[:, 0] = xyxy_target[:, 0]
    xywh_for_nms[:, 1] = xyxy_target[:, 1]
    xywh_for_nms[:, 2] = bbox_cxcywh[:, 2]
    xywh_for_nms[:, 3] = bbox_cxcywh[:, 3]

    nms_indices = cv2.dnn.NMSBoxes(
        xywh_for_nms.tolist(),
        class_conf.astype(np.float32).tolist(),
        score_threshold=confidence_threshold,
        nms_threshold=iou_threshold,
    )
    if len(nms_indices) == 0:
        return []
    nms_indices = np.asarray(nms_indices).reshape(-1)

    xyxy_target = xyxy_target[nms_indices]
    class_idx = class_idx[nms_indices]
    class_conf = class_conf[nms_indices]

    # Map back to source-frame pixels (undo letterbox).
    xyxy_source = invert_letterbox_xyxy(xyxy_target, letterbox_meta)

    detections: list[Detection] = []
    for i in range(xyxy_source.shape[0]):
        x1, y1, x2, y2 = (float(v) for v in xyxy_source[i])
        cls = class_names[int(class_idx[i])]
        conf = float(class_conf[i])
        # Foot point: bottom-centre of the bbox. This is what the homography
        # layer's foot projector consumes — see backbone/shared/geometry.py.
        foot_u = (x1 + x2) / 2.0
        foot_v = y2
        detections.append(
            Detection(
                camera_id=camera_id,
                capture_ts=capture_ts,
                cls=cls,
                confidence=conf,
                bbox_xyxy=(x1, y1, x2, y2),
                foot_uv=(foot_u, foot_v),
                keypoints_uv=None,
            )
        )

    # Sort by confidence descending — handy for consumers that want the
    # most-likely-correct detections first (e.g. visualizers).
    detections.sort(key=lambda d: d.confidence, reverse=True)
    return detections


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Stable sigmoid (clamp extreme logits to avoid overflow)."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def decode_yolo11_seg(
    head: np.ndarray,
    protos: np.ndarray,
    *,
    camera_id: str,
    capture_ts: float,
    letterbox_meta: LetterboxResult,
    target_hw: tuple[int, int],
    class_names: list[str],
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    keep_classes: list[str] | None = None,
    mask_threshold: float = 0.5,
    decode_masks: bool = True,
) -> list[Detection]:
    """Decode one image's YOLO11-seg outputs into ``Detection`` objects (with
    ``mask`` populated — full-frame HxW bool ndarray in source coords).

    ``decode_masks=False`` skips the mask assembly entirely (``mask=None`` on
    every Detection) — for consumers that only need boxes/foot points (the
    Backbone pipeline), where per-detection full-frame masks are pure CPU cost.

    Args:
        head: ``(4 + nc + nm, A)`` float32 — per-anchor bbox + class scores +
            mask coefficients. Caller passes the per-image slice (e.g.
            ``outputs[0][i]``).
        protos: ``(nm, mh, mw)`` float32 — mask prototype maps for this image
            (caller passes the per-image slice of the second ONNX output).
        target_hw: the letterbox target size used at preprocess (typically
            ``(640, 640)``). Used to scale bbox → proto-space coords.
        Other args mirror :func:`decode_yolo11_detect`.

    Returns:
        ``Detection`` objects (sorted by descending confidence) whose ``mask``
        field is a full-frame ``bool`` ndarray of shape ``source_shape_hw`` —
        ``True`` inside the predicted instance, ``False`` elsewhere.
    """
    if head.ndim != 2:
        raise ValueError(f"expected (C, A) head output, got shape {head.shape}")
    if protos.ndim != 3:
        raise ValueError(f"expected (nm, mh, mw) protos, got shape {protos.shape}")
    nc = len(class_names)
    nm = protos.shape[0]
    # YOLO26 / YOLOv10-style END-TO-END (NMS-free) seg head: already-decoded
    # detections shaped (num_det, 6 + nm) — rows of [x1,y1,x2,y2, score, class_id,
    # *mask_coeffs]. This is a different layout from yolo11's raw anchor-grid head
    # (4+nc+nm, A); dispatch on the row width (6+nm) so the raw path below is left
    # completely unchanged. (Raw anchors number in the thousands, never 6+nm.)
    if head.shape[1] == 6 + nm:
        return _decode_seg_end2end(
            head, protos,
            camera_id=camera_id, capture_ts=capture_ts,
            letterbox_meta=letterbox_meta, target_hw=target_hw,
            class_names=class_names, confidence_threshold=confidence_threshold,
            keep_classes=keep_classes, mask_threshold=mask_threshold,
            decode_masks=decode_masks,
        )
    expected_channels = 4 + nc + nm
    if head.shape[0] != expected_channels:
        raise ValueError(
            f"YOLO-seg head has {head.shape[0]} channels; expected {expected_channels} "
            f"for nc={nc} + nm={nm} (4 bbox + nc class + nm mask coeffs). "
            f"Did class_names match the ONNX model? (YOLO26/YOLOv10 end-to-end heads "
            f"are (num_det, 6+nm) and use the end-to-end branch — this means neither matched.)"
        )

    pred = head.transpose(1, 0)        # (A, 4 + nc + nm)
    bbox_cxcywh = pred[:, :4]
    class_scores = pred[:, 4:4 + nc]
    coeff_all = pred[:, 4 + nc:4 + nc + nm]

    class_idx = class_scores.argmax(axis=1)
    class_conf = class_scores.max(axis=1)

    keep_mask = class_conf >= confidence_threshold
    if keep_classes is not None:
        keep_set = {class_names.index(c) for c in keep_classes if c in class_names}
        if not keep_set:
            return []
        keep_mask &= np.isin(class_idx, list(keep_set))
    if not keep_mask.any():
        return []

    bbox_cxcywh = bbox_cxcywh[keep_mask]
    class_idx = class_idx[keep_mask]
    class_conf = class_conf[keep_mask]
    coeff_all = coeff_all[keep_mask]

    # cxcywh → xyxy (target-frame).
    xyxy_target = np.empty_like(bbox_cxcywh)
    half_w = bbox_cxcywh[:, 2] / 2.0
    half_h = bbox_cxcywh[:, 3] / 2.0
    xyxy_target[:, 0] = bbox_cxcywh[:, 0] - half_w
    xyxy_target[:, 1] = bbox_cxcywh[:, 1] - half_h
    xyxy_target[:, 2] = bbox_cxcywh[:, 0] + half_w
    xyxy_target[:, 3] = bbox_cxcywh[:, 1] + half_h

    xywh_for_nms = np.empty_like(bbox_cxcywh)
    xywh_for_nms[:, 0] = xyxy_target[:, 0]
    xywh_for_nms[:, 1] = xyxy_target[:, 1]
    xywh_for_nms[:, 2] = bbox_cxcywh[:, 2]
    xywh_for_nms[:, 3] = bbox_cxcywh[:, 3]
    nms_indices = cv2.dnn.NMSBoxes(
        xywh_for_nms.tolist(),
        class_conf.astype(np.float32).tolist(),
        score_threshold=confidence_threshold,
        nms_threshold=iou_threshold,
    )
    if len(nms_indices) == 0:
        return []
    nms_indices = np.asarray(nms_indices).reshape(-1)

    xyxy_target = xyxy_target[nms_indices]
    class_idx = class_idx[nms_indices]
    class_conf = class_conf[nms_indices]
    coeff = coeff_all[nms_indices]   # (M, nm)
    xyxy_source = invert_letterbox_xyxy(xyxy_target, letterbox_meta)

    # ---- mask decode ----
    # masks_proto[i] = sigmoid( coeff[i] @ protos_flat ), reshaped to (mh, mw)
    target_h, target_w = target_hw
    src_h, src_w = letterbox_meta.source_shape_hw
    if decode_masks:
        nm_, mh, mw = protos.shape
        assert nm_ == nm
        protos_flat = protos.reshape(nm, mh * mw)             # (nm, mh*mw)
        masks_proto = _sigmoid(coeff @ protos_flat)           # (M, mh*mw)
        masks_proto = masks_proto.reshape(-1, mh, mw)         # (M, mh, mw)
        sx = mw / float(target_w)
        sy = mh / float(target_h)

    detections: list[Detection] = []
    for i in range(xyxy_source.shape[0]):
        mask_full = None
        if decode_masks:
            # Crop the proto-space mask to the bbox (in proto coords).
            x1_t, y1_t, x2_t, y2_t = xyxy_target[i]
            px1 = max(0, min(int(np.floor(x1_t * sx)), mw - 1))
            py1 = max(0, min(int(np.floor(y1_t * sy)), mh - 1))
            px2 = max(px1 + 1, min(int(np.ceil(x2_t * sx)),  mw))
            py2 = max(py1 + 1, min(int(np.ceil(y2_t * sy)),  mh))
            mp = masks_proto[i, py1:py2, px1:px2]

        # Source-frame bbox (clipped).
        x1_s, y1_s, x2_s, y2_s = xyxy_source[i]
        x1i = max(0, min(round(x1_s), src_w - 1))
        y1i = max(0, min(round(y1_s), src_h - 1))
        x2i = max(x1i + 1, min(round(x2_s), src_w))
        y2i = max(y1i + 1, min(round(y2_s), src_h))
        bw, bh = x2i - x1i, y2i - y1i

        # Resize the small proto crop to source bbox size + threshold.
        if decode_masks:
            if mp.size == 0 or bw <= 0 or bh <= 0:
                mask_full = np.zeros((src_h, src_w), dtype=bool)
            else:
                mp_resized = cv2.resize(mp.astype(np.float32), (bw, bh),
                                        interpolation=cv2.INTER_LINEAR)
                mask_full = np.zeros((src_h, src_w), dtype=bool)
                mask_full[y1i:y2i, x1i:x2i] = mp_resized > mask_threshold

        cls = class_names[int(class_idx[i])]
        conf = float(class_conf[i])
        foot_u = (x1_s + x2_s) / 2.0
        foot_v = float(y2_s)
        detections.append(Detection(
            camera_id=camera_id,
            capture_ts=capture_ts,
            cls=cls,
            confidence=conf,
            bbox_xyxy=(float(x1_s), float(y1_s), float(x2_s), float(y2_s)),
            foot_uv=(foot_u, foot_v),
            keypoints_uv=None,
            mask=mask_full,
        ))

    detections.sort(key=lambda d: d.confidence, reverse=True)
    return detections


def _decode_seg_end2end(
    head: np.ndarray,
    protos: np.ndarray,
    *,
    camera_id: str,
    capture_ts: float,
    letterbox_meta: LetterboxResult,
    target_hw: tuple[int, int],
    class_names: list[str],
    confidence_threshold: float = 0.25,
    keep_classes: list[str] | None = None,
    mask_threshold: float = 0.5,
    decode_masks: bool = True,
) -> list[Detection]:
    """Decode an END-TO-END / NMS-free YOLO-seg head (YOLO26 / YOLOv10-style).

    Unlike the raw anchor-grid head, this output is already NMS-filtered and
    decoded: ``head`` is ``(num_det, 6 + nm)`` where each row is
    ``[x1, y1, x2, y2, score, class_id, *mask_coeffs]`` in letterbox-target pixel
    coords, sorted by descending score and padded with low-score rows. So there is
    no transpose, no per-class argmax (the class id is explicit), and **no NMS** —
    only confidence/class filtering, letterbox inversion, and the same mask
    assembly as :func:`decode_yolo11_seg`. Padding rows fall out via the gate.
    """
    nc = len(class_names)
    nm = protos.shape[0]
    xyxy_target = head[:, :4].astype(np.float32)
    scores = head[:, 4].astype(np.float32)
    cls_idx = head[:, 5].astype(np.int64)
    coeff_all = head[:, 6:6 + nm].astype(np.float32)

    keep = scores >= confidence_threshold
    keep &= (cls_idx >= 0) & (cls_idx < nc)         # drop padding / out-of-range rows
    if keep_classes is not None:
        keep_set = {class_names.index(c) for c in keep_classes if c in class_names}
        if not keep_set:
            return []
        keep &= np.isin(cls_idx, list(keep_set))
    if not keep.any():
        return []

    xyxy_target = xyxy_target[keep]
    scores = scores[keep]
    cls_idx = cls_idx[keep]
    coeff = coeff_all[keep]
    xyxy_source = invert_letterbox_xyxy(xyxy_target, letterbox_meta)

    # ---- mask decode (mirrors decode_yolo11_seg's mask assembly) ----
    target_h, target_w = target_hw
    src_h, src_w = letterbox_meta.source_shape_hw
    if decode_masks:
        _nm, mh, mw = protos.shape
        protos_flat = protos.reshape(nm, mh * mw)
        masks_proto = _sigmoid(coeff @ protos_flat).reshape(-1, mh, mw)
        sx = mw / float(target_w)
        sy = mh / float(target_h)

    detections: list[Detection] = []
    for i in range(xyxy_source.shape[0]):
        mask_full = None
        x1_s, y1_s, x2_s, y2_s = xyxy_source[i]
        if decode_masks:
            x1_t, y1_t, x2_t, y2_t = xyxy_target[i]
            px1 = max(0, min(int(np.floor(x1_t * sx)), mw - 1))
            py1 = max(0, min(int(np.floor(y1_t * sy)), mh - 1))
            px2 = max(px1 + 1, min(int(np.ceil(x2_t * sx)), mw))
            py2 = max(py1 + 1, min(int(np.ceil(y2_t * sy)), mh))
            mp = masks_proto[i, py1:py2, px1:px2]

            x1i = max(0, min(round(x1_s), src_w - 1))
            y1i = max(0, min(round(y1_s), src_h - 1))
            x2i = max(x1i + 1, min(round(x2_s), src_w))
            y2i = max(y1i + 1, min(round(y2_s), src_h))
            bw, bh = x2i - x1i, y2i - y1i

            if mp.size == 0 or bw <= 0 or bh <= 0:
                mask_full = np.zeros((src_h, src_w), dtype=bool)
            else:
                mp_resized = cv2.resize(mp.astype(np.float32), (bw, bh),
                                        interpolation=cv2.INTER_LINEAR)
                mask_full = np.zeros((src_h, src_w), dtype=bool)
                mask_full[y1i:y2i, x1i:x2i] = mp_resized > mask_threshold

        cls = class_names[int(cls_idx[i])]
        conf = float(scores[i])
        foot_u = (x1_s + x2_s) / 2.0
        foot_v = float(y2_s)
        detections.append(Detection(
            camera_id=camera_id,
            capture_ts=capture_ts,
            cls=cls,
            confidence=conf,
            bbox_xyxy=(float(x1_s), float(y1_s), float(x2_s), float(y2_s)),
            foot_uv=(foot_u, foot_v),
            keypoints_uv=None,
            mask=mask_full,
        ))

    detections.sort(key=lambda d: d.confidence, reverse=True)
    return detections


def decode_rfdetr_seg(
    dets: np.ndarray,
    labels: np.ndarray,
    masks: np.ndarray | None,
    *,
    camera_id: str,
    capture_ts: float,
    source_wh: tuple[int, int],
    class_names: list[str],
    confidence_threshold: float = 0.3,
    mask_threshold: float = 0.5,
    num_select: int = 300,
) -> list[Detection]:
    """Decode one image's RF-DETR-seg ONNX outputs into ``Detection`` objects.

    RF-DETR is a DETR-style, NMS-free detector with a fixed set of object queries
    (200 for this model). Unlike YOLO it uses **stretch-resize** preprocessing
    (no letterbox / no pad), so normalised boxes map straight to source pixels by
    multiplying x/w by ``orig_w`` and y/h by ``orig_h``. This faithfully mirrors
    ``trainer/isidet/src/inference/onnx_inferencer.py::_postprocess_rfdetr``.

    Args:
        dets: ``(num_queries, 4)`` float32 — per-query box as **cxcywh, normalised
            [0, 1]** (DETR convention). Caller passes the per-image slice
            ``outputs['dets'][0]``.
        labels: ``(num_queries, head_nc)`` float32 — per-query class **logits**.
            RF-DETR is sigmoid/focal (not softmax). The head follows COCO: column
            0 is background/no-object; columns ``1..len(class_names)`` are the
            fine-tuned classes, i.e. ``class_names[i]`` lives at column ``i + 1``.
            Columns outside the trained set are ignored.
        masks: ``(num_queries, mh, mw)`` float32 mask **logits**, or ``None`` for a
            detect-only RF-DETR. When present, each kept query's mask is
            ``sigmoid → threshold → resize to its source bbox → placed into a
            full-frame bool array``.
        source_wh: ``(orig_w, orig_h)`` of the source frame the boxes/masks map to.
        class_names: trained class names; ``class_names[i]`` ↔ logits column
            ``i + 1``. Defaults in the plugin to ``["palette", "carton", "polybag"]``.
        confidence_threshold: queries scoring below this (post-sigmoid) are dropped.
        mask_threshold: per-pixel sigmoid threshold for the bool mask.
        num_select: topk-over-(queries x classes) budget — matches the native
            RF-DETR selection (cap 300); only here to bound the sort, the
            confidence gate is what actually filters.

    Returns:
        ``Detection`` objects in source-frame coords, sorted by descending
        confidence. Robust to zero kept queries (returns ``[]``).
    """
    if dets.ndim != 2 or dets.shape[1] != 4:
        raise ValueError(f"expected (num_queries, 4) dets, got shape {dets.shape}")
    if labels.ndim != 2:
        raise ValueError(f"expected (num_queries, head_nc) labels, got shape {labels.shape}")
    if dets.shape[0] != labels.shape[0]:
        raise ValueError(
            f"dets has {dets.shape[0]} queries but labels has {labels.shape[0]}"
        )

    orig_w, orig_h = source_wh
    head_nc = labels.shape[1]
    # Skip the background column (index 0); take the next `nc` trained classes.
    # After the slice, view column i == class_names[i].
    effective_nc = min(len(class_names), head_nc - 1)
    if effective_nc <= 0:
        return []
    logits = labels[:, 1:1 + effective_nc]

    # Native RF-DETR topk-over-(queries*classes), sigmoid/focal scoring.
    probs = _sigmoid(logits)                                  # (Q, effective_nc)
    flat = probs.reshape(-1)
    k = min(num_select, flat.size)
    topk_idx = np.argpartition(-flat, k - 1)[:k]
    topk_idx = topk_idx[np.argsort(-flat[topk_idx])]          # sort by score desc
    scores = flat[topk_idx]
    query_idx = topk_idx // effective_nc
    class_idx = (topk_idx % effective_nc).astype(int)         # 0-based into class_names

    keep = scores > confidence_threshold
    if not np.any(keep):
        return []
    scores = scores[keep]
    query_idx = query_idx[keep]
    class_idx = class_idx[keep]

    # cxcywh-norm -> xyxy, then stretch-map to source pixels (cx,w * orig_w;
    # cy,h * orig_h) -- RF-DETR's target_sizes = (orig_w, orig_h) convention.
    chosen = dets[query_idx]
    cx, cy, w, h = chosen[:, 0], chosen[:, 1], chosen[:, 2], chosen[:, 3]
    xyxy = np.column_stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    xyxy[:, [0, 2]] *= orig_w
    xyxy[:, [1, 3]] *= orig_h
    xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, orig_w)
    xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, orig_h)

    # ---- mask assembly (sigmoid → resize to box → full-frame bool) ----
    masks_kept = None
    if masks is not None:
        if masks.ndim != 3:
            raise ValueError(f"expected (num_queries, mh, mw) masks, got shape {masks.shape}")
        masks_kept = _sigmoid(masks[query_idx])               # (K, mh, mw)
        mh, mw = masks_kept.shape[1], masks_kept.shape[2]
        sx = mw / float(orig_w)                               # source → mask scale
        sy = mh / float(orig_h)

    detections: list[Detection] = []
    for i in range(xyxy.shape[0]):
        x1_s, y1_s, x2_s, y2_s = (float(v) for v in xyxy[i])
        cls = class_names[int(class_idx[i])]
        conf = float(scores[i])
        foot_u = (x1_s + x2_s) / 2.0
        foot_v = y2_s

        mask_full = None
        if masks_kept is not None:
            x1i = max(0, min(round(x1_s), orig_w - 1))
            y1i = max(0, min(round(y1_s), orig_h - 1))
            x2i = max(x1i + 1, min(round(x2_s), orig_w))
            y2i = max(y1i + 1, min(round(y2_s), orig_h))
            bw, bh = x2i - x1i, y2i - y1i

            px1 = max(0, min(int(np.floor(x1_s * sx)), mw - 1))
            py1 = max(0, min(int(np.floor(y1_s * sy)), mh - 1))
            px2 = max(px1 + 1, min(int(np.ceil(x2_s * sx)), mw))
            py2 = max(py1 + 1, min(int(np.ceil(y2_s * sy)), mh))
            mp = masks_kept[i, py1:py2, px1:px2]

            mask_full = np.zeros((orig_h, orig_w), dtype=bool)
            if mp.size and bw > 0 and bh > 0:
                mp_resized = cv2.resize(mp.astype(np.float32), (bw, bh),
                                        interpolation=cv2.INTER_LINEAR)
                mask_full[y1i:y2i, x1i:x2i] = mp_resized > mask_threshold

        detections.append(Detection(
            camera_id=camera_id,
            capture_ts=capture_ts,
            cls=cls,
            confidence=conf,
            bbox_xyxy=(x1_s, y1_s, x2_s, y2_s),
            foot_uv=(foot_u, foot_v),
            keypoints_uv=None,
            mask=mask_full,
        ))

    detections.sort(key=lambda d: d.confidence, reverse=True)
    return detections


def _ankle_foot(kpts, box, kpt_conf, left_ankle, right_ankle):
    """Foot point = midpoint of the visible ankle keypoint(s); falls back to the
    bbox bottom-centre. This is the only floor-contact (Z≈0) point of a person,
    so it's the one the homography may validly project to metres."""
    vis = [kpts[i] for i in (left_ankle, right_ankle)
           if i < len(kpts) and kpts[i, 2] >= kpt_conf]
    if vis:
        return (float(np.mean([p[0] for p in vis])), float(np.mean([p[1] for p in vis])))
    x1, _y1, x2, y2 = box
    return ((x1 + x2) / 2.0, float(y2))


def decode_yolo11_pose(
    raw_output: np.ndarray,
    *,
    camera_id: str,
    capture_ts: float,
    letterbox_meta: LetterboxResult,
    class_names: list[str],
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    kpt_conf: float = 0.3,
    left_ankle: int = 15,
    right_ankle: int = 16,
) -> list[Detection]:
    """Decode one image's YOLO11-pose head ``(4 + nc + K*3, A)`` into person
    ``Detection`` objects with ``keypoints_uv`` (K, 3) and ``foot_uv`` at the
    ankle midpoint (the floor-contact point the homography consumes).

    Cols: ``0:4`` cxcywh (target px), ``4:4+nc`` class scores, ``4+nc:`` keypoints
    flattened as ``[x, y, conf] * K`` (x,y in target px). Box + keypoint xy are
    inverted back to source-frame pixels."""
    if raw_output.ndim != 2:
        raise ValueError(f"expected (C, A) pose head, got shape {raw_output.shape}")
    nc = len(class_names)
    c = raw_output.shape[0]
    k = (c - 4 - nc) // 3
    if k <= 0 or 4 + nc + k * 3 != c:
        raise ValueError(
            f"YOLO-pose head has {c} channels; expected 4 + nc({nc}) + K*3 "
            f"(4 bbox + nc class + K keypoints x3). Did class_names match the model?"
        )

    pred = raw_output.transpose(1, 0)                  # (A, C)
    bbox_cxcywh = pred[:, :4]
    class_scores = pred[:, 4:4 + nc]
    kpts = pred[:, 4 + nc:].reshape(-1, k, 3)          # (A, K, 3) target-frame x,y,conf

    class_idx = class_scores.argmax(axis=1)
    class_conf = class_scores.max(axis=1)
    keep = class_conf >= confidence_threshold
    if not keep.any():
        return []
    bbox_cxcywh = bbox_cxcywh[keep]
    class_idx = class_idx[keep]
    class_conf = class_conf[keep]
    kpts = kpts[keep]

    xyxy_target = np.empty_like(bbox_cxcywh)
    half_w = bbox_cxcywh[:, 2] / 2.0
    half_h = bbox_cxcywh[:, 3] / 2.0
    xyxy_target[:, 0] = bbox_cxcywh[:, 0] - half_w
    xyxy_target[:, 1] = bbox_cxcywh[:, 1] - half_h
    xyxy_target[:, 2] = bbox_cxcywh[:, 0] + half_w
    xyxy_target[:, 3] = bbox_cxcywh[:, 1] + half_h

    xywh = np.column_stack([xyxy_target[:, 0], xyxy_target[:, 1],
                            bbox_cxcywh[:, 2], bbox_cxcywh[:, 3]])
    nms_indices = cv2.dnn.NMSBoxes(
        xywh.tolist(), class_conf.astype(np.float32).tolist(),
        score_threshold=confidence_threshold, nms_threshold=iou_threshold,
    )
    if len(nms_indices) == 0:
        return []
    nms_indices = np.asarray(nms_indices).reshape(-1)
    xyxy_target = xyxy_target[nms_indices]
    class_idx = class_idx[nms_indices]
    class_conf = class_conf[nms_indices]
    kpts = kpts[nms_indices]

    xyxy_source = invert_letterbox_xyxy(xyxy_target, letterbox_meta)
    # Undo the letterbox on keypoint (x, y); keep their confidence as-is.
    scale = letterbox_meta.scale
    pad_x, pad_y = letterbox_meta.pad_xy
    kpts_src = kpts.astype(np.float64).copy()
    kpts_src[:, :, 0] = (kpts[:, :, 0] - pad_x) / scale
    kpts_src[:, :, 1] = (kpts[:, :, 1] - pad_y) / scale

    detections: list[Detection] = []
    for i in range(xyxy_source.shape[0]):
        x1, y1, x2, y2 = (float(v) for v in xyxy_source[i])
        kp = kpts_src[i]
        foot = _ankle_foot(kp, (x1, y1, x2, y2), kpt_conf, left_ankle, right_ankle)
        detections.append(Detection(
            camera_id=camera_id, capture_ts=capture_ts,
            cls=class_names[int(class_idx[i])], confidence=float(class_conf[i]),
            bbox_xyxy=(x1, y1, x2, y2), foot_uv=foot, keypoints_uv=kp,
        ))
    detections.sort(key=lambda d: d.confidence, reverse=True)
    return detections
