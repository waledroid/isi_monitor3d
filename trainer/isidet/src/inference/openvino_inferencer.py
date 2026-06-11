"""
OpenVINO Inference Engine — optimized for CPU (Intel) deployment.
Loads models exported via export_engine.py (.xml + .bin format).
"""

import cv2
import numpy as np
import supervision as sv
from pathlib import Path
from typing import List
import logging

logger = logging.getLogger(__name__)

try:
    import openvino as ov
except ImportError:
    raise ImportError("openvino not installed. pip install openvino")

from src.inference.base_inferencer import BaseInferencer


class OpenVINOInferencer(BaseInferencer):
    """OpenVINO inference engine — runs on CPU (or Intel iGPU if available)."""

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.5,
        device: str = None,
        imgsz: int = None,
    ):
        super().__init__(model_path, conf_threshold, device, imgsz)

        xml_path = Path(model_path)
        if not xml_path.exists():
            raise FileNotFoundError(f"OpenVINO model not found: {xml_path}")

        core = ov.Core()
        model = core.read_model(str(xml_path))

        # Determine device: "CPU", "GPU" (Intel iGPU), or "AUTO"
        ov_device = "CPU"
        if self.device and self.device.upper() in ("GPU", "AUTO"):
            available = core.available_devices
            if "GPU" in available:
                ov_device = "GPU"
            else:
                logger.info(f"Intel GPU not available, using CPU. Available: {available}")

        self.compiled = core.compile_model(model, ov_device)
        self.infer_request = self.compiled.create_infer_request()

        # Parse model IO
        self.input_layer = self.compiled.input(0)
        input_shape = self.input_layer.shape  # [1, 3, H, W]
        self.model_h = input_shape[2]
        self.model_w = input_shape[3]

        # Detect model type from output names/shapes. Store the output_names
        # list so RF-DETR postprocess can pick by name rather than position
        # (different RF-DETR exporters emit pred_boxes/pred_logits/pred_masks
        # vs dets/labels/masks).
        self.output_names = [self.compiled.output(i).get_any_name()
                             for i in range(len(self.compiled.outputs))]
        self.is_rfdetr = any(n in self.output_names
                              for n in ('dets', 'pred_logits', 'bboxes', 'labels'))

        # Hard-refuse RF-DETR OpenVINO IR. OpenVINO 2026 mistranslates the
        # transformer / Einsum ops in RF-DETR's segmentation head, yielding
        # logits that diverge from ONNX by |Δ| up to ~9 — the model loads
        # happily but returns zero detections at inference. Raise now with
        # a clear remediation, rather than letting the operator stare at a
        # working-looking dashboard that never counts anything. Direct the
        # user to the two working RF-DETR paths: ONNX (CPU or CUDA) and
        # native .pth (GPU only). See OPENVINO_TUTORIAL.md §11 for the full
        # investigation.
        if self.is_rfdetr:
            raise ValueError(
                "RF-DETR is not supported via OpenVINO — the converted IR "
                "produces wrong logits on OpenVINO 2026 (known transformer/"
                "Einsum conversion limitation). Switch to an RF-DETR .onnx "
                "or .pth weight instead (ONNX CPU works correctly; use GPU "
                f"for real-time). Rejected model: {xml_path}"
            )

        self._has_mask_output = len(self.compiled.outputs) > 1
        self._num_outputs = len(self.compiled.outputs)

        # Number of classes. RF-DETR head emits 91 logits (COCO convention)
        # but the app only uses the first `nc` fine-tuned classes — we keep
        # self.nc from BaseInferencer (read from configs/train.yaml) and
        # let the postprocess slice [1:1+nc] out of the head. For YOLO,
        # post-NMS exports give no per-class column (only an argmaxed id),
        # so we also rely on the config value. The block below only kicks in
        # for pre-NMS YOLO exports that DO have the per-class score columns.
        if not self.is_rfdetr:
            out_shape = self.compiled.output(0).shape
            if len(out_shape) == 3 and max(out_shape[1], out_shape[2]) > 1000:
                feature_dim = out_shape[2] if out_shape[1] > out_shape[2] else out_shape[1]
                mask_coeffs = 32 if self._has_mask_output else 0
                inferred = feature_dim - 4 - mask_coeffs
                if inferred > 0:
                    self.nc = inferred

        logger.info(f"OpenVINO model loaded: {xml_path}")
        logger.info(f"  Type: {'RF-DETR' if self.is_rfdetr else 'YOLO'} | "
                     f"Input: {self.model_w}x{self.model_h} | "
                     f"Classes: {self.nc} | Device: {ov_device}")

    def _load_model(self):
        return self.compiled if hasattr(self, 'compiled') else None

    # ── Preprocessing — MUST match OptimizedONNXInferencer exactly ──────────
    # Ultralytics YOLO trains on letterboxed inputs (aspect-preserving pad).
    # Stretch-resizing at inference distorts objects the model has never
    # seen that way — causes class-flipping on non-square sources.
    # RF-DETR (DINOv2 backbone) trains on stretch resize + ImageNet norm.

    _IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    _IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    def _letterbox(self, frame: np.ndarray, pad_color: int = 114):
        """Aspect-preserving resize + pad to (model_h, model_w).

        Returns (padded_img, ratio, pad_left, pad_top) — the same tuple
        shape the ONNX inferencer stores, so mask post-processing logic
        transplants 1:1.
        """
        orig_h, orig_w = frame.shape[:2]
        r = min(self.model_h / orig_h, self.model_w / orig_w)
        new_w, new_h = int(round(orig_w * r)), int(round(orig_h * r))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_w = self.model_w - new_w
        pad_h = self.model_h - new_h
        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                     cv2.BORDER_CONSTANT, value=(pad_color,) * 3)
        return padded, r, left, top

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        # Ultralytics and RF-DETR train on RGB. OpenCV/RTSP frames are BGR.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.is_rfdetr:
            resized = cv2.resize(rgb, (self.model_w, self.model_h), interpolation=cv2.INTER_LINEAR)
            self._last_letterbox = None
            img = resized.transpose((2, 0, 1)).astype(np.float32) / 255.0
            img = (img - self._IMAGENET_MEAN) / self._IMAGENET_STD
        else:
            padded, ratio, pad_x, pad_y = self._letterbox(rgb)
            self._last_letterbox = (ratio, pad_x, pad_y)
            img = padded.transpose((2, 0, 1)).astype(np.float32) / 255.0
        return np.ascontiguousarray(img[np.newaxis, ...])

    # ── Inference ────────────────────────────────────────────────────────────

    def predict_frame(self, frame: np.ndarray) -> sv.Detections:
        input_tensor = self.preprocess(frame)

        try:
            self.infer_request.infer({0: input_tensor})
            outputs = [self.infer_request.get_output_tensor(i).data.copy()
                       for i in range(self._num_outputs)]
        except Exception as e:
            logger.error(f"OpenVINO inference failed: {e}")
            return sv.Detections.empty()

        orig_h, orig_w = frame.shape[:2]
        try:
            if self.is_rfdetr:
                return self._postprocess_rfdetr(outputs, orig_w, orig_h)
            else:
                return self._postprocess_yolo(outputs, orig_w, orig_h)
        except Exception as e:
            logger.error(f"Postprocessing failed: {e}")
            return sv.Detections.empty()

    # ── YOLO Postprocessing ──────────────────────────────────────────────────
    # Two Ultralytics ONNX layouts must be handled:
    #   post-NMS (nms=True):  [1, N<=300, 6|6+32]  = [x1,y1,x2,y2, score, class, (mask_coeffs)]
    #   raw (nms=False):      [1, A, 4+nc(+32)]    where A is anchor count (~3549 for 416)
    # Distinguish by the anchor-axis magnitude; raw exports have a dim > 1000.

    def _postprocess_yolo(self, outputs: List[np.ndarray], orig_w: int, orig_h: int) -> sv.Detections:
        raw = outputs[0][0] if outputs[0].ndim == 3 else outputs[0]
        if raw.shape[0] == 0:
            return sv.Detections.empty()

        is_raw = max(raw.shape[0], raw.shape[1]) > 1000
        if is_raw:
            preds = raw if raw.shape[0] > raw.shape[1] else raw.T
            return self._postprocess_yolo_raw(preds, outputs, orig_w, orig_h)

        # ── Post-NMS path: 300 rows, col 4 = confidence, col 5 = class_id ──
        preds = raw
        n_cols = preds.shape[1]
        has_mask_coeffs = n_cols > 6

        boxes = preds[:, :4].copy()
        confidences = preds[:, 4].astype(np.float32)
        class_ids = preds[:, 5].astype(int)
        mask_coeffs = preds[:, 6:] if has_mask_coeffs else None

        keep = confidences > self.conf_threshold
        if not np.any(keep):
            return sv.Detections.empty()

        boxes = boxes[keep]
        class_ids = class_ids[keep]
        confidences = confidences[keep]
        if mask_coeffs is not None:
            mask_coeffs = mask_coeffs[keep]

        # Invert the letterbox transform to bring boxes back into original
        # image coordinates: orig = (model - pad) / ratio
        ratio, pad_x, pad_y = getattr(self, '_last_letterbox', (1.0, 0, 0))
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / ratio
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / ratio
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

        valid = (boxes[:, 2] > boxes[:, 0] + 1) & (boxes[:, 3] > boxes[:, 1] + 1)
        if not np.all(valid):
            boxes, class_ids, confidences = boxes[valid], class_ids[valid], confidences[valid]
            if mask_coeffs is not None:
                mask_coeffs = mask_coeffs[valid]

        if len(boxes) == 0:
            return sv.Detections.empty()

        masks = None
        if mask_coeffs is not None and len(outputs) > 1:
            proto = outputs[1][0] if outputs[1].ndim == 4 else outputs[1]
            masks = self._process_masks(proto, mask_coeffs, boxes, orig_h, orig_w)

        return sv.Detections(
            xyxy=boxes, confidence=confidences,
            class_id=class_ids.astype(int), mask=masks,
        )

    def _postprocess_yolo_raw(self, preds: np.ndarray, outputs: List[np.ndarray],
                              orig_w: int, orig_h: int) -> sv.Detections:
        """Pre-NMS YOLO export — cols 4:4+nc are per-class scores, cols 4+nc: are mask coeffs."""
        boxes_cxcywh = preds[:, :4].copy()
        scores = preds[:, 4:4 + self.nc]
        mask_coeffs = preds[:, 4 + self.nc:] if preds.shape[1] > 4 + self.nc else None

        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        keep = confidences > self.conf_threshold
        if not np.any(keep):
            return sv.Detections.empty()

        boxes_cxcywh = boxes_cxcywh[keep]
        class_ids = class_ids[keep]
        confidences = confidences[keep]
        if mask_coeffs is not None:
            mask_coeffs = mask_coeffs[keep]

        cx, cy, w, h = boxes_cxcywh.T
        boxes = np.column_stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

        # Raw YOLO preds are in letterboxed model pixel space — invert to orig.
        ratio, pad_x, pad_y = getattr(self, '_last_letterbox', (1.0, 0, 0))
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / ratio
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / ratio
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

        # NMS — raw exports skip Ultralytics' embedded NMS so we run it manually.
        idx = cv2.dnn.NMSBoxes(
            boxes.tolist(), confidences.tolist(),
            self.conf_threshold, 0.45,
        )
        if len(idx) == 0:
            return sv.Detections.empty()
        idx = idx.flatten() if hasattr(idx, 'flatten') else np.asarray(idx).flatten()
        boxes, class_ids, confidences = boxes[idx], class_ids[idx], confidences[idx]
        if mask_coeffs is not None:
            mask_coeffs = mask_coeffs[idx]

        masks = None
        if mask_coeffs is not None and len(outputs) > 1:
            proto = outputs[1][0] if outputs[1].ndim == 4 else outputs[1]
            masks = self._process_masks(proto, mask_coeffs, boxes, orig_h, orig_w)

        return sv.Detections(
            xyxy=boxes, confidence=confidences,
            class_id=class_ids.astype(int), mask=masks,
        )

    def _process_masks(self, proto: np.ndarray, coeffs: np.ndarray,
                       boxes: np.ndarray, orig_h: int, orig_w: int) -> np.ndarray:
        """Decode YOLO mask coefficients — mirrors OptimizedONNXInferencer.

        Key correctness points:
          * Proto coords map from the *letterboxed* model space, not
            stretch-resized orig space — use ratio + pad, not plain ratio.
          * Only the bbox region is upsampled (cost scales with box area,
            not frame area).
          * Supervision annotators expect bool masks.
        """
        try:
            n = len(boxes)
            if n == 0:
                return np.zeros((0, orig_h, orig_w), dtype=bool)

            proto_h, proto_w = proto.shape[1], proto.shape[2]
            masks_raw = coeffs @ proto.reshape(proto.shape[0], -1)
            masks_raw = masks_raw.reshape(n, proto_h, proto_w)
            masks_raw = 1.0 / (1.0 + np.exp(-masks_raw))

            ratio, pad_x, pad_y = getattr(self, '_last_letterbox', (1.0, 0, 0))
            sx = proto_w / self.model_w
            sy = proto_h / self.model_h
            masks = np.zeros((n, orig_h, orig_w), dtype=bool)

            for i in range(n):
                x1, y1, x2, y2 = boxes[i]
                x1i = max(0, min(int(x1), orig_w - 1))
                y1i = max(0, min(int(y1), orig_h - 1))
                x2i = max(x1i + 1, min(int(x2), orig_w))
                y2i = max(y1i + 1, min(int(y2), orig_h))
                bw, bh = x2i - x1i, y2i - y1i

                # orig → letterboxed model → proto coords.
                px1 = int(np.floor((x1 * ratio + pad_x) * sx))
                py1 = int(np.floor((y1 * ratio + pad_y) * sy))
                px2 = int(np.ceil((x2 * ratio + pad_x) * sx))
                py2 = int(np.ceil((y2 * ratio + pad_y) * sy))
                px1 = max(0, min(px1, proto_w - 1))
                py1 = max(0, min(py1, proto_h - 1))
                px2 = max(px1 + 1, min(px2, proto_w))
                py2 = max(py1 + 1, min(py2, proto_h))

                mask_crop = masks_raw[i, py1:py2, px1:px2]
                mask_resized = cv2.resize(mask_crop, (bw, bh), interpolation=cv2.INTER_LINEAR)
                masks[i, y1i:y2i, x1i:x2i] = mask_resized > 0.5

            return masks
        except Exception as e:
            logger.warning(f"Mask processing failed: {e}")
            return None

    # ── RF-DETR Postprocessing — MUST match OptimizedONNXInferencer ─────────
    # Three correctness items the previous OpenVINO version got wrong:
    #   1. RF-DETR's 91-class head reserves index 0 for background. Slice
    #      logits[:, 1:1+nc] before argmax so detections are real classes.
    #   2. Native .pth path does top-K over (queries × classes), not naive
    #      per-query argmax. Replicate that here for matching outputs.
    #   3. App class_names is 1-indexed for RF-DETR ({1: carton, 2: polybag}),
    #      so emit class_ids = (topk_idx % nc) + 1.

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        # Clip to avoid overflow on very negative mask logits (~-125).
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))

    def _postprocess_rfdetr(self, outputs: List[np.ndarray], orig_w: int, orig_h: int) -> sv.Detections:
        out_map = {name: outputs[i] for i, name in enumerate(self.output_names)}

        def _pick(*names):
            for n in names:
                if n in out_map:
                    return out_map[n]
            return None

        bboxes_raw = _pick('dets', 'pred_boxes', 'bboxes')
        logits_raw = _pick('labels', 'pred_logits')
        mask_raw = _pick('masks', 'pred_masks')

        if bboxes_raw is None or logits_raw is None:
            logger.warning(f"RF-DETR outputs missing expected names. Got: {self.output_names}")
            return sv.Detections.empty()

        bboxes = bboxes_raw[0]   # [N_queries, 4] normalized cxcywh
        logits = logits_raw[0]   # [N_queries, num_classes_head=91]

        # Drop the background column (index 0 of the COCO-shaped 91 head).
        head_nc = logits.shape[-1]
        effective_nc = min(self.nc, head_nc - 1)
        logits = logits[:, 1:1 + effective_nc]

        # Top-K over (queries × classes) — matches the native .pth path.
        probs = self._sigmoid(logits)
        flat = probs.reshape(-1)
        num_select = min(300, flat.size)
        topk_idx = np.argpartition(-flat, num_select - 1)[:num_select]
        topk_idx = topk_idx[np.argsort(-flat[topk_idx])]
        scores = flat[topk_idx]
        query_idx = topk_idx // effective_nc
        # +1 to match 1-indexed class_names ({1: carton, 2: polybag}).
        class_ids = (topk_idx % effective_nc).astype(int) + 1

        keep = scores > self.conf_threshold
        if not np.any(keep):
            return sv.Detections.empty()
        scores = scores[keep]
        query_idx = query_idx[keep]
        class_ids = class_ids[keep]

        chosen_boxes = bboxes[query_idx]
        cx, cy, w, h = chosen_boxes[:, 0], chosen_boxes[:, 1], chosen_boxes[:, 2], chosen_boxes[:, 3]
        boxes_xyxy = np.column_stack([
            cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2,
        ])

        # RF-DETR uses stretch resize (no letterbox), so normalised boxes
        # map directly to original frame by multiplication.
        boxes_xyxy[:, [0, 2]] *= orig_w
        boxes_xyxy[:, [1, 3]] *= orig_h
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, orig_w)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, orig_h)

        masks = None
        if mask_raw is not None:
            masks = self._process_rfdetr_masks(mask_raw[0], query_idx, orig_h, orig_w, boxes_xyxy)

        return sv.Detections(
            xyxy=boxes_xyxy,
            confidence=scores.astype(np.float32),
            class_id=class_ids,
            mask=masks,
        )

    def _process_rfdetr_masks(self, mask_preds: np.ndarray, query_idx: np.ndarray,
                               orig_h: int, orig_w: int, boxes: np.ndarray) -> np.ndarray:
        """Decode RF-DETR per-query mask predictions.

        RF-DETR uses stretch-resize preprocessing, so the mask in mask-space
        maps directly to the original frame. Bbox-confined upsample keeps
        cost proportional to detection area, not frame area.
        """
        try:
            gathered = mask_preds[query_idx]
            gathered = self._sigmoid(gathered)
            mh, mw = gathered.shape[1], gathered.shape[2]
            sx = mw / orig_w
            sy = mh / orig_h

            n = len(query_idx)
            masks = np.zeros((n, orig_h, orig_w), dtype=bool)
            for i in range(n):
                x1, y1, x2, y2 = boxes[i]
                x1i = max(0, min(int(x1), orig_w - 1))
                y1i = max(0, min(int(y1), orig_h - 1))
                x2i = max(x1i + 1, min(int(x2), orig_w))
                y2i = max(y1i + 1, min(int(y2), orig_h))
                bw, bh = x2i - x1i, y2i - y1i

                px1 = max(0, min(int(np.floor(x1 * sx)), mw - 1))
                py1 = max(0, min(int(np.floor(y1 * sy)), mh - 1))
                px2 = max(px1 + 1, min(int(np.ceil(x2 * sx)), mw))
                py2 = max(py1 + 1, min(int(np.ceil(y2 * sy)), mh))

                m = gathered[i, py1:py2, px1:px2]
                mask_resized = cv2.resize(m, (bw, bh), interpolation=cv2.INTER_LINEAR)
                masks[i, y1i:y2i, x1i:x2i] = mask_resized > 0.5
            return masks
        except Exception as e:
            logger.warning(f"RF-DETR mask processing failed: {e}")
            return None

    # ── Required abstract methods ────────────────────────────────────────────

    def predict(self, source: str, show: bool = False, save: bool = False):
        frame = cv2.imread(source)
        if frame is None:
            raise FileNotFoundError(f"Source not found: {source}")
        detections = self.predict_frame(frame)
        yield {"path": source, "detections": detections, "raw": detections}

    def get_summary(self, result: dict) -> dict:
        detections = result['detections']
        class_counts = {}
        if detections.class_id is not None:
            for cid in detections.class_id:
                name = self.class_names.get(cid, str(cid))
                class_counts[name] = class_counts.get(name, 0) + 1
        return {
            "file_name": Path(result['path']).name,
            "total_detections": len(detections),
            "class_counts": class_counts,
        }
