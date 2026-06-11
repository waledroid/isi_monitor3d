"""
Optimized ONNX Inference Engine — GPU (CUDA) or CPU fallback.
Compatible with ONNX Runtime 1.24+
Handles both YOLO (nms=True export) and RF-DETR ONNX models.
"""

import cv2
import numpy as np
import supervision as sv
from pathlib import Path
from typing import List
import logging

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
except ImportError:
    raise ImportError("onnxruntime not installed. pip install onnxruntime-gpu or onnxruntime")

# Auto-fix: if both onnxruntime and onnxruntime-gpu are installed, the CPU
# package shadows the GPU one and CUDAExecutionProvider disappears.
# Uninstall the CPU-only package automatically so the GPU one takes over.
if 'CUDAExecutionProvider' not in ort.get_available_providers():
    try:
        import importlib.metadata as _meta
        _meta.distribution('onnxruntime')
        _meta.distribution('onnxruntime-gpu')
        # Both installed — CPU is shadowing GPU. Auto-fix.
        logger.warning("Both onnxruntime and onnxruntime-gpu installed — CPU is shadowing GPU. "
                        "Auto-removing onnxruntime (CPU) to fix CUDA support...")
        import subprocess, sys
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'uninstall', 'onnxruntime', '-y'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        # Reload onnxruntime so CUDA provider becomes available
        import importlib
        ort = importlib.reload(ort)
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            logger.info(f"Fixed! CUDA provider now available: {ort.get_available_providers()}")
        else:
            logger.warning("Removed CPU onnxruntime but CUDA still not available. "
                           "Restart the process for changes to take effect.")
    except _meta.PackageNotFoundError:
        pass  # Only one version installed — no conflict
    except Exception as e:
        logger.warning(f"Auto-fix failed: {e}. "
                       "Manual fix: pip uninstall onnxruntime -y && pip install onnxruntime-gpu")

from src.inference.base_inferencer import BaseInferencer


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically-stable sigmoid. Clips logits to [-50, 50] before exp to
    avoid overflow warnings on very negative mask logits (can reach -125).
    At those magnitudes the output saturates to 0 anyway — clipping changes
    nothing meaningful but silences the RuntimeWarning.
    """
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


# ── ONNX preload for CUDA kernel cache priming ─────────────────────────────
# Previous implementation cached session objects across threads — caused
# massive cross-thread CUDA stream sync stalls (sessions bound to the build
# thread's stream; Flask request thread then waits seconds for re-binding).
#
# The pragmatic fix: build a *throwaway* session at boot so that CUDNN's
# algorithm selection cache is populated at the **process / driver level**.
# When StreamHandler later builds its own session for real inference, it
# reuses those compiled kernels and skips autotuning — getting most of the
# speed-up without the thread-safety hazard.
def _canon_path(model_path) -> str:
    try:
        return str(Path(model_path).resolve())
    except Exception:
        return str(model_path)


def preload_onnx(model_path: str, gpu_device_id: int = 0) -> None:
    """Prime the CUDNN kernel cache for this model. Safe to call from a
    background thread at app startup; the session built here is discarded
    immediately so it can never be used from the wrong thread later.
    """
    try:
        path = _canon_path(model_path)
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.log_severity_level = 3
        available = ort.get_available_providers()
        if 'CUDAExecutionProvider' in available:
            providers = [
                ('CUDAExecutionProvider', {
                    'device_id': gpu_device_id,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'cudnn_conv_algo_search': 'HEURISTIC',
                    'do_copy_in_default_stream': True,
                }),
                'CPUExecutionProvider',
            ]
        else:
            providers = ['CPUExecutionProvider']
        session = ort.InferenceSession(path, sess_options=opts, providers=providers)
        # Run one dummy inference to force CUDNN algorithm selection + kernel
        # JIT. After this, the CUDA driver retains compiled kernels for the
        # lifetime of the process.
        inp = session.get_inputs()[0]
        shape = [1 if d is None or isinstance(d, str) else d for d in inp.shape]
        import numpy as _np
        dummy = _np.random.randn(*shape).astype(_np.float32)
        session.run(None, {inp.name: dummy})
        del session  # drop the session; keep the kernel cache (process-level)
        logger.info(f"🔥 ONNX preload (CUDA kernels warm, session discarded): {path}")
    except Exception as e:
        logger.warning(f"ONNX preload skipped for {model_path}: {e}")


class OptimizedONNXInferencer(BaseInferencer):
    """ONNX inference engine with automatic GPU/CPU selection."""

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.5,
        device: str = None,
        imgsz: int = None,
        gpu_device_id: int = 0,
    ):
        super().__init__(model_path, conf_threshold, device, imgsz)

        self.gpu_device_id = gpu_device_id

        # Session setup
        self.sess_options = ort.SessionOptions()
        self.sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess_options.enable_cpu_mem_arena = False
        self.sess_options.enable_mem_pattern = True
        self.sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.sess_options.intra_op_num_threads = 2
        self.sess_options.inter_op_num_threads = 2
        self.sess_options.log_severity_level = 3

        self.session = self._create_session(str(model_path))
        self._parse_model_metadata()

        # Match RF-DETR .pth convention: class_names 1-indexed (carton=1,
        # polybag=2). sv.MaskAnnotator/BoxAnnotator index the colour palette
        # by class_id, so this keeps YOLO ↔ RF-DETR palette slots distinct
        # — operators see colours flip on hot-swap as visual confirmation.
        # Downstream lookups use `class_names.get(id)` and dict keys are
        # strings (carton/polybag), so counts/CSV/UDP behave identically.
        if self.is_rfdetr:
            self.class_names = {i + 1: name for i, name in self.class_names.items()}

        self._warmup()

        logger.info(f"ONNX model loaded: {model_path}")
        logger.info(f"  Type: {'RF-DETR' if self.is_rfdetr else 'YOLO'} | "
                     f"Input: {self.model_w}x{self.model_h} | "
                     f"Classes: {self.nc} | "
                     f"Provider: {self.session.get_providers()[0]}")
        for o in self.session.get_outputs():
            logger.info(f"    out: name={o.name!r} shape={o.shape} dtype={o.type}")

    def _load_model(self):
        return self.session if hasattr(self, 'session') else None

    # ── Session Creation ─────────────────────────────────────────────────────

    def _create_session(self, model_path: str) -> ort.InferenceSession:
        available = ort.get_available_providers()
        force_cpu = (self.device == "cpu")

        if force_cpu or 'CUDAExecutionProvider' not in available:
            providers = ['CPUExecutionProvider']
        else:
            cuda_opts = {
                'device_id': self.gpu_device_id,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'cudnn_conv_algo_search': 'HEURISTIC',
                'do_copy_in_default_stream': True,
            }
            providers = [('CUDAExecutionProvider', cuda_opts), 'CPUExecutionProvider']

        # Always build a fresh session in the caller's thread. See comment
        # at `preload_onnx` above for why we don't share session objects.
        return ort.InferenceSession(model_path, sess_options=self.sess_options, providers=providers)

    # ── Model Metadata ───────────────────────────────────────────────────────

    def _parse_model_metadata(self):
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_shape = list(inp.shape)
        if self.input_shape[0] is None:
            self.input_shape[0] = 1

        self.model_h = self.input_shape[2]
        self.model_w = self.input_shape[3]

        self.output_names = [o.name for o in self.session.get_outputs()]
        self.is_rfdetr = any(n in self.output_names for n in
                             ('dets', 'pred_logits', 'pred_boxes', 'bboxes', 'labels'))

        # self.nc is already set by BaseInferencer from configs/train.yaml
        # (class_names list). For RF-DETR the ONNX head keeps its pretrained
        # width (~91 COCO slots); we handle that in postprocessing by slicing
        # to the first `self.nc` fine-tuned classes. For YOLO with nms=True
        # the output encodes class as a single column, so nc is not derivable
        # from shape — trust the config.
        self.head_nc = None
        for o in self.session.get_outputs():
            if o.name in ('labels', 'pred_logits'):
                self.head_nc = o.shape[-1]
                break

        self._has_mask_output = len(self.session.get_outputs()) > 1

    def _warmup(self):
        dummy = np.random.randn(*self.input_shape).astype(np.float32)
        for _ in range(3):
            self.session.run(None, {self.input_name: dummy})

    # ── Preprocessing ────────────────────────────────────────────────────────

    def _letterbox(self, frame: np.ndarray, pad_color: int = 114):
        """Aspect-preserving resize + pad to (model_h, model_w).
        Returns (padded_img, ratio, pad_left, pad_top) — needed to invert on output boxes.
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

    # ImageNet statistics — required by RF-DETR (DINOv2 backbone).
    _IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    _IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        # Ultralytics and RF-DETR train on RGB. OpenCV/RTSP frames are BGR.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.is_rfdetr:
            # RF-DETR (DINOv2) is trained on stretch-resized inputs; the native
            # .pth path computes target_sizes = (orig_w, orig_h) and scales
            # normalised boxes directly. No letterbox, no pad.
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
            outputs = self.session.run(None, {self.input_name: input_tensor})
        except Exception as e:
            logger.error(f"ONNX inference failed: {e}")
            return sv.Detections.empty()

        if not getattr(self, '_logged_runtime_shapes', False):
            for i, o in enumerate(outputs):
                logger.info(f"ONNX runtime output[{i}] name={self.output_names[i]!r} "
                             f"shape={o.shape} dtype={o.dtype}")
            self._logged_runtime_shapes = True

        orig_h, orig_w = frame.shape[:2]
        try:
            if self.is_rfdetr:
                return self._postprocess_rfdetr(outputs, orig_w, orig_h)
            else:
                return self._postprocess_yolo(outputs, orig_w, orig_h, frame)
        except Exception as e:
            logger.error(f"Postprocessing failed: {e}")
            return sv.Detections.empty()

    # ── YOLO Postprocessing ──────────────────────────────────────────────────

    def _postprocess_yolo(self, outputs: List[np.ndarray], orig_w: int, orig_h: int,
                          frame: np.ndarray) -> sv.Detections:
        # Two possible Ultralytics ONNX layouts:
        #   post-NMS (nms=True):  [1, N<=300, 6|6+32]  = [x1,y1,x2,y2, score, class, (mask_coeffs)]
        #   raw (nms=False):      [1, 4+nc(+32), A]    or  [1, A, 4+nc(+32)]  where A is anchor count (~3549 for 416)
        # We distinguish by anchor-axis magnitude; raw exports have a dim > 1000.
        raw = outputs[0][0]
        s0, s1 = raw.shape[0], raw.shape[1]
        is_raw = max(s0, s1) > 1000

        if is_raw:
            if s0 > s1:
                preds = raw                  # [A, features]
            else:
                preds = raw.T                # [A, features]
            return self._postprocess_yolo_raw(preds, outputs, orig_w, orig_h)

        # ── Post-NMS path ────────────────────────────────────────────────────
        preds = raw
        if preds.shape[0] == 0:
            return sv.Detections.empty()

        n_cols = preds.shape[1]
        has_mask_coeffs = n_cols > 6
        boxes = preds[:, :4].copy()
        confidences = preds[:, 4].astype(np.float32)
        class_ids = preds[:, 5].astype(int)
        mask_coeffs = preds[:, 6:] if has_mask_coeffs else None

        keep = confidences > self.conf_threshold
        if not np.any(keep):
            return sv.Detections.empty()
        boxes = boxes[keep]; class_ids = class_ids[keep]; confidences = confidences[keep]
        if mask_coeffs is not None:
            mask_coeffs = mask_coeffs[keep]

        ratio, pad_x, pad_y = getattr(self, '_last_letterbox', (1.0, 0, 0))
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / ratio
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / ratio
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

        valid = (boxes[:, 2] > boxes[:, 0] + 1) & (boxes[:, 3] > boxes[:, 1] + 1)
        if not np.all(valid):
            boxes = boxes[valid]; class_ids = class_ids[valid]; confidences = confidences[valid]
            if mask_coeffs is not None:
                mask_coeffs = mask_coeffs[valid]

        if len(boxes) == 0:
            return sv.Detections.empty()

        masks = None
        if mask_coeffs is not None and len(outputs) > 1:
            masks = self._process_yolo_masks(outputs[1][0], mask_coeffs, boxes, orig_h, orig_w)

        if not getattr(self, '_logged_mask_sample', False):
            if masks is not None and len(masks) > 0:
                logger.info(f"YOLO masks: N={len(masks)} shape={masks.shape} "
                             f"dtype={masks.dtype} nonzero_px={int(np.count_nonzero(masks))}")
                self._logged_mask_sample = True
            elif masks is None:
                logger.warning(f"YOLO masks: None (has_mask_coeffs={mask_coeffs is not None}, "
                                f"outputs_count={len(outputs)})")
                self._logged_mask_sample = True

        return sv.Detections(
            xyxy=boxes, confidence=confidences, class_id=class_ids, mask=masks,
        )

    def _postprocess_yolo_raw(self, preds: np.ndarray, outputs: List[np.ndarray],
                               orig_w: int, orig_h: int,
                               iou_threshold: float = 0.45) -> sv.Detections:
        # preds shape: [A, 4 + nc + (32 if seg)]
        #   cols 0:4  = box cxcywh in model (letterboxed) pixels
        #   cols 4:4+nc = per-class scores (already probability)
        #   cols 4+nc: = mask coefficients (seg only)
        ncols = preds.shape[1]
        ncls = self.nc
        has_masks = ncols > 4 + ncls

        class_scores = preds[:, 4:4 + ncls]
        confidences = class_scores.max(axis=1).astype(np.float32)
        class_ids = class_scores.argmax(axis=1).astype(int)

        keep = confidences > self.conf_threshold
        if not np.any(keep):
            return sv.Detections.empty()

        boxes_cxcywh = preds[keep, :4]
        confidences = confidences[keep]
        class_ids = class_ids[keep]
        mask_coeffs = preds[keep, 4 + ncls:] if has_masks else None

        cx, cy, w, h = boxes_cxcywh[:, 0], boxes_cxcywh[:, 1], boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]
        boxes = np.column_stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]).astype(np.float32)

        # Class-aware NMS using cv2.dnn.NMSBoxes (expects xywh)
        xywh = np.column_stack([boxes[:, 0], boxes[:, 1],
                                 boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]])
        keep_idx_all = []
        for cid in np.unique(class_ids):
            cls_mask = class_ids == cid
            cls_boxes = xywh[cls_mask].tolist()
            cls_confs = confidences[cls_mask].tolist()
            kept = cv2.dnn.NMSBoxes(cls_boxes, cls_confs, self.conf_threshold, iou_threshold)
            if len(kept) == 0:
                continue
            kept = np.array(kept).flatten()
            global_idx = np.where(cls_mask)[0][kept]
            keep_idx_all.extend(global_idx.tolist())

        if not keep_idx_all:
            return sv.Detections.empty()

        keep_idx_all = np.array(keep_idx_all, dtype=int)
        boxes = boxes[keep_idx_all]
        confidences = confidences[keep_idx_all]
        class_ids = class_ids[keep_idx_all]
        if mask_coeffs is not None:
            mask_coeffs = mask_coeffs[keep_idx_all]

        ratio, pad_x, pad_y = getattr(self, '_last_letterbox', (1.0, 0, 0))
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / ratio
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / ratio
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

        valid = (boxes[:, 2] > boxes[:, 0] + 1) & (boxes[:, 3] > boxes[:, 1] + 1)
        if not np.all(valid):
            boxes = boxes[valid]; confidences = confidences[valid]; class_ids = class_ids[valid]
            if mask_coeffs is not None:
                mask_coeffs = mask_coeffs[valid]
        if len(boxes) == 0:
            return sv.Detections.empty()

        masks = None
        if mask_coeffs is not None and len(outputs) > 1:
            masks = self._process_yolo_masks(outputs[1][0], mask_coeffs, boxes, orig_h, orig_w)

        return sv.Detections(
            xyxy=boxes, confidence=confidences, class_id=class_ids, mask=masks,
        )

    def _process_yolo_masks(self, proto: np.ndarray, coeffs: np.ndarray,
                            boxes: np.ndarray, orig_h: int, orig_w: int) -> np.ndarray:
        """Decode YOLO mask coefficients using mask prototypes.

        Only the bbox region of each mask is upsampled to original resolution —
        cost scales with box area, not frame area. A full-frame upsample for
        every detection would dominate wall-clock on 1080p/4K sources.

        Args:
            proto: [32, proto_h, proto_w] mask prototypes (letterboxed model space).
            coeffs: [N, 32] per-detection coefficients.
            boxes: [N, 4] xyxy detection boxes in original image space.
        """
        try:
            n = len(boxes)
            if n == 0:
                return np.zeros((0, orig_h, orig_w), dtype=bool)

            proto_h, proto_w = proto.shape[1], proto.shape[2]
            masks_raw = coeffs @ proto.reshape(proto.shape[0], -1)
            masks_raw = masks_raw.reshape(n, proto_h, proto_w)
            masks_raw = _sigmoid(masks_raw)

            ratio, pad_x, pad_y = getattr(self, '_last_letterbox', (1.0, 0, 0))
            sx = proto_w / self.model_w        # orig→proto scale (model_x * sx)
            sy = proto_h / self.model_h
            # supervision annotators (and sv.Detections.from_ultralytics) use bool masks.
            masks = np.zeros((n, orig_h, orig_w), dtype=bool)

            for i in range(n):
                x1, y1, x2, y2 = boxes[i]
                x1i = max(0, min(int(x1), orig_w - 1))
                y1i = max(0, min(int(y1), orig_h - 1))
                x2i = max(x1i + 1, min(int(x2), orig_w))
                y2i = max(y1i + 1, min(int(y2), orig_h))
                bw, bh = x2i - x1i, y2i - y1i

                # Map bbox from original → letterboxed model → proto coords
                px1 = int(np.floor((x1 * ratio + pad_x) * sx))
                py1 = int(np.floor((y1 * ratio + pad_y) * sy))
                px2 = int(np.ceil((x2 * ratio + pad_x) * sx))
                py2 = int(np.ceil((y2 * ratio + pad_y) * sy))
                px1 = max(0, min(px1, proto_w - 1))
                py1 = max(0, min(py1, proto_h - 1))
                px2 = max(px1 + 1, min(px2, proto_w))
                py2 = max(py1 + 1, min(py2, proto_h))

                mask_crop = masks_raw[i, py1:py2, px1:px2]
                # Resize only the bbox region — not the full frame.
                mask_resized = cv2.resize(mask_crop, (bw, bh), interpolation=cv2.INTER_LINEAR)
                masks[i, y1i:y2i, x1i:x2i] = mask_resized > 0.5

            return masks
        except Exception as e:
            logger.warning(f"Mask processing failed: {e}")
            return None

    # ── RF-DETR Postprocessing ───────────────────────────────────────────────

    def _log_rfdetr_diag(self, logits_sliced: np.ndarray, probs_full: np.ndarray,
                          bboxes: np.ndarray, orig_w: int, orig_h: int) -> None:
        if getattr(self, '_rfdetr_diag_logged', False):
            return
        self._rfdetr_diag_logged = True
        logger.info(f"RF-DETR diag: logits shape={logits_sliced.shape} "
                     f"min={logits_sliced.min():.3f} max={logits_sliced.max():.3f} "
                     f"mean={logits_sliced.mean():.3f}")
        top5 = np.sort(probs_full.reshape(-1))[-5:][::-1]
        logger.info(f"RF-DETR diag: top-5 (query,class) sigmoid scores = {top5}")
        logger.info(f"RF-DETR diag: conf_threshold={self.conf_threshold}  "
                     f"count_above={int((probs_full > self.conf_threshold).sum())}")
        bb = bboxes
        logger.info(f"RF-DETR diag: box stats cxcywh "
                     f"cx=[{bb[:,0].min():.3f},{bb[:,0].max():.3f}] "
                     f"cy=[{bb[:,1].min():.3f},{bb[:,1].max():.3f}] "
                     f"w=[{bb[:,2].min():.3f},{bb[:,2].max():.3f}] "
                     f"h=[{bb[:,3].min():.3f},{bb[:,3].max():.3f}]")
        logger.info(f"RF-DETR diag: model_w={self.model_w} model_h={self.model_h} "
                     f"orig_w={orig_w} orig_h={orig_h}")

    def _postprocess_rfdetr(self, outputs: List[np.ndarray], orig_w: int, orig_h: int) -> sv.Detections:
        # Build a name-keyed view so we don't rely on positional output order
        # (exporters may emit pred_boxes/pred_logits/pred_masks or dets/labels/masks).
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
        logits = logits_raw[0]   # [N_queries, num_classes_head]

        # RF-DETR's 91-class head follows COCO convention:
        #   index 0 = background/no-object  (always near-zero sigmoid)
        #   indices 1..nc = the fine-tuned classes (carton, polybag, ...)
        # Skip the background column and take the next `nc` slots. After the
        # slice, column 0 == carton (app class 0), column 1 == polybag (app class 1).
        head_nc = logits.shape[-1]
        effective_nc = min(self.nc, head_nc - 1)
        logits = logits[:, 1:1 + effective_nc]

        # Replicate the native RF-DETR topk-over-(queries*classes) selection
        # (src/training/trainers/rfdetr.py:92-140). This is what the .pth path does.
        probs = _sigmoid(logits)                              # sigmoid
        self._log_rfdetr_diag(logits, probs, bboxes, orig_w, orig_h)
        flat = probs.reshape(-1)
        num_select = min(300, flat.size)
        topk_idx = np.argpartition(-flat, num_select - 1)[:num_select]
        topk_idx = topk_idx[np.argsort(-flat[topk_idx])]     # sort by score desc
        scores = flat[topk_idx]
        query_idx = topk_idx // effective_nc
        # +1 to match 1-indexed class_names ({1: carton, 2: polybag}).
        # After the slice logits[:, 1:1+nc], column 0 of the view is the
        # first fine-tuned class, so `% nc` gives 0..nc-1 within the view,
        # and +1 restores the original 91-head column index we want to emit.
        class_ids = (topk_idx % effective_nc).astype(int) + 1

        keep = scores > self.conf_threshold
        if not np.any(keep):
            return sv.Detections.empty()
        scores = scores[keep]
        query_idx = query_idx[keep]
        class_ids = class_ids[keep]

        chosen_boxes = bboxes[query_idx]                     # [K, 4] normalized cxcywh
        cx, cy, w, h = chosen_boxes[:, 0], chosen_boxes[:, 1], chosen_boxes[:, 2], chosen_boxes[:, 3]
        boxes_xyxy = np.column_stack([
            cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2,
        ])

        # RF-DETR preprocesses with stretch-resize (no letterbox), so normalised
        # boxes map directly to original frame by multiplication — same as the
        # native .pth path which uses target_sizes = (orig_w, orig_h).
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
        """Process RF-DETR per-query mask predictions.

        RF-DETR uses stretch-resize preprocessing, so the mask in model/mask
        space maps directly to the original frame by a simple resize.
        We still bbox-confine to keep it fast.
        """
        try:
            gathered = mask_preds[query_idx]
            gathered = _sigmoid(gathered)
            mh, mw = gathered.shape[1], gathered.shape[2]
            sx = mw / orig_w   # orig→mask scale
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


# Alias for backward compatibility
ONNXInferencer = OptimizedONNXInferencer
