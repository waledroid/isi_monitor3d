"""
TensorRT Inference Engine — maximum throughput on NVIDIA GPUs.
Loads .engine files produced by export_engine.py.

Requires: tensorrt + pycuda (pip install tensorrt pycuda)
Gracefully raises ImportError if not installed.
"""

import cv2
import numpy as np
import supervision as sv
from pathlib import Path
from typing import List
import logging

logger = logging.getLogger(__name__)

from src.inference.base_inferencer import BaseInferencer

# Lazy imports — checked at instantiation, not at module load
_trt = None
_cuda = None


def _check_tensorrt():
    global _trt, _cuda
    if _trt is not None:
        return
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401 — initializes CUDA context
        _trt = trt
        _cuda = cuda
    except ImportError:
        raise ImportError(
            "TensorRT inference requires: pip install tensorrt pycuda\n"
            "And NVIDIA TensorRT libraries on LD_LIBRARY_PATH."
        )


class TensorRTInferencer(BaseInferencer):
    """TensorRT inference engine — loads serialized .engine files."""

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.5,
        device: str = None,
        imgsz: int = None,
    ):
        _check_tensorrt()
        super().__init__(model_path, conf_threshold, device, imgsz)

        engine_path = Path(model_path)
        if not engine_path.exists():
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")

        # Load engine
        trt_logger = _trt.Logger(_trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            runtime = _trt.Runtime(trt_logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # Parse bindings
        self._inputs = []
        self._outputs = []
        self._bindings = []
        self._stream = _cuda.Stream()

        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            shape = self.engine.get_binding_shape(i)
            dtype = _trt.nptype(self.engine.get_binding_dtype(i))
            size = int(np.prod(shape)) * np.dtype(dtype).itemsize

            host_mem = _cuda.pagelocked_empty(int(np.prod(shape)), dtype)
            device_mem = _cuda.mem_alloc(size)

            self._bindings.append(int(device_mem))
            binding = {'name': name, 'shape': shape, 'dtype': dtype,
                       'host': host_mem, 'device': device_mem}

            if self.engine.binding_is_input(i):
                self._inputs.append(binding)
                if len(shape) == 4:
                    self.model_h = shape[2]
                    self.model_w = shape[3]
            else:
                self._outputs.append(binding)

        # Detect model type from output shapes
        self.is_rfdetr = any(b['name'] in ('dets', 'labels', 'pred_logits') for b in self._outputs)
        self._has_mask_output = len(self._outputs) > 1

        # Infer nc
        if self.is_rfdetr:
            for b in self._outputs:
                if b['name'] in ('labels', 'pred_logits'):
                    self.nc = b['shape'][-1]
                    break
        else:
            out_shape = self._outputs[0]['shape']
            if len(out_shape) == 3:
                mask_c = 32 if self._has_mask_output else 0
                self.nc = out_shape[2] - 4 - mask_c
                if self.nc <= 0:
                    self.nc = 2

        logger.info(f"TensorRT engine loaded: {engine_path}")
        logger.info(f"  Type: {'RF-DETR' if self.is_rfdetr else 'YOLO'} | "
                     f"Input: {self.model_w}x{self.model_h} | Classes: {self.nc}")

    def _load_model(self):
        return self.engine if hasattr(self, 'engine') else None

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        img = cv2.resize(frame, (self.model_w, self.model_h), interpolation=cv2.INTER_LINEAR)
        img = img.transpose((2, 0, 1)).astype(np.float32) / 255.0
        return np.ascontiguousarray(img.ravel())

    def predict_frame(self, frame: np.ndarray) -> sv.Detections:
        input_data = self.preprocess(frame)

        # Copy input to device
        np.copyto(self._inputs[0]['host'], input_data)
        _cuda.memcpy_htod_async(self._inputs[0]['device'], self._inputs[0]['host'], self._stream)

        # Run inference
        self.context.execute_async_v2(bindings=self._bindings, stream_handle=self._stream.handle)

        # Copy outputs back
        outputs = []
        for out in self._outputs:
            _cuda.memcpy_dtoh_async(out['host'], out['device'], self._stream)
        self._stream.synchronize()

        for out in self._outputs:
            outputs.append(out['host'].reshape(out['shape']))

        orig_h, orig_w = frame.shape[:2]
        try:
            if self.is_rfdetr:
                return self._postprocess_rfdetr(outputs, orig_w, orig_h)
            else:
                return self._postprocess_yolo(outputs, orig_w, orig_h)
        except Exception as e:
            logger.error(f"TensorRT postprocessing failed: {e}")
            return sv.Detections.empty()

    # ── Postprocessing (same logic as ONNX/OpenVINO) ─────────────────────────

    def _postprocess_yolo(self, outputs: List[np.ndarray], orig_w: int, orig_h: int) -> sv.Detections:
        preds = outputs[0][0] if outputs[0].ndim == 3 else outputs[0]
        if preds.shape[0] == 0:
            return sv.Detections.empty()

        boxes = preds[:, :4].copy()
        scores = preds[:, 4:4 + self.nc]
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        keep = confidences > self.conf_threshold
        if not np.any(keep):
            return sv.Detections.empty()

        boxes, class_ids, confidences = boxes[keep], class_ids[keep], confidences[keep]

        scale_x, scale_y = orig_w / self.model_w, orig_h / self.model_h
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

        valid = (boxes[:, 2] > boxes[:, 0] + 1) & (boxes[:, 3] > boxes[:, 1] + 1)
        if not np.all(valid):
            boxes, class_ids, confidences = boxes[valid], class_ids[valid], confidences[valid]
        if len(boxes) == 0:
            return sv.Detections.empty()

        return sv.Detections(
            xyxy=boxes, confidence=confidences,
            class_id=class_ids.astype(int), mask=None,
        )

    def _postprocess_rfdetr(self, outputs: List[np.ndarray], orig_w: int, orig_h: int) -> sv.Detections:
        bboxes = outputs[0][0] if outputs[0].ndim == 3 else outputs[0]
        logits = outputs[1][0] if outputs[1].ndim == 3 else outputs[1]

        probs = 1.0 / (1.0 + np.exp(-logits))
        scores = np.max(probs, axis=1)
        class_ids = np.argmax(probs, axis=1)

        keep = scores > self.conf_threshold
        if not np.any(keep):
            return sv.Detections.empty()

        bboxes, scores, class_ids = bboxes[keep], scores[keep], class_ids[keep]
        cx, cy, w, h = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
        boxes = np.column_stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2])
        boxes[:, [0, 2]] *= orig_w
        boxes[:, [1, 3]] *= orig_h
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

        return sv.Detections(
            xyxy=boxes, confidence=scores,
            class_id=class_ids.astype(int), mask=None,
        )

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
