# src/inference/yolo_inferencer.py
import os
import torch
import cv2
import supervision as sv
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from src.inference.base_inferencer import BaseInferencer

class YOLOInferencer(BaseInferencer):
    """Concrete implementation for YOLO Segmentation models with 640px scaling."""

    def __init__(self, model_path: str, conf_threshold: float = 0.5, device: str = None, imgsz: int = None):
        super().__init__(model_path, conf_threshold, device, imgsz)
        # Resolve once: YOLO uses int 0 for first GPU, "cpu" for CPU
        if self.device == "cpu":
            self._device = "cpu"
        else:
            self._device = 0 if torch.cuda.is_available() else "cpu"
        self.class_names = self.model.names

    def _load_model(self):
        return YOLO(self.model_path)

    def predict_frame(self, frame: np.ndarray):
        """Processes a video frame with shared preprocessing."""
        # 1. Shared Scaling Logic
        input_frame, scale, needs_scaling, orig_w, orig_h = self._preprocess_frame(frame)

        # 2. Inference
        results = self.model(input_frame, conf=self.conf_threshold, imgsz=self.imgsz, verbose=False, device=self._device)[0]
        detections = sv.Detections.from_ultralytics(results)
        del results  # free GPU tensors immediately — don't wait for GC

        # 3. Shared Scaling Back
        if needs_scaling:
            detections = self._rescale_detections(detections, scale, orig_w, orig_h)
            
        return detections

    def predict(self, source: str, show: bool = False, save: bool = False):
        source_val = int(source) if str(source).isdigit() else source
        results_gen = self.model.predict(
            source=source_val, conf=self.conf_threshold, stream=True, verbose=False, device=self._device
        )
        for r in results_gen:
            detections = sv.Detections.from_ultralytics(r)
            yield {"path": r.path, "detections": detections, "raw": r}

    def get_summary(self, result) -> dict:
        detections = result['detections']
        return {"file_name": Path(result['path']).name, "counts": {"Total": len(detections)}}
