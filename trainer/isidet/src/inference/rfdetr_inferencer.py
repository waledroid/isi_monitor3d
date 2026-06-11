import cv2
import torch
import supervision as sv
from pathlib import Path
from PIL import Image
import numpy as np
from src.inference.base_inferencer import BaseInferencer


class RFDETRInferencer(BaseInferencer):
    """Concrete implementation for RF-DETR Segmentation with 640px scaling."""

    def __init__(self, model_path: str, conf_threshold: float = 0.5, device: str = None, imgsz: int = None):
        super().__init__(model_path, conf_threshold, device, imgsz)
        # Resolve once
        if self.device == "cpu":
            self._device = "cpu"
        else:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        # RF-DETR Shift: Starts from 1 (COCO Convention)
        self.class_names = {i+1: name for i, name in self.class_names.items()}
        # Move model to target device at load time
        try:
            self.model.to(self._device)
        except Exception:
            pass

    def _load_model(self):
        # Lazy import — rfdetr's CUDA extensions segfault at import time in some Docker containers
        if "small" in str(self.model_path).lower():
            from rfdetr import RFDETRSegSmall
            model = RFDETRSegSmall(pretrain_weights=str(self.model_path))
        else:
            from rfdetr import RFDETRSegMedium
            model = RFDETRSegMedium(pretrain_weights=str(self.model_path))
        model.optimize_for_inference()
        return model

    def predict_frame(self, frame: np.ndarray):
        """Processes a video frame with shared preprocessing."""
        # 1. Shared Scaling Logic
        input_frame, scale, needs_scaling, orig_w, orig_h = self._preprocess_frame(frame)
        
        # 2. PIL Conversion & Inference
        image = Image.fromarray(cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB))
        detections = self.model.predict(image, threshold=self.conf_threshold)
        del image  # release immediately — don't wait for GC at high frame rates
        
        # 3. Shared Scaling Back
        if needs_scaling:
            detections = self._rescale_detections(detections, scale, orig_w, orig_h)
            
        return detections

    def predict(self, source: str, show: bool = False, save: bool = False):
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"Source image not found: {source}")

        image = Image.open(source_path).convert("RGB")
        detections = self.model.predict(image, threshold=self.conf_threshold)
        
        if show or save:
            self._visualize(source_path, detections, show, save)
        yield {"path": str(source_path), "detections": detections, "raw": detections}

    def _visualize(self, path: Path, detections: sv.Detections, show: bool, save: bool):
        cv_image = cv2.imread(str(path))
        mask_annotator, box_annotator = sv.MaskAnnotator(), sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
        
        annotated = mask_annotator.annotate(scene=cv_image, detections=detections)
        annotated = box_annotator.annotate(scene=annotated, detections=detections)
        
        labels = [f"{self.class_names.get(c_id, 'obj')} {conf:.2f}" 
                  for c_id, conf in zip(detections.class_id, detections.confidence)]
        annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
        
        if save:
            # CWD-relative; assumes caller runs from repo root (matches every
            # other script in the repo). runs/ lives under isidet/.
            out_dir = Path("isidet/runs/segment/predict/rfdetr")
            out_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_dir / path.name), annotated)
        if show:
            cv2.imshow("RF-DETR Seg", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def get_summary(self, result) -> dict:
        detections = result['detections']
        return {"file_name": Path(result['path']).name, "counts": {"Total": len(detections)}}
