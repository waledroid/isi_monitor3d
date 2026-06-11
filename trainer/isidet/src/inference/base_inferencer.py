# src/inference/base_inferencer.py
from abc import ABC, abstractmethod
from pathlib import Path
import logging
import cv2
import yaml
import numpy as np

logger = logging.getLogger(__name__)

class BaseInferencer(ABC):
    """
    The blueprint for all inference engines.
    Keeps the main application clean and model-agnostic.
    """
    def __init__(self, model_path: str, conf_threshold: float = 0.5, device: str = None, imgsz: int = None):
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        # None = auto (GPU if available, else CPU), "cpu" = force CPU, "cuda" = force GPU
        self.device = device
        
        # 1. Dynamic Class Loading (Master Config)
        # Search multiple locations for the config file
        config_path = None
        # After the bucket restructure, this file lives at
        # isidet/src/inference/base_inferencer.py, so parent×3 = isidet/
        # and configs live at isidet/configs/train.yaml. The CWD-relative
        # fallback assumes the caller runs from repo root (convention
        # across every script + the Docker WORKDIR).
        candidates = [
            Path(__file__).resolve().parent.parent.parent / "configs/train.yaml",  # isidet/configs/
            Path("isidet/configs/train.yaml"),                                       # CWD=repo root
        ]
        # Also try relative to model path (for mounted volumes in Docker)
        for depth in range(min(5, len(self.model_path.parents))):
            candidates.append(self.model_path.parents[depth] / "configs/train.yaml")
        for p in candidates:
            if p.exists():
                config_path = p
                break
        if config_path is None:
            config_path = Path("isidet/configs/train.yaml")  # will fail gracefully below
        
        config = {}
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
            self.raw_classes = config.get('class_names', ['carton', 'polybag'])
            self.class_names = {i: name.lower() for i, name in enumerate(self.raw_classes)}
            self.nc = len(self.raw_classes)
        except Exception as e:
            logger.warning(f"⚠️ Could not load config at {config_path}: {e}")
            self.class_names = {0: "carton", 1: "polybag"}
            self.nc = 2

        self.imgsz = imgsz if imgsz is not None else config.get('image_size', 640)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"❌ Weights not found at {self.model_path}")
            
        self.model = self._load_model()

    def _preprocess_frame(self, frame: np.ndarray, target_size: int = None):
        """Standard aspect-ratio aware downscaling."""
        if target_size is None:
            target_size = self.imgsz
            
        orig_h, orig_w = frame.shape[:2]
        if max(orig_h, orig_w) > target_size:
            scale = target_size / max(orig_h, orig_w)
            new_w, new_h = int(orig_w * scale), int(orig_h * scale)
            input_frame = cv2.resize(frame, (new_w, new_h))
            needs_scaling = True
        else:
            input_frame = frame
            scale = 1.0
            needs_scaling = False
        return input_frame, scale, needs_scaling, orig_w, orig_h

    def _rescale_detections(self, detections, scale: float, orig_w: int, orig_h: int):
        """Scales detections and masks back to original resolution."""
        if len(detections) == 0:
            return detections
            
        inv_scale = 1.0 / scale
        detections.xyxy = detections.xyxy.copy() * inv_scale
        
        if detections.mask is not None:
            resized_masks = []
            for mask in detections.mask:
                m = cv2.resize(mask.astype(np.uint8), (orig_w, orig_h), 
                             interpolation=cv2.INTER_NEAREST)
                resized_masks.append(m.astype(bool))
            detections.mask = np.array(resized_masks)
            
        return detections

    @abstractmethod
    def _load_model(self):
        """Loads the model into RAM/VRAM."""
        pass

    @abstractmethod
    def predict(self, source: str, show: bool = False, save: bool = False):
        """Runs the inference loop (must yield results one by one for video streams)."""
        pass

    @abstractmethod
    def get_summary(self, result) -> dict:
        """Translates the model's complex output into a simple dictionary."""
        pass
