"""Remote RF-DETR Inferencer — proxies inference to the rfdetr microservice container.

Used in Docker where the rfdetr package can't import locally.
Sends frames via HTTP to the rfdetr service running in a separate container.
"""

import base64
import logging

import cv2
import numpy as np
import requests
import supervision as sv
from pathlib import Path

from src.inference.base_inferencer import BaseInferencer

logger = logging.getLogger(__name__)

RFDETR_SERVICE_URL = "http://rfdetr:9510"


class RemoteRFDETRInferencer(BaseInferencer):
    """Proxy inferencer that delegates to the rfdetr Docker service."""

    def __init__(self, model_path: str, conf_threshold: float = 0.5,
                 device: str = None, imgsz: int = None,
                 service_url: str = None):
        # Don't call super().__init__() fully — we don't load a model locally
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.device = device
        self.service_url = service_url or RFDETR_SERVICE_URL

        # Load class names from config (same as base)
        self._load_config()

        # Tell the rfdetr service to load the model
        self._remote_load(str(model_path), conf_threshold, imgsz, device)

    def _load_config(self):
        """Load class names from configs/train.yaml."""
        import yaml
        config = {}
        # isidet/src/inference/remote_rfdetr_inferencer.py → parent×3 = isidet/;
        # CWD fallback assumes caller runs from repo root.
        candidates = [
            Path(__file__).resolve().parent.parent.parent / "configs/train.yaml",
            Path("isidet/configs/train.yaml"),
        ]
        for p in candidates:
            if p.exists():
                try:
                    with open(p) as f:
                        config = yaml.safe_load(f) or {}
                    break
                except Exception:
                    pass

        raw = config.get('class_names', ['carton', 'polybag'])
        # RF-DETR uses 1-indexed class IDs (COCO convention)
        self.class_names = {i + 1: name.lower() for i, name in enumerate(raw)}
        self.nc = len(raw)

    def _remote_load(self, weights: str, conf_threshold: float, imgsz: int = None, device: str = None):
        """Tell the rfdetr service to load a model."""
        try:
            payload = {"weights": weights, "conf_threshold": conf_threshold}
            if imgsz is not None:
                payload["imgsz"] = imgsz
            if device is not None:
                payload["device"] = device
                
            resp = requests.post(
                f"{self.service_url}/load",
                json=payload,
                timeout=120,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"RF-DETR service error: {resp.json()}")
            logger.info(f"RF-DETR remote model loaded: {weights}")
        except requests.ConnectionError:
            raise RuntimeError(
                "RF-DETR service not reachable. Is the rfdetr container running? "
                "Check: docker compose ps"
            )

    def _load_model(self):
        return None  # No local model

    def predict_frame(self, frame: np.ndarray) -> sv.Detections:
        # Encode frame to JPEG → base64
        _, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        frame_b64 = base64.b64encode(buf).decode('ascii')

        try:
            resp = requests.post(
                f"{self.service_url}/predict",
                json={"frame": frame_b64},
                timeout=30,
            )
            if resp.status_code != 200:
                logger.error(f"RF-DETR predict failed: {resp.text}")
                return sv.Detections.empty()

            data = resp.json()
        except Exception as e:
            logger.error(f"RF-DETR service error: {e}")
            return sv.Detections.empty()

        if not data.get('xyxy'):
            return sv.Detections.empty()

        xyxy = np.array(data['xyxy'], dtype=np.float32)
        confidence = np.array(data['confidence'], dtype=np.float32)
        class_id = np.array(data['class_id'], dtype=int)

        # Decode masks if present
        masks = None
        if 'masks' in data and data['masks']:
            mask_list = []
            for m_b64 in data['masks']:
                m_bytes = base64.b64decode(m_b64)
                m_img = cv2.imdecode(np.frombuffer(m_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
                mask_list.append((m_img > 127).astype(bool))
            masks = np.array(mask_list)

        return sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
            mask=masks,
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
