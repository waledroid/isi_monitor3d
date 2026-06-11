"""DepthAnythingV2 depth control map — via transformers' depth-estimation
pipeline (no extra package; ~100 MB checkpoint auto-cached to HF_HOME).

The min-max-normalized uint8 grayscale PNG this produces is exactly the input
format the SD3.5-Large depth ControlNet expects in phase 7.
"""

from __future__ import annotations

import numpy as np

from .base import CONTROL_MAP_EXTRACTORS, ControlMapExtractor


@CONTROL_MAP_EXTRACTORS.register("depth_anything_v2")
class DepthAnythingV2Extractor(ControlMapExtractor):
    map_name = "depth"

    def __init__(self, model_id: str = "depth-anything/Depth-Anything-V2-Small-hf",
                 device: str = "cuda", **cfg) -> None:
        super().__init__(model_id=model_id, device=device, **cfg)
        self.model_id = model_id
        self.device = device
        self._pipe = None

    def load(self) -> None:
        import torch  # lazy — heavy
        from transformers import pipeline
        device = self.device if (self.device != "cuda" or torch.cuda.is_available()) else "cpu"
        self._pipe = pipeline("depth-estimation", model=self.model_id, device=device)

    def extract(self, image_bgr: np.ndarray) -> np.ndarray:
        if self._pipe is None:
            self.load()
        from PIL import Image  # lazy
        rgb = image_bgr[:, :, ::-1]
        result = self._pipe(Image.fromarray(rgb))
        depth = np.asarray(result["depth"], dtype=np.float32)
        lo, hi = float(depth.min()), float(depth.max())
        if hi - lo < 1e-6:
            return np.zeros(depth.shape, dtype=np.uint8)
        return ((depth - lo) / (hi - lo) * 255.0).astype(np.uint8)

    def close(self) -> None:
        self._pipe = None
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
