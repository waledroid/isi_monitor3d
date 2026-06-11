"""CLIP-score filter — Phase 8a.

Cosine similarity between the generated image's CLIP embedding and its prompt's
text embedding (openai/clip-vit-base-patch32, ~600 MB). Low score = the model
hallucinated away from the prompt (mangled geometry, wrong object) → discard.
"""

from __future__ import annotations

import numpy as np

from .base import QUALITY_FILTERS, QualityFilter


@QUALITY_FILTERS.register("clip_score")
class ClipScoreFilter(QualityFilter):
    def __init__(self, model_id: str = "openai/clip-vit-base-patch32",
                 device: str = "cuda", **cfg) -> None:
        super().__init__(model_id=model_id, device=device, **cfg)
        self.model_id = model_id
        self.device = device
        self._model = None
        self._processor = None

    def load(self) -> None:
        import torch  # lazy — heavy
        from transformers import CLIPModel, CLIPProcessor
        device = self.device if (self.device != "cuda" or torch.cuda.is_available()) else "cpu"
        self._model = CLIPModel.from_pretrained(self.model_id).to(device).eval()
        self._processor = CLIPProcessor.from_pretrained(self.model_id)
        self._device = device

    def score(self, image: np.ndarray, prompt: str) -> float:
        if self._model is None:
            self.load()
        import torch
        from PIL import Image
        rgb = Image.fromarray(np.ascontiguousarray(image[:, :, ::-1]))
        inputs = self._processor(text=[prompt], images=[rgb], return_tensors="pt",
                                 padding=True, truncation=True, max_length=77)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model(**inputs)
            img = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
            txt = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
            return float((img * txt).sum())

    def close(self) -> None:
        self._model = None
        self._processor = None
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
