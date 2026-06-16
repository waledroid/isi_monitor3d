"""BLIP captioner — image-aware captions (vs the template's random-background bank).

Runs BLIP on each curated image to describe what's *actually* there, then
prepends the class TRIGGER so the LoRA still binds identity to the trigger word
(anti-bleed). Slower than the template (a model + per-image inference), so the
captions phase shows a live progress bar.

Config (project.yaml ``phases.captioning.blip``): ``model_id`` (default
``Salesforce/blip-image-captioning-base`` — ungated, ~990 MB), ``device``,
``max_new_tokens``, and ``template`` (how to combine trigger + description;
default ``"{trigger} {description}"``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from .base import CAPTIONERS, Captioner

if TYPE_CHECKING:
    from ...core.manifest import ManifestRecord
    from ...core.project import ProjectConfig

logger = logging.getLogger(__name__)


@CAPTIONERS.register("blip")
class BlipCaptioner(Captioner):
    def __init__(self, model_id: str = "Salesforce/blip-image-captioning-base",
                 device: str = "cuda", project_dir: str = ".",
                 max_new_tokens: int = 40,
                 template: str = "{trigger} {description}", **cfg) -> None:
        super().__init__(model_id=model_id, device=device, project_dir=project_dir,
                         max_new_tokens=max_new_tokens, template=template, **cfg)
        self.model_id = model_id
        self.device = device
        self.project_dir = Path(project_dir)
        self.max_new_tokens = int(max_new_tokens)
        self.template = template
        self._model = None
        self._proc = None

    def load(self) -> None:
        import torch  # lazy — heavy
        from transformers import BlipForConditionalGeneration, BlipProcessor
        device = self.device if (self.device != "cuda" or torch.cuda.is_available()) else "cpu"
        self.device = device
        logger.info("captions[blip]: loading %s on %s", self.model_id, device)
        self._proc = BlipProcessor.from_pretrained(self.model_id)
        self._model = BlipForConditionalGeneration.from_pretrained(self.model_id).to(device)

    def caption(self, record: ManifestRecord, project: ProjectConfig) -> str:
        if self._model is None:
            self.load()
        import torch
        from PIL import Image
        spec = project.class_by_name(record.class_name)
        img = Image.open(self.project_dir / record.image).convert("RGB")
        inputs = self._proc(img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        desc = self._proc.decode(out[0], skip_special_tokens=True).strip()
        return self.template.format(trigger=spec.trigger, description=desc,
                                    class_name=record.class_name).strip()
