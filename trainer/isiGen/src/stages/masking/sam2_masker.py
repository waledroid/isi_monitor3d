"""SAM2 masker — Meta's Segment Anything 2 from the HF hub.

`sam2.1-hiera-small` (~185 MB) fits the 12 GB card trivially;
`sam2.1-hiera-base-plus` is a config-only upgrade if small's masks disappoint.

Prompted path: Studio click/box prompts (grouped by class) → one predict per
class. Auto path: SAM2AutomaticMaskGenerator for promptless records.
"""

from __future__ import annotations

import numpy as np

from ...core.manifest import MaskPrompt
from .base import MASKERS, Masker


@MASKERS.register("sam2")
class Sam2Masker(Masker):
    def __init__(self, model_id: str = "facebook/sam2.1-hiera-small",
                 device: str = "cuda", multimask_output: bool = False,
                 fallback_auto: bool = True, **cfg) -> None:
        super().__init__(model_id=model_id, device=device,
                         multimask_output=multimask_output,
                         fallback_auto=fallback_auto, **cfg)
        self.model_id = model_id
        self.device = device
        self.multimask_output = bool(multimask_output)
        self.fallback_auto = bool(fallback_auto)
        self._predictor = None
        self._auto = None

    def load(self) -> None:
        import torch  # lazy — heavy
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        device = self.device if (self.device != "cuda" or torch.cuda.is_available()) else "cpu"
        self._predictor = SAM2ImagePredictor.from_pretrained(self.model_id, device=device)

    def segment_prompted(self, image_bgr: np.ndarray,
                         prompts: list[MaskPrompt]) -> dict[str, np.ndarray]:
        if self._predictor is None:
            self.load()
        h, w = image_bgr.shape[:2]
        self._predictor.set_image(np.ascontiguousarray(image_bgr[:, :, ::-1]))  # SAM2 wants RGB (contiguous: torch rejects negative strides)
        out: dict[str, np.ndarray] = {}
        by_class: dict[str, list[MaskPrompt]] = {}
        for p in prompts:
            by_class.setdefault(p.class_name, []).append(p)
        for cls_name, plist in by_class.items():
            points = [p.xy for p in plist if p.kind == "point" and p.xy]
            labels = [p.label for p in plist if p.kind == "point" and p.xy]
            boxes = [p.xyxy for p in plist if p.kind == "box" and p.xyxy]
            union = np.zeros((h, w), dtype=bool)
            # SAM2 takes one box per predict; points can accompany a box or
            # stand alone. Predict per box (with all points), else points only.
            if boxes:
                for box in boxes:
                    masks, _, _ = self._predictor.predict(
                        point_coords=np.array(points, dtype=np.float32) if points else None,
                        point_labels=np.array(labels, dtype=np.int32) if points else None,
                        box=np.array(box, dtype=np.float32),
                        multimask_output=self.multimask_output,
                    )
                    union |= masks[0].astype(bool)
            elif points:
                masks, _, _ = self._predictor.predict(
                    point_coords=np.array(points, dtype=np.float32),
                    point_labels=np.array(labels, dtype=np.int32),
                    multimask_output=self.multimask_output,
                )
                union |= masks[0].astype(bool)
            if union.any():
                out[cls_name] = union
        return out

    def segment_auto(self, image_bgr: np.ndarray) -> list[np.ndarray]:
        if self._auto is None:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            if self._predictor is None:
                self.load()
            assert isinstance(self._predictor, SAM2ImagePredictor)
            self._auto = SAM2AutomaticMaskGenerator(self._predictor.model)
        results = self._auto.generate(np.ascontiguousarray(image_bgr[:, :, ::-1]))
        results.sort(key=lambda r: int(r.get("area", 0)), reverse=True)
        return [r["segmentation"].astype(bool) for r in results]

    def close(self) -> None:
        self._predictor = None
        self._auto = None
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
