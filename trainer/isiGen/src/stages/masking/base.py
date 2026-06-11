"""Masker seam — Phase 2's ground-truth side.

Produces per-class boolean masks from operator prompts (Studio clicks/boxes,
stored in the manifest) or automatically as a fallback. The COLOR compositing
into the final ground-truth PNG is NOT the masker's job — `core/runners.py`
paints class masks with the project class colors (classes order = paint order,
so later classes overwrite on overlap: cartons over palettes stays correct).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ...core.manifest import MaskPrompt
from ...core.registry import Registry

MASKERS: Registry[Masker] = Registry("Masker")


class Masker(ABC):
    """Segment objects in a BGR image. Heavy deps lazily in ``load()``."""

    def __init__(self, **cfg) -> None:
        self.cfg = cfg

    def load(self) -> None:  # noqa: B027 — optional hook
        """Acquire the model. Default no-op."""

    @abstractmethod
    def segment_prompted(self, image_bgr: np.ndarray,
                         prompts: list[MaskPrompt]) -> dict[str, np.ndarray]:
        """Operator prompts → {class_name: bool HxW mask} (union per class)."""

    @abstractmethod
    def segment_auto(self, image_bgr: np.ndarray) -> list[np.ndarray]:
        """Promptless fallback → unlabeled bool masks (largest-first)."""

    def close(self) -> None:  # noqa: B027 — optional hook
        """Release the model/VRAM. Default no-op."""
