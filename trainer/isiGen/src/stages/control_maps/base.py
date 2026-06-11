"""ControlMapExtractor seam — Phase 2's generation-side maps.

A control map is what the SD3.5 ControlNet consumes to force geometry: a depth
map (DepthAnythingV2) or Canny edges. One extractor per map kind; the maps
runner calls ``load()`` once, ``extract()`` per image, ``close()`` at the end.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ...core.registry import Registry

CONTROL_MAP_EXTRACTORS: Registry[ControlMapExtractor] = Registry("ControlMapExtractor")


class ControlMapExtractor(ABC):
    """Extracts one control map from a BGR image.

    ``map_name`` is the manifest field family it fills ("depth" | "canny").
    Heavy deps (torch/transformers) must be imported lazily in ``load()`` —
    never at module top — so the package imports in a deps-free test env.
    """

    map_name: str = ""

    def __init__(self, **cfg) -> None:
        self.cfg = cfg

    def load(self) -> None:  # noqa: B027 — optional hook
        """Acquire models/VRAM. Default no-op (pure-cv2 extractors)."""

    @abstractmethod
    def extract(self, image_bgr: np.ndarray) -> np.ndarray:
        """BGR image → uint8 map (HxW grayscale or HxWx3), same spatial size."""

    def close(self) -> None:  # noqa: B027 — optional hook
        """Release models/VRAM. Default no-op."""
