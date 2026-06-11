"""QualityFilter seam — Phase 8a: score generated images against their prompt;
discard hallucinations/mangled generations below the threshold."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ...core.registry import Registry

QUALITY_FILTERS: Registry[QualityFilter] = Registry("QualityFilter")


class QualityFilter(ABC):
    def __init__(self, **cfg) -> None:
        self.cfg = cfg

    def load(self) -> None:  # noqa: B027 — optional hook
        """Acquire the model. Default no-op."""

    def close(self) -> None:  # noqa: B027 — optional hook
        """Release the model/VRAM. Default no-op."""

    @abstractmethod
    def score(self, image: np.ndarray, prompt: str) -> float:
        """Higher = better prompt/image agreement."""
