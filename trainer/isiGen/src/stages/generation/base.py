"""ImageGenerator seam — Phases 5+7: pipeline init (load()) + minting (generate()).

The ControlNet forces the geometry from the scaffold's control map; the base
model hallucinates the background from the prompt; the project LoRA applies the
correct object textures."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ...core.registry import Registry

IMAGE_GENERATORS: Registry[ImageGenerator] = Registry("ImageGenerator")


class ImageGenerator(ABC):
    def __init__(self, **cfg) -> None:
        self.cfg = cfg

    @abstractmethod
    def load(self) -> None:
        """Phase 5 lives here: build the pipeline (base + ControlNet + LoRA)."""

    @abstractmethod
    def generate(self, prompt: str, control_image: np.ndarray, *,
                 seed: int = -1, **params) -> np.ndarray:
        """One synthetic image (BGR uint8) from prompt + control map."""

    def close(self) -> None:  # noqa: B027 — optional hook
        """Release the pipeline/VRAM. Default no-op."""
