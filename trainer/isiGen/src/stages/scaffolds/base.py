"""ScaffoldSource seam — Phase 6: procedural layouts for NEW synthetic images.

Each scaffold yields a PAIRED (control_map, ground_truth_mask, meta) — the
control map drives generation, the mask becomes the free label."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np

from ...core.registry import Registry

if TYPE_CHECKING:
    from ...core.project import ProjectConfig

SCAFFOLD_SOURCES: Registry[ScaffoldSource] = Registry("ScaffoldSource")


class ScaffoldSource(ABC):
    def __init__(self, **cfg) -> None:
        self.cfg = cfg

    @abstractmethod
    def generate(self, project: ProjectConfig, count: int
                 ) -> Iterator[tuple[np.ndarray, np.ndarray, dict]]:
        """Yield (control_map u8, color mask u8 HxWx3, meta) pairs."""
