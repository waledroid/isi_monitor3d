"""LoraTrainer seam — Phase 4: train the SDXL LoRA (the project's
"material and texture dictionary") on the curated images + captions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from ...core.registry import Registry

if TYPE_CHECKING:
    from ...core.project import ProjectConfig

LORA_TRAINERS: Registry[LoraTrainer] = Registry("LoraTrainer")


class LoraTrainer(ABC):
    def __init__(self, **cfg) -> None:
        self.cfg = cfg

    @abstractmethod
    def train(self, project: ProjectConfig, run_dir: Path) -> Path:
        """Train; return the LoRA weights path (safetensors)."""
