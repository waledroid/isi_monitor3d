"""Captioner seam — Phase 3's anti-bleed captions.

Each curated image gets a text caption pairing the class's unique TRIGGER word
with an exhaustive BACKGROUND description, so the LoRA binds the trigger to the
object — not to "concrete floor" (the background-bleed failure mode).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ...core.registry import Registry

if TYPE_CHECKING:
    from ...core.manifest import ManifestRecord
    from ...core.project import ProjectConfig

CAPTIONERS: Registry[Captioner] = Registry("Captioner")


class Captioner(ABC):
    def __init__(self, **cfg) -> None:
        self.cfg = cfg

    @abstractmethod
    def caption(self, record: ManifestRecord, project: ProjectConfig) -> str:
        """One caption line for this record."""
