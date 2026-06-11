"""DatasetExporter seam — Phase 8b: package (generated image, ground-truth mask)
pairs into a training-ready dataset (YOLO-seg, LabelMe, ...)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from ...core.registry import Registry

if TYPE_CHECKING:
    from ...core.manifest import ManifestRecord
    from ...core.project import ProjectConfig

DATASET_EXPORTERS: Registry[DatasetExporter] = Registry("DatasetExporter")


class DatasetExporter(ABC):
    def __init__(self, **cfg) -> None:
        self.cfg = cfg

    @abstractmethod
    def export(self, project: ProjectConfig, records: list[ManifestRecord],
               out_dir: Path) -> Path:
        """Write the dataset; return its root directory."""
