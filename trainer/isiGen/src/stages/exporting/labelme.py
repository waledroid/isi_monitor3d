"""LabelMe exporter — Phase 8b.

Color ground-truth mask → polygon shapes → one LabelMe JSON next to each copied
image (flat folder, X-AnyLabeling-ready) for human review before training.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import cv2

from ...core import progress
from .base import DATASET_EXPORTERS, DatasetExporter
from .yolo_seg import mask_to_polygons

if TYPE_CHECKING:
    from ...core.manifest import ManifestRecord
    from ...core.project import ProjectConfig


@DATASET_EXPORTERS.register("labelme")
class LabelmeExporter(DatasetExporter):
    def __init__(self, min_area: float = 50.0, **cfg) -> None:
        super().__init__(min_area=min_area, **cfg)
        self.min_area = float(min_area)

    def export(self, project: ProjectConfig, records: list[ManifestRecord],
               out_dir: Path) -> Path:
        root = Path(out_dir) / "labelme"
        root.mkdir(parents=True, exist_ok=True)
        project_dir = Path(out_dir).parent
        for i, rec in enumerate(records, 1):
            progress.report(i, len(records), "export:labelme")
            if not rec.image:
                continue
            img_path = project_dir / rec.image
            if not img_path.exists():
                continue
            shapes = []
            if rec.mask:
                mask_path = project_dir / rec.mask
                if not mask_path.exists():
                    continue
                mask = cv2.imread(str(mask_path))
                if mask is None:
                    continue
                h, w = mask.shape[:2]
                for spec in project.classes:
                    for poly in mask_to_polygons(mask, spec.color, min_area=self.min_area):
                        shapes.append({
                            "label": spec.name,
                            "points": [[float(x), float(y)] for x, y in poly],
                            "group_id": None,
                            "shape_type": "polygon",
                            "flags": {},
                        })
                if not shapes:
                    continue                      # masked but no polygons → skip (quality)
            else:
                # background negative (no mask) → keep with empty shapes
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                h, w = img.shape[:2]
            img_name = f"{rec.id}{img_path.suffix}"
            shutil.copy2(img_path, root / img_name)
            (root / f"{rec.id}.json").write_text(json.dumps({
                "version": "5.4.1",
                "flags": {},
                "shapes": shapes,
                "imagePath": img_name,
                "imageData": None,
                "imageHeight": h,
                "imageWidth": w,
            }, indent=2))
        return root
