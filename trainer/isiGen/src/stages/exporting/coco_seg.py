"""COCO instance-segmentation exporter — Phase 8b.

Color ground-truth mask → per-class contours → polygons → the COCO
instance-segmentation layout (detectron2 / mmdet-ready):

    out/coco_seg/images/{train,val}/<id>.jpg
    out/coco_seg/annotations/instances_{train,val}.json

Each polygon contour becomes one COCO annotation (mirrors the YOLO-seg exporter,
which writes one label line per contour). Category ids are 1-based
(``category_id = class_index + 1``, the COCO convention). Background negatives
are listed in ``images`` with no annotations. Split is the SAME stable record-id
hash as the YOLO-seg exporter, so the two formats agree on train/val.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from ...core import progress
from .base import DATASET_EXPORTERS, DatasetExporter
from .yolo_seg import _split_for, mask_to_polygons

if TYPE_CHECKING:
    from ...core.manifest import ManifestRecord
    from ...core.project import ProjectConfig


def _poly_bbox_area(poly: np.ndarray) -> tuple[list[float], float]:
    """COCO bbox [x, y, w, h] + polygon area (shoelace) from an (N,2) array."""
    xs, ys = poly[:, 0], poly[:, 1]
    x0, y0, x1, y1 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
    area = float(abs(cv2.contourArea(poly.astype(np.float32))))
    return [x0, y0, x1 - x0, y1 - y0], area


@DATASET_EXPORTERS.register("coco_seg")
class CocoSegExporter(DatasetExporter):
    def __init__(self, val_fraction: float = 0.1, min_area: float = 50.0, **cfg) -> None:
        super().__init__(val_fraction=val_fraction, min_area=min_area, **cfg)
        self.val_fraction = float(val_fraction)
        self.min_area = float(min_area)

    def export(self, project: ProjectConfig, records: list[ManifestRecord],
               out_dir: Path) -> Path:
        import shutil
        root = Path(out_dir) / "coco_seg"
        for split in ("train", "val"):
            (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "annotations").mkdir(parents=True, exist_ok=True)
        project_dir = Path(out_dir).parent

        categories = [{"id": i + 1, "name": spec.name, "supercategory": "object"}
                      for i, spec in enumerate(project.classes)]
        docs = {s: {"info": {"description": f"isiGen {project.name}"}, "licenses": [],
                    "images": [], "annotations": [], "categories": categories}
                for s in ("train", "val")}
        img_id = {"train": 0, "val": 0}
        ann_id = {"train": 0, "val": 0}

        for i, rec in enumerate(records, 1):
            progress.report(i, len(records), "export:coco_seg")
            if not rec.image:
                continue
            img_path = project_dir / rec.image
            if not img_path.exists():
                continue
            anns = []
            if rec.mask:
                mask_path = project_dir / rec.mask
                if not mask_path.exists():
                    continue
                mask = cv2.imread(str(mask_path))
                if mask is None:
                    continue
                h, w = mask.shape[:2]
                for idx, spec in enumerate(project.classes):
                    for poly in mask_to_polygons(mask, spec.color, min_area=self.min_area):
                        bbox, area = _poly_bbox_area(poly)
                        anns.append((idx + 1, poly.flatten().tolist(), bbox, area))
                if not anns:
                    continue                      # masked but no polygons → skip (quality)
            else:                                 # background negative → image, no anns
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                h, w = img.shape[:2]

            split = _split_for(rec.id, self.val_fraction)
            img_id[split] += 1
            iid = img_id[split]
            fname = f"{rec.id}{img_path.suffix}"
            shutil.copy2(img_path, root / "images" / split / fname)
            docs[split]["images"].append(
                {"id": iid, "file_name": fname, "width": w, "height": h})
            for cat_id, seg, bbox, area in anns:
                ann_id[split] += 1
                docs[split]["annotations"].append({
                    "id": ann_id[split], "image_id": iid, "category_id": cat_id,
                    "segmentation": [seg], "area": area, "bbox": bbox, "iscrowd": 0,
                })

        for split in ("train", "val"):
            (root / "annotations" / f"instances_{split}.json").write_text(
                json.dumps(docs[split]))
        return root
