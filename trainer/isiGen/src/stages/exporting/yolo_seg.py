"""YOLO-seg exporter — Phase 8b.

Color ground-truth mask → per-class contours → normalized polygons → the YOLO
segmentation layout isidet trains on:

    out/yolo_seg/images/{train,val}/<id>.jpg
    out/yolo_seg/labels/{train,val}/<id>.txt    # "cls x1 y1 x2 y2 ..." normalized
    out/yolo_seg/data.yaml                      # nc / names / path

Split is a STABLE hash of the record id (re-exports keep images in the same
split). Polygons are simplified (~0.2 % of the perimeter) and tiny specks
(< min_area px) dropped.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import yaml

from ...core import progress
from .base import DATASET_EXPORTERS, DatasetExporter

if TYPE_CHECKING:
    from ...core.manifest import ManifestRecord
    from ...core.project import ProjectConfig


def _split_for(record_id: str, val_fraction: float) -> str:
    h = int(hashlib.sha256(record_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "val" if h < val_fraction else "train"


def mask_to_polygons(mask_bgr: np.ndarray, color_rgb: list[int], *,
                     min_area: float = 50.0, epsilon_frac: float = 0.002
                     ) -> list[np.ndarray]:
    """Binary-select one class color from the painted mask → simplified contours
    (each an (N,2) float array in pixel coords)."""
    r, g, b = color_rgb
    binary = np.all(mask_bgr == (b, g, r), axis=2).astype(np.uint8)
    if not binary.any():
        return []
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        eps = epsilon_frac * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(np.float64)
        if len(approx) >= 3:
            polys.append(approx)
    return polys


@DATASET_EXPORTERS.register("yolo_seg")
class YoloSegExporter(DatasetExporter):
    def __init__(self, val_fraction: float = 0.1, min_area: float = 50.0, **cfg) -> None:
        super().__init__(val_fraction=val_fraction, min_area=min_area, **cfg)
        self.val_fraction = float(val_fraction)
        self.min_area = float(min_area)

    def export(self, project: ProjectConfig, records: list[ManifestRecord],
               out_dir: Path) -> Path:
        root = Path(out_dir) / "yolo_seg"
        for split in ("train", "val"):
            (root / "images" / split).mkdir(parents=True, exist_ok=True)
            (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        names = project.class_names()
        exported = 0
        # records carry paths relative to the PROJECT dir = out_dir's parent by
        # convention (export/ lives inside the project); resolve against it.
        project_dir = Path(out_dir).parent
        for i, rec in enumerate(records, 1):
            progress.report(i, len(records), "export:yolo_seg")
            if not rec.image:
                continue
            img_path = project_dir / rec.image
            if not img_path.exists():
                continue
            lines: list[str] = []
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
                        norm = poly / np.array([w, h], dtype=np.float64)
                        coords = " ".join(f"{v:.6f}" for v in norm.clip(0, 1).flatten())
                        lines.append(f"{idx} {coords}")
                if not lines:
                    continue                      # masked but no polygons → skip (quality)
            # else: background negative → empty .txt (YOLO treats it as a negative)
            split = _split_for(rec.id, self.val_fraction)
            dst_img = root / "images" / split / f"{rec.id}{img_path.suffix}"
            shutil.copy2(img_path, dst_img)
            (root / "labels" / split / f"{rec.id}.txt").write_text("\n".join(lines) + "\n")
            exported += 1
        (root / "data.yaml").write_text(yaml.safe_dump({
            "path": ".", "train": "images/train", "val": "images/val",
            "nc": len(names), "names": names,
        }, sort_keys=False))
        return root
