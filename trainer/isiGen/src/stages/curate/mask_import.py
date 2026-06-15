"""Import existing LabelMe masks at curate time — skip SAM2 for pre-labeled images.

A sidecar ``<image>.json`` (LabelMe format) next to an image is rasterized into
the same color-coded ``maps/mask/<id>.png`` SAM2 would produce, so an imported
record is indistinguishable downstream (export, scaffolds) save for its
``mask_source="imported"`` tag. Coexists with SAM2 per-record: images without a
sidecar JSON still flow to phase 3.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np

from ...core.project import ProjectConfig

logger = logging.getLogger(__name__)


def sidecar_json(image_path: Path) -> Path | None:
    """The LabelMe JSON beside an image (same stem), if it exists."""
    j = Path(image_path).with_suffix(".json")
    return j if j.is_file() else None


def _shape_polygon(shape: dict) -> list | None:
    """LabelMe shape → polygon point list, or None if unsupported."""
    pts = shape.get("points") or []
    st = shape.get("shape_type", "polygon")
    if st == "rectangle" and len(pts) == 2:
        (x0, y0), (x1, y1) = pts
        return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    if st in ("polygon", "linestrip") and len(pts) >= 3:
        return pts
    return None


def labelme_to_mask(json_path: Path, project: ProjectConfig,
                    target_hw: tuple[int, int]) -> np.ndarray | None:
    """Rasterize a LabelMe JSON into a color-coded BGR mask at ``target_hw``.

    Each shape's ``label`` must be a project class (others are skipped + warned).
    Points are scaled from the JSON's ``imageWidth/Height`` to the target dims.
    Painted in project-class order so later classes overwrite on overlap (parity
    with ``runners._composite_mask``). Returns the mask, or None if nothing
    valid was painted.
    """
    data = json.loads(Path(json_path).read_text())
    h, w = int(target_hw[0]), int(target_hw[1])
    lw = int(data.get("imageWidth") or 0)
    lh = int(data.get("imageHeight") or 0)
    sx = w / lw if lw else 1.0
    sy = h / lh if lh else 1.0

    known = {c.name for c in project.classes}
    by_class: dict[str, list[np.ndarray]] = {}
    for shape in data.get("shapes", []):
        label = shape.get("label")
        if label not in known:
            logger.warning("mask-import: %s — label %r is not a project class; skipped",
                           Path(json_path).name, label)
            continue
        poly = _shape_polygon(shape)
        if poly is None:
            logger.warning("mask-import: %s — unsupported shape_type %r; skipped",
                           Path(json_path).name, shape.get("shape_type"))
            continue
        arr = (np.array(poly, dtype=np.float64) * (sx, sy)).round().astype(np.int32)
        by_class.setdefault(label, []).append(arr)

    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    painted = False
    for spec in project.classes:                       # project order = paint order
        polys = by_class.get(spec.name)
        if not polys:
            continue
        r, g, b = spec.color
        cv2.fillPoly(canvas, polys, (b, g, r))         # BGR, like _composite_mask
        painted = True
    return canvas if painted else None
