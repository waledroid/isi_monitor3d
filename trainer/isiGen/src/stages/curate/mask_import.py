"""Import existing masks at curate time — skip SAM2 for pre-labeled images.

Three formats are auto-detected per image and rasterized into the same
color-coded ``maps/mask/<id>.png`` SAM2 would produce, so an imported record is
indistinguishable downstream (export, scaffolds) save for its
``mask_source="imported"`` tag:

- **LabelMe** — per-image sidecar ``<image>.json`` with ``shapes``
  (polygon/linestrip/rectangle), points in pixel coords scaled by the JSON's
  ``imageWidth/Height``.
- **YOLO** — per-image sidecar ``<image>.txt``, normalized; each line is
  ``cls x1 y1 x2 y2 …`` (polygon-seg) or ``cls cx cy w h`` (bbox → rectangle).
  Class index → name via a ``data.yaml``/``classes.txt`` at the ingest root if
  present, else the project's class order.
- **COCO** — one dataset-level ``*.json`` (``images``/``annotations``/
  ``categories``) at the ingest root, matched to images by ``file_name``;
  polygon ``segmentation`` (or ``bbox`` fallback; RLE is skipped).

All coexist with SAM2 per-record: images without any of these still flow to
phase 3.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ...core.project import ProjectConfig

logger = logging.getLogger(__name__)

Polys = dict[str, list[np.ndarray]]   # class_name -> list of (N,2) int32 pixel polygons


# --------------------------------------------------------------------------- #
# Shared rasterizer
# --------------------------------------------------------------------------- #

def _paint(project: ProjectConfig, class_polys: Polys,
           target_hw: tuple[int, int]) -> np.ndarray | None:
    """Paint per-class pixel polygons onto a BGR canvas in project-class order
    (later classes overwrite on overlap — parity with ``runners._composite_mask``).
    Polygons for labels not in the project are ignored. None if nothing painted."""
    h, w = int(target_hw[0]), int(target_hw[1])
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    painted = False
    for spec in project.classes:
        polys = class_polys.get(spec.name)
        if not polys:
            continue
        r, g, b = spec.color
        cv2.fillPoly(canvas, polys, (b, g, r))         # BGR, like _composite_mask
        painted = True
    return canvas if painted else None


def _add(out: Polys, name: str, pts: np.ndarray) -> None:
    out.setdefault(name, []).append(np.asarray(pts).round().astype(np.int32))


# --------------------------------------------------------------------------- #
# LabelMe (per-image .json with `shapes`)
# --------------------------------------------------------------------------- #

def sidecar_json(image_path: Path) -> Path | None:
    """The LabelMe JSON beside an image (same stem), if it exists."""
    j = Path(image_path).with_suffix(".json")
    return j if j.is_file() else None


def _shape_polygon(shape: dict) -> list | None:
    pts = shape.get("points") or []
    st = shape.get("shape_type", "polygon")
    if st == "rectangle" and len(pts) == 2:
        (x0, y0), (x1, y1) = pts
        return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    if st in ("polygon", "linestrip") and len(pts) >= 3:
        return pts
    return None


def labelme_polys(json_path: Path, target_hw: tuple[int, int]) -> Polys:
    data = json.loads(Path(json_path).read_text())
    h, w = int(target_hw[0]), int(target_hw[1])
    lw = int(data.get("imageWidth") or 0)
    lh = int(data.get("imageHeight") or 0)
    sx = w / lw if lw else 1.0
    sy = h / lh if lh else 1.0
    out: Polys = {}
    for shape in data.get("shapes", []):
        poly = _shape_polygon(shape)
        if poly is None:
            continue
        _add(out, shape.get("label"), np.array(poly, dtype=np.float64) * (sx, sy))
    return out


def labelme_to_mask(json_path: Path, project: ProjectConfig,
                    target_hw: tuple[int, int]) -> np.ndarray | None:
    """Back-compat entry point: LabelMe JSON → color-coded BGR mask (or None)."""
    return _paint(project, labelme_polys(json_path, target_hw), target_hw)


# --------------------------------------------------------------------------- #
# YOLO (per-image .txt, normalized)
# --------------------------------------------------------------------------- #

def yolo_polys(txt_path: Path, names: list[str], target_hw: tuple[int, int]) -> Polys:
    h, w = int(target_hw[0]), int(target_hw[1])
    out: Polys = {}
    for line in Path(txt_path).read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            idx = int(float(parts[0]))
            nums = [float(x) for x in parts[1:]]
        except ValueError:
            continue
        if idx < 0 or idx >= len(names):
            logger.warning("mask-import: %s — class index %d out of range; skipped",
                           Path(txt_path).name, idx)
            continue
        if len(nums) == 4:                                  # bbox cx cy w h
            cx, cy, bw, bh = nums
            x0, y0 = (cx - bw / 2) * w, (cy - bh / 2) * h
            x1, y1 = (cx + bw / 2) * w, (cy + bh / 2) * h
            pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        elif len(nums) >= 6 and len(nums) % 2 == 0:         # polygon-seg
            pts = np.array(nums).reshape(-1, 2) * (w, h)
        else:
            continue
        _add(out, names[idx], pts)
    return out


def load_yolo_names(source: Path) -> list[str] | None:
    """Class names from a `data.yaml` (`names:`) or `classes.txt` at the ingest
    root, used to map YOLO indices → names. None ⇒ fall back to project order."""
    dy = Path(source) / "data.yaml"
    if dy.is_file():
        import yaml
        names = (yaml.safe_load(dy.read_text()) or {}).get("names")
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names)]
        if names:
            return [str(n) for n in names]
    ct = Path(source) / "classes.txt"
    if ct.is_file():
        return [ln.strip() for ln in ct.read_text().splitlines() if ln.strip()]
    return None


# --------------------------------------------------------------------------- #
# COCO (one dataset-level .json, matched by file_name)
# --------------------------------------------------------------------------- #

class CocoIndex:
    """A loaded COCO dataset: image basename → annotations, mapped to polygons."""

    def __init__(self, data: dict) -> None:
        self.cat = {c["id"]: c["name"] for c in data.get("categories", [])}
        imgs = {im["id"]: im for im in data.get("images", [])}
        self.by_file: dict[str, dict] = {
            Path(im["file_name"]).name: {"w": im.get("width"), "h": im.get("height"),
                                         "anns": []}
            for im in data.get("images", [])}
        for ann in data.get("annotations", []):
            im = imgs.get(ann.get("image_id"))
            if im is not None:
                self.by_file[Path(im["file_name"]).name]["anns"].append(ann)

    def polys_for(self, filename: str, target_hw: tuple[int, int]) -> Polys:
        e = self.by_file.get(filename)
        if not e:
            return {}
        h, w = int(target_hw[0]), int(target_hw[1])
        sx = w / e["w"] if e["w"] else 1.0
        sy = h / e["h"] if e["h"] else 1.0
        out: Polys = {}
        for ann in e["anns"]:
            name = self.cat.get(ann.get("category_id"))
            if name is None:
                continue
            seg = ann.get("segmentation")
            polys: list[np.ndarray] = []
            if isinstance(seg, list):                       # polygon segmentation
                for s in seg:
                    if isinstance(s, list) and len(s) >= 6:
                        polys.append(np.array(s, dtype=np.float64).reshape(-1, 2))
            if not polys:                                   # bbox fallback (incl. RLE)
                bb = ann.get("bbox")
                if bb and len(bb) == 4:
                    x, y, bw, bh = bb
                    polys.append(np.array([[x, y], [x + bw, y],
                                           [x + bw, y + bh], [x, y + bh]],
                                          dtype=np.float64))
            for p in polys:
                _add(out, name, p * (sx, sy))
        return out


def load_coco(source: Path) -> CocoIndex | None:
    """First dataset-level COCO json at the ingest root (has images/annotations/
    categories), or None."""
    for jp in sorted(Path(source).glob("*.json")):
        try:
            data = json.loads(jp.read_text())
        except Exception:
            continue
        if isinstance(data, dict) and {"images", "annotations", "categories"} <= set(data):
            logger.info("mask-import: COCO annotations found: %s", jp.name)
            return CocoIndex(data)
    return None


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

@dataclass
class ImportContext:
    coco: CocoIndex | None
    yolo_names: list[str] | None


def prepare_source(source: Path) -> ImportContext:
    """Load folder-level annotation sources once before the ingest loop."""
    return ImportContext(coco=load_coco(source), yolo_names=load_yolo_names(source))


def import_mask(image_path: Path, project: ProjectConfig,
                target_hw: tuple[int, int], ctx: ImportContext) -> np.ndarray | None:
    """Build a color-coded mask for an image from whatever annotation is present.
    Precedence: COCO (dataset file) → LabelMe (.json) → YOLO (.txt). None if none
    apply or nothing valid was painted."""
    if ctx.coco is not None:
        polys = ctx.coco.polys_for(Path(image_path).name, target_hw)
        if polys:
            mask = _paint(project, polys, target_hw)
            if mask is not None:
                return mask
    j = Path(image_path).with_suffix(".json")
    if j.is_file():
        mask = _paint(project, labelme_polys(j, target_hw), target_hw)
        if mask is not None:
            return mask
    t = Path(image_path).with_suffix(".txt")
    if t.is_file():
        names = ctx.yolo_names or [c.name for c in project.classes]
        mask = _paint(project, yolo_polys(t, names, target_hw), target_hw)
        if mask is not None:
            return mask
    return None
