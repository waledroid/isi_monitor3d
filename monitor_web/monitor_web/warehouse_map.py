"""Warehouse layout twin — load / validate / write ``warehouse_map.yaml``.

A consumer-side config (sibling to ``zones.yaml``) describing the *static*
structure of the floor: racks, walls, obstacles as metric floor-contact
footprints plus a height for the 2.5D render. No Backbone import.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import yaml

ELEMENT_TYPES = {"rack", "wall", "obstacle"}
SHAPES = {"rectangle", "polygon"}


def _validate_footprint(fp, where: str) -> list[list[float]]:
    if not isinstance(fp, list) or len(fp) < 3:
        raise ValueError(f"{where}: footprint must be a polygon of >=3 [x, y] points")
    out = []
    for pt in fp:
        if not (isinstance(pt, (list, tuple)) and len(pt) == 2):
            raise ValueError(f"{where}: each footprint point must be [x, y] metres")
        out.append([float(pt[0]), float(pt[1])])
    return out


def validate_map(data: dict) -> dict:
    """Normalise + validate a layout dict. Raises ``ValueError`` on bad input."""
    elements = []
    for i, el in enumerate(data.get("elements") or []):
        etype = el.get("type")
        if etype not in ELEMENT_TYPES:
            raise ValueError(f"element[{i}]: type must be one of {sorted(ELEMENT_TYPES)}")
        shape = el.get("shape", "polygon")
        if shape not in SHAPES:
            raise ValueError(f"element[{i}]: shape must be one of {sorted(SHAPES)}")
        try:
            height = float(el.get("height_m", 0.0))
        except (TypeError, ValueError):
            raise ValueError(f"element[{i}]: height_m must be a number") from None
        # Rack shelving levels (3D twin) — clamped 1..8; meaningful for type=rack.
        try:
            levels = max(1, min(8, int(el.get("levels", 3))))
        except (TypeError, ValueError):
            levels = 3
        try:
            rotation_deg = float(el.get("rotation_deg", 0.0))
        except (TypeError, ValueError):
            rotation_deg = 0.0
        elements.append({
            "id": str(el.get("id") or f"el_{i}"),
            "type": etype,
            "shape": shape,
            "footprint": _validate_footprint(el.get("footprint"), f"element[{i}]"),
            "height_m": height,
            "levels": levels,
            "rotation_deg": rotation_deg,
            "label": str(el.get("label") or ""),
        })
    outline = None
    raw_outline = data.get("outline")
    if raw_outline and raw_outline.get("footprint"):
        outline = {"footprint": _validate_footprint(raw_outline["footprint"], "outline")}
    return {"elements": elements, "outline": outline}


def read_map(path: Path) -> dict:
    """Load + validate the layout YAML. Missing/empty file → empty layout."""
    path = Path(path)
    if not path.exists():
        return {"elements": [], "outline": None}
    raw = yaml.safe_load(path.read_text()) or {}
    return validate_map(raw)


def write_map(path: Path, data: dict) -> None:
    """Validate then atomically write the layout YAML (tempfile + os.replace)."""
    validated = validate_map(data)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            yaml.safe_dump(validated, f, sort_keys=False, allow_unicode=True)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
