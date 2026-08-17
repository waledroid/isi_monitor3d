"""Étagère (bin-rack) zone configuration — shared by isistream, the Backbone
and monitor_web (all may import backbone.shared).

An étagère zone is a per-camera IMAGE-SPACE grid of cells (rows x cols
axis-aligned rectangles in source-frame pixels), NOT a floor polygon — it is
deliberately kept out of zones.yaml / zone_patches so it never enters the
floor-projection pipeline. Authored by the dashboard Settings (4 corners →
auto-split → per-cell drag-adjust), consumed by isistream for per-cell crop
inference. Missing file ⇒ feature off.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_FILENAME = "etagere.yaml"


class EtagereCell(BaseModel):
    model_config = ConfigDict(extra="forbid")
    r: int = Field(..., ge=1)
    c: int = Field(..., ge=1)
    rect: tuple[float, float, float, float]   # x0, y0, x1, y1 (source px)

    @model_validator(mode="after")
    def _ordered(self) -> "EtagereCell":  # noqa: UP037
        x0, y0, x1, y1 = self.rect
        if not (x1 > x0 and y1 > y0):
            raise ValueError(f"cell r{self.r}c{self.c}: rect must be x1>x0, y1>y0")
        return self


class EtagereZone(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    name: str = ""
    camera: str
    frame_wh: tuple[int, int]
    corners: tuple[tuple[float, float], ...] = ()
    rows: int = Field(3, ge=1)
    cols: int = Field(3, ge=1)
    cells: tuple[EtagereCell, ...]
    max_fps: float | None = None

    @model_validator(mode="after")
    def _grid(self) -> "EtagereZone":  # noqa: UP037
        if self.corners and len(self.corners) != 4:
            raise ValueError("corners must be empty or exactly 4 points (TL,TR,BR,BL)")
        expect = [(r, c) for r in range(1, self.rows + 1) for c in range(1, self.cols + 1)]
        got = [(cell.r, cell.c) for cell in self.cells]
        if got != expect:
            raise ValueError(
                f"zone {self.id!r}: cells must be exactly rows*cols in reading order "
                f"(expected {expect[:3]}…, got {got[:3]}…)")
        return self


class EtagereModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    onnx_path: str
    class_names: list[str] = ["empty_box", "filled_box"]
    imgsz: int = Field(320, ge=64)
    confidence_threshold: float = Field(0.3, ge=0.0, le=1.0)
    crop_margin: float = Field(0.08, ge=0.0, le=0.5)
    max_fps: float = Field(2.0, gt=0.0)
    providers: str | None = None


class EtagereConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: EtagereModel | None = None
    zones: tuple[EtagereZone, ...] = ()

    @property
    def enabled(self) -> bool:
        return self.model is not None and len(self.zones) > 0


def load_etagere_config(path: str | Path | None) -> EtagereConfig:
    """Load ``etagere.yaml``; a missing/None path is the disabled config."""
    if path is None:
        return EtagereConfig()
    p = Path(path)
    if not p.exists():
        return EtagereConfig()
    data = yaml.safe_load(p.read_text()) or {}
    return EtagereConfig.model_validate(data)


def resolve_config_path(backbone_cfg: dict, backbone_yaml_path: str | Path) -> Path:
    """``backbone.yaml``'s ``etagere.config_path`` or ``<its dir>/etagere.yaml``."""
    explicit = (backbone_cfg.get("etagere") or {}).get("config_path")
    if explicit:
        return Path(explicit)
    return Path(backbone_yaml_path).parent / DEFAULT_FILENAME


def cells_from_corners(corners, rows: int = 3, cols: int = 3) -> list[EtagereCell]:
    """Auto-split an outer quad (TL,TR,BR,BL) into rows*cols cells.

    Bilinear interpolation of the quad at the grid fractions gives each cell's
    4 corners; the cell rect is that quad's axis-aligned bounding box (crops
    are rectangles). Reading order: r1c1, r1c2, …
    """
    (tlx, tly), (trx, try_), (brx, bry), (blx, bly) = [(float(x), float(y)) for x, y in corners]

    def pt(u: float, v: float) -> tuple[float, float]:
        topx, topy = tlx * (1 - u) + trx * u, tly * (1 - u) + try_ * u
        botx, boty = blx * (1 - u) + brx * u, bly * (1 - u) + bry * u
        return topx * (1 - v) + botx * v, topy * (1 - v) + boty * v

    out: list[EtagereCell] = []
    for r in range(rows):
        for c in range(cols):
            us = (c / cols, (c + 1) / cols)
            vs = (r / rows, (r + 1) / rows)
            quad = [pt(us[0], vs[0]), pt(us[1], vs[0]), pt(us[1], vs[1]), pt(us[0], vs[1])]
            xs = [p[0] for p in quad]
            ys = [p[1] for p in quad]
            out.append(EtagereCell(r=r + 1, c=c + 1,
                                   rect=(min(xs), min(ys), max(xs), max(ys))))
    return out
