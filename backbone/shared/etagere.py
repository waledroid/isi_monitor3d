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
    # Rotation of the cell rectangle about its own centre, degrees, positive =
    # clockwise on screen (y down). isistream warps the cell UPRIGHT by this
    # angle before cropping, so tilted racks still yield training-like crops.
    angle_deg: float = Field(0.0, ge=-180.0, le=180.0)

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
        if self.frame_wh[0] < 1 or self.frame_wh[1] < 1:
            raise ValueError(
                f"zone {self.id!r}: frame_wh must be >= 1x1 (got {self.frame_wh}) — "
                "a 0x0 frame_wh means the camera's frame size wasn't known when the "
                "zone was drawn and would ZeroDivisionError every crop at inference time")
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

    @model_validator(mode="after")
    def _decide_labels_present(self) -> "EtagereModel":  # noqa: UP037
        # isistream.etagere.decide() hardcodes these two label names — a
        # differently-labelled model would silently classify every cell
        # "unknown" forever. Fail loudly at config load instead.
        missing = {"empty_box", "filled_box"} - set(self.class_names)
        if missing:
            raise ValueError(
                f"class_names must include {sorted(missing)} — decide() matches "
                "on these exact labels; a model without them would leave every "
                "cell 'unknown'")
        return self


class EtagereConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: EtagereModel | None = None
    zones: tuple[EtagereZone, ...] = ()
    # Operator switch: turn étagère DETECTION off while keeping the zones and
    # model configured (mirrors detection.pose_enabled / object_enabled).
    detection_enabled: bool = True

    @property
    def enabled(self) -> bool:
        return self.detection_enabled and self.model is not None and len(self.zones) > 0


def _resolve_model_path(cfg: EtagereConfig, yaml_path: str | Path) -> EtagereConfig:
    """Resolve a relative ``model.onnx_path`` against the REPO ROOT — the
    parent of the yaml's own directory (``config/etagere.yaml`` → repo root is
    ``yaml_path.parent.parent``, matching ``backbone.yaml``'s convention).

    isistream is spawned by the dashboard (cwd inherited, usually
    ``monitor_web``) or by systemd (cwd ``/``) — a relative ``onnx_path``
    resolves against neither, so ORT fails to find the file and the failure
    is swallowed upstream, leaving every cell silently ``unknown``. Absolute
    paths pass through untouched.
    """
    if cfg.model is None:
        return cfg
    onnx = Path(cfg.model.onnx_path)
    if onnx.is_absolute():
        return cfg
    repo_root = Path(yaml_path).resolve().parent.parent
    resolved = str((repo_root / onnx).resolve())
    return cfg.model_copy(update={"model": cfg.model.model_copy(update={"onnx_path": resolved})})


def load_etagere_config(path: str | Path | None) -> EtagereConfig:
    """Load ``etagere.yaml``; a missing/None path is the disabled config."""
    if path is None:
        return EtagereConfig()
    p = Path(path)
    if not p.exists():
        return EtagereConfig()
    data = yaml.safe_load(p.read_text()) or {}
    cfg = EtagereConfig.model_validate(data)
    return _resolve_model_path(cfg, p)


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


# --- pixel helpers shared by isistream (inference) and monitor_web (panel) ---
# ONE implementation of "what a cell crop looks like", so the dashboard's
# étagère panel shows exactly the pixels the model sees.

def crop_cell(image, frame_wh, rect, angle_deg: float = 0.0, margin: float = 0.08):
    """Crop one cell out of ``image`` (H, W, 3).

    ``rect`` (x0, y0, x1, y1) is in the zone's declared ``frame_wh`` pixel
    space and is scaled to the actual image; the crop is padded by ``margin``
    (fraction of the scaled rect's side). With ``angle_deg`` != 0 the cell is
    a rectangle rotated about its own centre (positive = clockwise on screen);
    the image is warped by the inverse rotation about that centre first, so
    the crop is the cell's content UPRIGHT (the axis-aligned framing the
    model was trained on). Returns ``None`` for a degenerate crop.
    """
    import cv2  # local import: keep the config models importable without OpenCV
    h, w = image.shape[:2]
    sx = w / float(frame_wh[0])
    sy = h / float(frame_wh[1])
    x0, y0, x1, y1 = rect[0] * sx, rect[1] * sy, rect[2] * sx, rect[3] * sy
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    if abs(angle_deg) < 1e-6:
        cx0, cy0 = max(int(x0 - mx), 0), max(int(y0 - my), 0)
        cx1, cy1 = min(int(x1 + mx), w), min(int(y1 + my), h)
        if cx1 - cx0 < 4 or cy1 - cy0 < 4:
            return None
        return image[cy0:cy1, cx0:cx1]
    # cv2's positive angle is counter-clockwise on a y-down image, so a
    # clockwise-tilted cell (angle_deg > 0) is uprighted by rotating the image
    # by +angle_deg about the cell centre.
    ccx, ccy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    cw, ch = round((x1 - x0) + 2 * mx), round((y1 - y0) + 2 * my)
    if cw < 4 or ch < 4:
        return None
    rot = cv2.getRotationMatrix2D((ccx, ccy), float(angle_deg), 1.0)
    rot[0, 2] += cw / 2.0 - ccx     # the upright padded rect's TL lands at (0, 0)
    rot[1, 2] += ch / 2.0 - ccy
    return cv2.warpAffine(image, rot, (cw, ch), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))


def letterbox_square(image, size: int = 320, pad_value: int = 114):
    """Aspect-preserving resize onto a ``size``x``size`` canvas (YOLO-style grey
    pad) — the same framing the trainer/producer use for cell crops."""
    import cv2
    import numpy as np
    h, w = image.shape[:2]
    r = size / float(max(h, w))
    nw, nh = max(1, round(w * r)), max(1, round(h * r))
    canvas = np.full((size, size, 3), pad_value, dtype=np.uint8)
    x0, y0 = (size - nw) // 2, (size - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    return canvas
