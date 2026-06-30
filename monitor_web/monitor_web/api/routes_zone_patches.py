"""Zone-patch ROIs — pixel-space "watch boxes" drawn directly on a camera frame.

Each ROI is cropped out of the live feed and run through the detector as a
*targeted SAHI* tile: a small region upscaled to the model's input makes far/small
objects easy to detect. Stored in ``zone_patches.yaml`` next to ``backbone.yaml``;
monitor_web-only (the Backbone never reads these — they're a UI monitoring aid, not
metric floor zones). Rects are in SOURCE-frame pixels, with the frame size they were
drawn at so the crop stays correct if the live resolution differs.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field, model_validator

from .. import dashboard_config

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_PATCHES = 6   # max zones the operator can create (mirrors the dashboard cap)


def load_patches(cfg) -> list[dict]:
    """Return the stored ROI list (``[]`` when none configured). Reads the merged
    ``zone_patches`` section of the unified dashboard config."""
    patches = dashboard_config.read_section(cfg, "zone_patches").get("patches", [])
    return patches if isinstance(patches, list) else []


def find_patch(cfg, patch_id: str) -> dict | None:
    """Look up one ROI by id."""
    return next((p for p in load_patches(cfg) if str(p.get("id")) == str(patch_id)), None)


def patch_rect(patch: dict) -> list[float] | None:
    """Axis-aligned bounding rect ``[x0,y0,x1,y1]`` (source px) for a patch — derived
    from its polygon if present (the drawn shape), else the stored rect. The polygon
    is the display boundary; the crop fed to detection is this bounding rectangle."""
    poly = patch.get("polygon")
    if isinstance(poly, list) and len(poly) >= 3:
        xs = [float(p[0]) for p in poly]
        ys = [float(p[1]) for p in poly]
        return [min(xs), min(ys), max(xs), max(ys)]
    return patch.get("rect")


def patch_pixel_box(rect, stored_wh, frame_wh):
    """Map a stored ROI ``rect`` (source px, drawn at ``stored_wh``) to an integer
    pixel box ``(x0, y0, x1, y1)`` on a frame of ``frame_wh`` — scaling when the live
    frame size differs (the frame-size guard). Clamped to the frame; ``None`` if the
    box is degenerate (< 2 px on a side)."""
    x0, y0, x1, y1 = (float(v) for v in rect)
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    fw, fh = int(frame_wh[0]), int(frame_wh[1])
    if stored_wh and (int(stored_wh[0]), int(stored_wh[1])) != (fw, fh):
        sw, sh = float(stored_wh[0]) or fw, float(stored_wh[1]) or fh
        sx, sy = fw / sw, fh / sh
        x0, x1, y0, y1 = x0 * sx, x1 * sx, y0 * sy, y1 * sy
    x0i, x1i = max(0, min(fw, round(x0))), max(0, min(fw, round(x1)))
    y0i, y1i = max(0, min(fh, round(y0))), max(0, min(fh, round(y1)))
    if x1i - x0i < 2 or y1i - y0i < 2:
        return None
    return (x0i, y0i, x1i, y1i)


class PatchRect(BaseModel):
    id: str
    name: str = ""
    camera: str = "cam_a"
    # The drawn shape, in SOURCE-frame pixels. A polygon (>=3 pts) is the display
    # boundary (red overlay on the cam); `rect` is its axis-aligned bounding box and
    # is what gets cropped + fed to detection. Either may be supplied — rect is
    # auto-derived from the polygon when omitted (and a bare rect stays valid).
    rect: list[float] | None = Field(default=None, min_length=4, max_length=4)
    polygon: list[list[float]] | None = None                     # [[u,v], ...] source px
    frame_wh: list[int] | None = None                            # [W,H] drawn at (guard)
    model: str | None = None    # per-zone detection model (onnx path); None = global
    # Detection input size for this zone: the bounding-rect crop is resized to fit
    # this (INTER_AREA on downscale) before inference — smaller = faster/lighter.
    infer_size: int = 320
    color: str | None = None    # outline colour on the cam overlay (hex); None = red
    confidence: float | None = None   # per-zone detection confidence; None = global
    # SAHI (Slicing Aided Hyper Inference) — slice this zone's crop into a
    # rows x cols grid of overlapping tiles, detect each at infer_size, NMS-merge,
    # remap to source. OFF by default (zero behaviour change). Only worth it on
    # FAR zones whose distant objects shrink to a few pixels under the single
    # resize. ``sahi_overlap`` is the fraction of tile size shared with neighbours.
    sahi: bool = False
    sahi_rows: int = 2
    sahi_cols: int = 2
    sahi_overlap: float = 0.2

    model_config = {"extra": "ignore"}   # tolerate legacy max_fps keys in saved YAML

    @model_validator(mode="after")
    def _derive_and_clamp(self) -> PatchRect:
        if (not self.rect or len(self.rect) != 4) and self.polygon and len(self.polygon) >= 3:
            xs = [float(p[0]) for p in self.polygon]
            ys = [float(p[1]) for p in self.polygon]
            self.rect = [min(xs), min(ys), max(xs), max(ys)]
        if not self.rect or len(self.rect) != 4:
            raise ValueError("zone patch needs a rect or a polygon of >=3 points")
        self.infer_size = max(64, min(1280, int(self.infer_size)))
        self.sahi_rows = max(1, min(4, int(self.sahi_rows)))
        self.sahi_cols = max(1, min(4, int(self.sahi_cols)))
        self.sahi_overlap = max(0.0, min(0.5, float(self.sahi_overlap)))
        return self


class PatchesBody(BaseModel):
    patches: list[PatchRect] = Field(default_factory=list, max_length=MAX_PATCHES)


@router.get("/api/zone-patches")
async def get_zone_patches(request: Request) -> dict:
    return {"patches": load_patches(request.app.state.settings)}


@router.post("/api/zone-patches")
def post_zone_patches(body: PatchesBody, request: Request) -> dict:
    # Sync handler on purpose — write_section fsyncs and mgr.reload() can join
    # worker threads; in the threadpool neither stalls the event loop.
    cfg = request.app.state.settings
    doc = {"patches": [p.model_dump() for p in body.patches]}
    dashboard_config.write_section(cfg, "zone_patches", doc)
    logger.info("zone-patches: saved %d ROI(s)", len(body.patches))
    # Sync the background detection workers to the new zone set (start/update/stop
    # per camera). getattr guard: apps built without the manager stay safe.
    mgr = getattr(request.app.state, "zone_manager", None)
    if mgr is not None:
        mgr.reload()
    return {"ok": True, "count": len(body.patches)}
