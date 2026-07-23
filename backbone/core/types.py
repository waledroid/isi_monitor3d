"""Shared dataclasses passed between Backbone sub-modules.

These are the in-process types. The on-wire types (UDP/JSON payloads)
live in `backbone.comms.schemas`; keep the two intentionally separate so
internal refactors do not break the public contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Frame:
    """A single decoded image with its capture-time clock value."""

    camera_id: str
    capture_ts: float
    frame_idx: int
    image: np.ndarray


@dataclass(slots=True)
class FramePair:
    """Two synchronized frames after NTP-pairing — the unit that drives the pipeline."""

    capture_ts: float
    frame_idx: int
    frames: dict[str, Frame]


@dataclass(slots=True)
class Detection:
    """A single per-camera detection from the Detector.

    `foot_uv` is the bottom-center of the bbox in pixels and is the only point that
    the homography layer projects. `keypoints_uv` is populated only when the detector
    is a pose model and the active subscriptions demand it. `mask` is populated only
    by seg detectors (yolo_onnx_seg) — a full-frame HxW ``bool`` array of the
    instance mask; detect detectors leave it ``None``.
    """

    camera_id: str
    capture_ts: float
    cls: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    foot_uv: tuple[float, float]
    keypoints_uv: np.ndarray | None = None
    mask: np.ndarray | None = None
    # When the detector ran on a CROP (zone-scoped detection), ``mask`` is
    # crop-sized and this is the crop's origin in frame pixels. ``None`` means
    # the mask (if any) spans the full frame. Mask AREA consumers
    # (pallet_occupancy._load_area) are offset-agnostic by construction.
    mask_offset_xy: tuple[int, int] | None = None
    # Zone-scoped detection: the source crop's window in FRAME pixels. Lets the
    # cross-crop deduper recognize a box that was CUT OFF by its own crop edge
    # (an offset partial view of an object another crop saw more fully).
    crop_xyxy: tuple[float, float, float, float] | None = None


@dataclass(slots=True)
class Track2D:
    """A fused, identity-stable track in metric floor coordinates.

    Pallet tracks may carry an occupancy verdict (the empty/full KPI): set by the
    ``PalletOccupancy`` enricher and published on the wire (schema v2). ``None`` for
    non-pallet tracks (and pallets not yet classified).
    """

    track_id: int
    cls: str
    capture_ts: float
    xy_m: tuple[float, float]
    vxy_m: tuple[float, float]
    confidence: float
    cameras_seeing: tuple[str, ...]
    occupancy_state: str | None = None         # "empty" | "full" | None
    occupancy_content: str | None = None        # "carton" | "polybag" | None (when full)
    occupancy_confidence: float = 0.0


@dataclass(slots=True)
class Track3D:
    """A 3D-lifted view of an existing Track2D. Shares the parent's track_id."""

    track_id: int
    cls: str
    capture_ts: float
    xyz_m: tuple[float, float, float]
    vxyz_m: tuple[float, float, float]
    contributing_cameras: tuple[str, ...]
    max_reprojection_error_px: float
    keypoints_xyz: np.ndarray | None = None
    # Single-view floor fallback (Mode 2 occlusion): when only one camera sees a
    # floor object, X,Y come from its homography and Z is pinned to 0 (feet on the
    # floor). single_view=True flags it; confidence is downgraded vs a true 2-view
    # triangulation. Default = a real 2-view triangulation.
    single_view: bool = False
    confidence: float = 1.0
