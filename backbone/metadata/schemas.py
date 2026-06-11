"""UDP/JSON envelopes — the **public contract** between the Backbone and modules.

Two envelope types:

* ``Track2DMessage`` — always-on output from the homography layer (S4).
* ``Track3DMessage`` — subscription-driven output from the triangulation
  layer (S5). Same ``track_id`` as the corresponding ``Track2DMessage`` —
  the "one identity space" principle.

These are **on-wire** types, validated and serialized by pydantic. The
in-process types (``backbone.core.types.Track2D``, ``.Track3D``) are
intentionally separate so internal refactors don't break the bus contract.

Versioning: ``schema_version`` is a single integer that consumers MUST read
before parsing further. Bump on any breaking change. Adding fields is
non-breaking if they're optional and default-valued; renaming or removing is.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from backbone.core.types import Track2D, Track3D

SCHEMA_VERSION = 3
"""Bumped on any breaking change to the UDP/JSON contract.

v2: added optional pallet ``occupancy_*`` fields to ``Track2DMessage`` (additive,
non-breaking — v1 consumers can ignore them)."""


class MessageType(str, Enum):
    TRACK_2D = "track_2d"
    TRACK_3D = "track_3d"


class Track2DMessage(BaseModel):
    """Wire format for a metric 2D track."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(default=SCHEMA_VERSION, ge=1)
    type: Literal[MessageType.TRACK_2D] = MessageType.TRACK_2D
    ts: float = Field(..., description="capture_ts in Unix seconds")
    track_id: int = Field(..., ge=0)
    cls: str
    xy_m: tuple[float, float]
    vxy_m: tuple[float, float]
    confidence: float = Field(..., ge=0.0, le=1.0)
    cameras_seeing: tuple[str, ...]
    # v2 — pallet occupancy (the empty/full KPI). Optional + defaulted ⇒ v1-safe.
    occupancy_state: str | None = None         # "empty" | "full" | None
    occupancy_content: str | None = None        # "carton" | "polybag" | None
    occupancy_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @classmethod
    def from_track(cls, track: Track2D) -> Track2DMessage:
        return cls(
            ts=track.capture_ts,
            track_id=track.track_id,
            cls=track.cls,
            xy_m=track.xy_m,
            vxy_m=track.vxy_m,
            confidence=track.confidence,
            cameras_seeing=track.cameras_seeing,
            occupancy_state=track.occupancy_state,
            occupancy_content=track.occupancy_content,
            occupancy_confidence=track.occupancy_confidence,
        )


class Track3DMessage(BaseModel):
    """Wire format for a metric 3D track (subscription-driven)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(default=SCHEMA_VERSION, ge=1)
    type: Literal[MessageType.TRACK_3D] = MessageType.TRACK_3D
    ts: float = Field(..., description="capture_ts in Unix seconds")
    track_id: int = Field(..., ge=0)
    cls: str
    xyz_m: tuple[float, float, float]
    vxyz_m: tuple[float, float, float]
    contributing_cameras: tuple[str, ...]
    max_reprojection_error_px: float = Field(..., ge=0.0)
    keypoints_xyz: list[tuple[float, float, float]] | None = None  # S5.5
    # Single-view floor fallback (Mode 2 occlusion): Z pinned to 0 from one camera.
    single_view: bool = False
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @classmethod
    def from_track(cls, track: Track3D) -> Track3DMessage:
        keypoints: list[tuple[float, float, float]] | None = None
        if track.keypoints_xyz is not None:
            keypoints = [(float(p[0]), float(p[1]), float(p[2])) for p in track.keypoints_xyz]
        return cls(
            ts=track.capture_ts,
            track_id=track.track_id,
            cls=track.cls,
            xyz_m=track.xyz_m,
            vxyz_m=track.vxyz_m,
            contributing_cameras=track.contributing_cameras,
            max_reprojection_error_px=track.max_reprojection_error_px,
            keypoints_xyz=keypoints,
            single_view=track.single_view,
            confidence=track.confidence,
        )


class SchemaVersionError(ValueError):
    """Raised when a received message has an incompatible schema_version."""


def parse_envelope(data: dict) -> Track2DMessage | Track3DMessage:
    """Discriminate by ``type`` field and parse with the right model.

    Consumer-side helper — modules will use this to decode UDP payloads.
    """
    version = int(data.get("schema_version", 0))
    if version != SCHEMA_VERSION:
        raise SchemaVersionError(
            f"received schema_version={version}, this Backbone speaks {SCHEMA_VERSION}"
        )
    msg_type = data.get("type")
    if msg_type == MessageType.TRACK_2D.value:
        return Track2DMessage.model_validate(data)
    if msg_type == MessageType.TRACK_3D.value:
        return Track3DMessage.model_validate(data)
    raise ValueError(f"unknown message type: {msg_type!r}")
