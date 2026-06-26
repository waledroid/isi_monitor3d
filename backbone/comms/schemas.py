"""UDP/JSON envelopes — the **public contract** between the Backbone and modules.

Six envelope types:

* ``Track2DMessage`` — always-on output from the homography layer (S4).
* ``Track3DMessage`` — subscription-driven output from the triangulation
  layer (S5). Same ``track_id`` as the corresponding ``Track2DMessage`` —
  the "one identity space" principle.
* ``PassingEventMessage`` — zone entry/leave event (Phase B). Emitted when a
  tracked object crosses a zone boundary. Published on the same sinks as
  track messages (UDP/JSON, MQTT).
* ``ImageRefMessage`` — image reference (Phase C). Emitted alongside a
  ``PassingEventMessage`` when snapshot-writing is enabled. Carries a URL
  (``file://`` or HTTP) to the saved JPEG; **never** raw image bytes.
* ``DiagnosticsMessage`` — periodic heartbeat (Phase 1). Published by the
  ``DiagnosticsPublisher`` every ``interval_sec`` seconds.  Carries node
  identity, mode, source liveness, fps, latency stats, and a calibration
  fact-check. Additive within v4.
* ``ConfigMessage`` — retained config advertisement (Phase 1). Published once
  at startup with ``retain=True`` on MQTT so new subscribers immediately know
  the node's zones/cameras/mode.  Additive within v4.

These are **on-wire** types, validated and serialized by pydantic. The
in-process types (``backbone.core.types.Track2D``, ``.Track3D``) are
intentionally separate so internal refactors don't break the bus contract.

Versioning: ``schema_version`` is a single integer that consumers MUST read
before parsing further. Bump on any breaking change. Adding fields is
non-breaking if they're optional and default-valued; renaming or removing is.

MQTT topic convention (orthogonal to ``schema_version``): topics are namespaced
``<base>/<version>/<node_id>/<suffix>`` — e.g. ``isi/v1/zone_a/track2d/person``.
``base`` is the deployment root (default ``isi``), ``version`` is ``TOPIC_VERSION``
below, ``node_id`` identifies the publishing Backbone, and ``suffix`` is the
per-message-type tail (``track2d/<cls>``, ``config``, ``diagnostics/heartbeat``,
…). The version lives in the operator-set MQTT ``prefix`` (``isi/v1/<node_id>``);
``MqttSink`` keeps a freeform ``prefix`` and does not parse it. The gateway
subscriber parses the version segment back out (and falls back to ``v0`` for
legacy unversioned ``isi/<node_id>/...`` topics during transition).
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from backbone.core.types import Track2D, Track3D

SCHEMA_VERSION = 4
"""Bumped on any breaking change to the UDP/JSON contract.

v2: added optional pallet ``occupancy_*`` fields to ``Track2DMessage`` (additive,
non-breaking — v1 consumers can ignore them).
v4: added ``PassingEventMessage`` (zone entry/leave events, Phase B). All prior
message shapes are unchanged; ``parse_envelope`` accepts both v3 and v4."""

_ACCEPTED_VERSIONS = frozenset({3, 4})

TOPIC_VERSION = "v1"
"""Current MQTT topic-contract version, shared so node + gateway agree.

Embedded in the topic tree as ``<base>/<version>/<node_id>/<suffix>`` (the
operator sets the MQTT sink ``prefix`` to ``isi/v1/<node_id>``). Independent of
``SCHEMA_VERSION`` (the payload contract): a topic-layout change bumps this; a
payload-shape change bumps ``SCHEMA_VERSION``."""


class MessageType(str, Enum):
    TRACK_2D = "track_2d"
    TRACK_3D = "track_3d"
    PASSING = "passing"
    IMAGE_REF = "image_ref"
    DIAGNOSTICS = "diagnostics"
    CONFIG = "config"


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


class PassingEventMessage(BaseModel):
    """Wire format for a zone entry/leave event (Phase B).

    Emitted when a tracked object crosses a zone boundary. Published through
    the same ``MetadataSink`` fan-out as ``Track2DMessage`` / ``Track3DMessage``.

    MQTT topic: ``{prefix}/zones/{zone}/passings`` (zone name sanitised).
    UDP: same JSON datagram channel as tracks.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(default=SCHEMA_VERSION, ge=1)
    type: Literal[MessageType.PASSING] = MessageType.PASSING
    ts: float = Field(..., description="capture_ts in Unix seconds")
    track_id: int = Field(..., ge=0)
    cls: str
    zone: str
    direction: Literal["enter", "leave"]

    @classmethod
    def from_event(cls, event: object) -> PassingEventMessage:
        """Construct from a ``PassingEvent`` (avoid hard import of shared module)."""
        return cls(
            ts=float(event.ts),          # type: ignore[attr-defined]
            track_id=int(event.track_id),  # type: ignore[attr-defined]
            cls=str(event.cls),            # type: ignore[attr-defined]
            zone=str(event.zone),          # type: ignore[attr-defined]
            direction=str(event.direction),  # type: ignore[attr-defined]
        )


class ImageRefMessage(BaseModel):
    """Wire format for an image-reference notification (Phase C).

    Published when a zone-passing event fires **and** snapshot-writing is
    enabled in ``backbone.yaml``.  The ``url`` points to the saved JPEG;
    raw image bytes are **never** included in this message.

    Fields mirror ``PassingEventMessage`` so consumers can correlate the two
    by ``(track_id, zone, ts)``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(default=SCHEMA_VERSION, ge=1)
    type: Literal[MessageType.IMAGE_REF] = MessageType.IMAGE_REF
    ts: float = Field(..., description="capture_ts in Unix seconds")
    track_id: int = Field(..., ge=0)
    cls: str
    zone: str
    url: str = Field(..., description="URL of the saved JPEG snapshot (no image bytes)")


class LatencyStats(BaseModel):
    """Latency percentiles from ``LatencyMeter.percentiles()`` (in milliseconds)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    n: int = 0


class CalibrationFactCheck(BaseModel):
    """Quick sanity-check on the loaded calibration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    loaded: bool
    rms_ok: bool
    mode: int = Field(..., ge=1, le=2)


class DiagnosticsMessage(BaseModel):
    """Periodic heartbeat emitted by ``DiagnosticsPublisher`` (Phase 1 distributed).

    Carries node identity, operational mode, per-source liveness, frame-rate,
    latency stats (p50/p95/p99 ms), zone/subscription counts, and a quick
    calibration fact-check.  Additive within schema_version 4.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(default=SCHEMA_VERSION, ge=1)
    type: Literal[MessageType.DIAGNOSTICS] = MessageType.DIAGNOSTICS
    ts: float = Field(..., description="wall-clock Unix seconds at emit time")
    node_id: str
    mode: str
    sources: dict[str, str]   # camera_id → "alive" | "exited" | "crashed"
    frame_count: int = Field(..., ge=0)
    fps: float = Field(..., ge=0.0)
    latency_ms: LatencyStats
    zones: int = Field(..., ge=0)
    subscriptions: int = Field(..., ge=0)
    calibration: CalibrationFactCheck


class ZoneSpec(BaseModel):
    """Serialisable description of one zone for the config advertisement."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    kind: str
    type: str
    severity: str
    polygon: list[list[float]]   # [[x, y], ...] in meters


class ConfigMessage(BaseModel):
    """Retained config advertisement published once at node startup (Phase 1 distributed).

    Consumers that subscribe after startup still receive this immediately
    via the broker's retained-message mechanism.  Additive within schema_version 4.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(default=SCHEMA_VERSION, ge=1)
    type: Literal[MessageType.CONFIG] = MessageType.CONFIG
    ts: float = Field(..., description="wall-clock Unix seconds at emit time")
    node_id: str
    area: str
    mode: str
    cameras: list[str]
    zones: list[ZoneSpec]
    calibration: CalibrationFactCheck


class SchemaVersionError(ValueError):
    """Raised when a received message has an incompatible schema_version."""


def parse_envelope(
    data: dict,
) -> (
    Track2DMessage
    | Track3DMessage
    | PassingEventMessage
    | ImageRefMessage
    | DiagnosticsMessage
    | ConfigMessage
):
    """Discriminate by ``type`` field and parse with the right model.

    Consumer-side helper — modules will use this to decode UDP payloads.

    Accepts both schema_version 3 (pre-Phase-B) and 4 (current). Version 3
    messages never carry the ``PASSING``, ``DIAGNOSTICS``, or ``CONFIG`` types,
    so consumers can parse mixed streams produced by older Backbone builds
    without rejecting old packets.
    Any version outside {3, 4} raises ``SchemaVersionError``.
    """
    version = int(data.get("schema_version", 0))
    if version not in _ACCEPTED_VERSIONS:
        raise SchemaVersionError(
            f"received schema_version={version}; "
            f"accepted versions are {sorted(_ACCEPTED_VERSIONS)}"
        )
    msg_type = data.get("type")
    if msg_type == MessageType.TRACK_2D.value:
        return Track2DMessage.model_validate(data)
    if msg_type == MessageType.TRACK_3D.value:
        return Track3DMessage.model_validate(data)
    if msg_type == MessageType.PASSING.value:
        return PassingEventMessage.model_validate(data)
    if msg_type == MessageType.IMAGE_REF.value:
        return ImageRefMessage.model_validate(data)
    if msg_type == MessageType.DIAGNOSTICS.value:
        return DiagnosticsMessage.model_validate(data)
    if msg_type == MessageType.CONFIG.value:
        return ConfigMessage.model_validate(data)
    raise ValueError(f"unknown message type: {msg_type!r}")
