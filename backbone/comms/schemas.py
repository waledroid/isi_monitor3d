"""UDP/JSON envelopes — the **public contract** between the Backbone and modules.

Seven envelope types:

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
* ``ZoneStateMessage`` — the current contents of one zone: every tracked
  object inside it with class + confidence (the WMS/FMS integration signal).
  Published retained on MQTT (``{prefix}/zone/{zone}``) on change plus a
  periodic refresh; an empty zone is an explicit empty ``objects`` tuple.
  Additive within v4.
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
``<base>/<version>/<node_id>/<suffix>`` — e.g. ``isiMonitor3D/v1/zone_a/track2d/person``.
``base`` is the deployment root (default ``isiMonitor3D``), ``version`` is ``TOPIC_VERSION``
below, ``node_id`` identifies the publishing Backbone, and ``suffix`` is the
per-message-type tail (``track2d/<cls>``, ``config``, ``diagnostics/heartbeat``,
…). The version lives in the operator-set MQTT ``prefix`` (``isiMonitor3D/v1/<node_id>``);
``MqttSink`` keeps a freeform ``prefix`` and does not parse it. The gateway
subscriber parses the version segment back out (and falls back to ``v0`` for
legacy unversioned ``isiMonitor3D/<node_id>/...`` topics during transition).
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from backbone.core.types import Track2D, Track3D

SCHEMA_VERSION = 6
"""Bumped on any breaking change to the UDP/JSON contract.

v2: added optional pallet ``occupancy_*`` fields to ``Track2DMessage`` (additive,
non-breaking — v1 consumers can ignore them).
v4: added ``PassingEventMessage`` (zone entry/leave events, Phase B). All prior
message shapes are unchanged; ``parse_envelope`` accepts both v3 and v4.
v5: added ``ProximityMessage`` (person↔object floor distances — the safety
signal). Additive; all prior shapes unchanged.
v6: added ``ObservationsMessage`` (per-camera raw detections for display
consumers — one perception, rendered everywhere). Additive; UDP-only."""

_ACCEPTED_VERSIONS = frozenset({3, 4, 5, 6})

TOPIC_VERSION = "v1"
"""Current MQTT topic-contract version, shared so node + gateway agree.

Embedded in the topic tree as ``<base>/<version>/<node_id>/<suffix>`` (the
operator sets the MQTT sink ``prefix`` to ``isiMonitor3D/v1/<node_id>``). Independent of
``SCHEMA_VERSION`` (the payload contract): a topic-layout change bumps this; a
payload-shape change bumps ``SCHEMA_VERSION``."""


class MessageType(str, Enum):
    TRACK_2D = "track_2d"
    TRACK_3D = "track_3d"
    PASSING = "passing"
    IMAGE_REF = "image_ref"
    ZONE_STATE = "zone_state"
    PROXIMITY = "proximity"
    OBSERVATIONS = "observations"
    DIAGNOSTICS = "diagnostics"
    CONFIG = "config"
    FRAGMENT = "fragment"


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

    MQTT topic: ``{prefix}/zone/{zone}/passings`` (zone name sanitised).
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


class ZoneObject(BaseModel):
    """One tracked object currently inside a zone (element of ``ZoneStateMessage``)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    track_id: int = Field(..., ge=0)
    cls: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    xy_m: tuple[float, float]
    # Pallet occupancy (the empty/full KPI) — optional, mirrors Track2DMessage.
    occupancy_state: str | None = None          # "empty" | "full" | None
    occupancy_content: str | None = None        # "carton" | "polybag" | None
    occupancy_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ZoneStateMessage(BaseModel):
    """Wire format for the current contents of one zone (the WMS/FMS signal).

    Lists every tracked object whose floor position lies inside the zone,
    with class and confidence. Published through the same ``MetadataSink``
    fan-out as tracks; on MQTT it is **retained** on ``{prefix}/zone/{zone}``
    so a late-joining consumer immediately sees every zone's occupancy.

    An **empty** zone is an explicit message with ``objects=()`` / ``count=0``
    — never silence — so consumers can distinguish "empty" from "unknown".
    ``node_id`` is deliberately absent: like track messages, the gateway
    derives it from the topic.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(default=SCHEMA_VERSION, ge=1)
    type: Literal[MessageType.ZONE_STATE] = MessageType.ZONE_STATE
    ts: float = Field(..., description="capture_ts of the frame producing this state")
    zone: str
    objects: tuple[ZoneObject, ...]
    count: int = Field(..., ge=0, description="len(objects) — consumer convenience")

    @classmethod
    def from_state(cls, state: object) -> ZoneStateMessage:
        """Construct from a ``ZoneState`` (avoid hard import of shared module)."""
        objects = tuple(
            ZoneObject(
                track_id=int(o.track_id),
                cls=str(o.cls),
                confidence=float(o.confidence),
                xy_m=o.xy_m,
                occupancy_state=o.occupancy_state,
                occupancy_content=o.occupancy_content,
                occupancy_confidence=float(o.occupancy_confidence),
            )
            for o in state.occupants  # type: ignore[attr-defined]
        )
        return cls(
            ts=float(state.ts),      # type: ignore[attr-defined]
            zone=str(state.zone),    # type: ignore[attr-defined]
            objects=objects,
            count=len(objects),
        )


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


class ObservationDet(BaseModel):
    """One raw per-camera detection (element of ``ObservationsMessage``)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cls: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox_xyxy: tuple[float, float, float, float]
    foot_uv: tuple[float, float]
    # Pallet occupancy hints (same semantics as ZoneObject); None = not a
    # pallet / undecided.
    occupancy_state: str | None = None
    occupancy_content: str | None = None
    occupancy_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    # Simplified instance-mask outline in frame coords ([[x, y], ...]) —
    # present only when the Backbone decodes masks. Consumers fall back to
    # the bbox when absent.
    mask_poly: tuple[tuple[float, float], ...] | None = None


class ObservationsMessage(BaseModel):
    """Per-camera raw detections — the display consumers' feed.

    ONE perception: the Backbone's zone-scoped detector is the single object
    detector in the system; the dashboard renders these observations instead
    of running its own models. Published per camera per pair (~pipeline rate)
    on the **UDP sink only** — it is a display concern, deliberately kept off
    MQTT/broker. ``frame_wh`` is the coordinate space (the camera's
    CALIBRATION frame); consumers scale to their display exactly like stored
    zone patches do.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(default=SCHEMA_VERSION, ge=1)
    type: Literal[MessageType.OBSERVATIONS] = MessageType.OBSERVATIONS
    ts: float = Field(..., description="capture_ts of the frame pair")
    camera_id: str
    frame_wh: tuple[int, int]
    dets: tuple[ObservationDet, ...]


class FragmentMessage(BaseModel):
    """One UDP-transport fragment of a larger JSON message.

    UDP datagrams that would exceed the path MTU get IP-fragmented, and some
    network layers silently drop the fragments (WSL2 ``networkingMode=mirrored``
    drops EVERY loopback UDP datagram over ~1.5 KB — observations with mask
    polygons never arrive). ``UdpSink`` therefore splits large payloads at the
    APPLICATION layer: the serialized JSON text is sliced into chunks, each
    wrapped in this envelope. Consumers reassemble with ``FragmentBuffer`` and
    re-parse the joined text. Fragments exist only on the UDP transport — MQTT
    (TCP) never fragments.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(default=SCHEMA_VERSION, ge=1)
    type: Literal[MessageType.FRAGMENT] = MessageType.FRAGMENT
    fid: str = Field(..., description="fragment-group id, unique per message")
    i: int = Field(..., ge=0, description="fragment index, 0-based")
    n: int = Field(..., ge=1, description="total fragments in the group")
    data: str = Field(..., description="slice of the original JSON text")


class FragmentBuffer:
    """Consumer-side reassembly for ``FragmentMessage`` streams.

    ``add()`` returns the complete original JSON text once every fragment of
    a group has arrived, else ``None``. Incomplete groups are pruned after
    ``max_age_s`` (UDP loss must not leak memory); at most ``max_groups``
    are held.
    """

    def __init__(self, max_age_s: float = 5.0, max_groups: int = 64) -> None:
        self._max_age_s = float(max_age_s)
        self._max_groups = int(max_groups)
        self._groups: dict[str, tuple[float, int, dict[int, str]]] = {}

    def add(self, frag: FragmentMessage, now: float) -> str | None:
        self._prune(now)
        first_ts, n, parts = self._groups.get(frag.fid, (now, frag.n, {}))
        if frag.n != n or frag.i >= n:
            self._groups.pop(frag.fid, None)   # inconsistent group — drop it
            return None
        parts[frag.i] = frag.data
        if len(parts) == n:
            self._groups.pop(frag.fid, None)
            return "".join(parts[k] for k in range(n))
        self._groups[frag.fid] = (first_ts, n, parts)
        return None

    def _prune(self, now: float) -> None:
        if self._groups:
            cutoff = now - self._max_age_s
            self._groups = {k: v for k, v in self._groups.items() if v[0] >= cutoff}
        while len(self._groups) >= self._max_groups:
            self._groups.pop(next(iter(self._groups)))


class ProximityPair(BaseModel):
    """One person↔object floor distance (element of ``ProximityMessage``)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    person_track_id: int = Field(..., ge=0)
    object_track_id: int = Field(..., ge=0)
    object_cls: str
    distance_m: float = Field(..., ge=0.0)
    person_xy_m: tuple[float, float]
    object_xy_m: tuple[float, float]


class ProximityMessage(BaseModel):
    """Wire format for person↔object proximity — the safety/AGV signal.

    Every (person, object) track pair whose floor distance is within
    ``max_distance_m``, computed by the Backbone from the SAME metric tracks
    it publishes (identity-stable ids — consumers can join on ``track_id``).
    An **empty** ``pairs`` message is sent once when the last pair clears —
    never silence — so consumers can distinguish "nobody near anything" from
    "unknown". On MQTT it is retained on ``{prefix}/proximity``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(default=SCHEMA_VERSION, ge=1)
    type: Literal[MessageType.PROXIMITY] = MessageType.PROXIMITY
    ts: float = Field(..., description="capture_ts of the frame producing this state")
    max_distance_m: float = Field(..., gt=0.0, description="the configured horizon")
    pairs: tuple[ProximityPair, ...]


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
    # Per-camera INGEST rate (frames arriving from each source, pre-sync) —
    # the operator-facing camera health signal. Additive: defaults empty so
    # payloads from older nodes still parse.
    fps_by_camera: dict[str, float] = Field(default_factory=dict)
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
    | ZoneStateMessage
    | ProximityMessage
    | ObservationsMessage
    | DiagnosticsMessage
    | ConfigMessage
    | FragmentMessage
):
    """Discriminate by ``type`` field and parse with the right model.

    Consumer-side helper — modules will use this to decode UDP payloads.

    Accepts both schema_version 3 (pre-Phase-B) and 4 (current). Version 3
    messages never carry the ``PASSING``, ``DIAGNOSTICS``, or ``CONFIG`` types,
    so consumers can parse mixed streams produced by older Backbone builds
    without rejecting old packets.
    Any version outside {3, 4, 5, 6} raises ``SchemaVersionError``.
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
    if msg_type == MessageType.ZONE_STATE.value:
        return ZoneStateMessage.model_validate(data)
    if msg_type == MessageType.PROXIMITY.value:
        return ProximityMessage.model_validate(data)
    if msg_type == MessageType.OBSERVATIONS.value:
        return ObservationsMessage.model_validate(data)
    if msg_type == MessageType.DIAGNOSTICS.value:
        return DiagnosticsMessage.model_validate(data)
    if msg_type == MessageType.CONFIG.value:
        return ConfigMessage.model_validate(data)
    if msg_type == MessageType.FRAGMENT.value:
        return FragmentMessage.model_validate(data)
    raise ValueError(f"unknown message type: {msg_type!r}")
