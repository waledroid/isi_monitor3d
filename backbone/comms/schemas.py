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
consumers — one perception, rendered everywhere). Additive; UDP-only.

Within v6 (no bump — additive + default-valued): ``zone_id`` (STABLE zone
identity) added to ``ZoneStateMessage``, ``PassingEventMessage``,
``ImageRefMessage`` and ``ZoneSpec``. Consumers MUST key zone semantics on
``zone_id`` (immutable), not ``zone`` (the renamable operator label). It
defaults to "" so a payload from a pre-id Backbone still parses; the current
Backbone always populates it.

Within v6 (no bump — additive + default-valued): optional ``decision``
(``ZoneDecisionModel`` — the PalletStateManager enum + content + counts) added
to ``ZoneStateMessage``. Defaults to None so a decision-less payload from an
older Backbone still parses everywhere.

Within v6 (no bump — additive + default-valued): ``z_base_m`` (height in
meters of the plane a zone's polygon lives on — mirrors ``Zone.z_base_m``)
added to ``ZoneSpec``. Defaults to 0.0 (the floor) so a legacy retained
``ConfigMessage`` advert still parses.

v6 additive (2026-08-17): etagere_state — EtagereStateMessage; defaulted fields,
no bump."""

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
    DETECTION_SET = "detection_set"
    ETAGERE_STATE = "etagere_state"


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
    # STABLE zone identity — consumers (AGV/WMS) MUST key on this, not ``zone``
    # (the human label, which is freely renamable). Defaulted "" so a legacy
    # v6 payload from a pre-id Backbone still parses; the current Backbone
    # always populates it. See module docstring / the parse_envelope note.
    zone_id: str = ""
    direction: Literal["enter", "leave"]

    @classmethod
    def from_event(cls, event: object) -> PassingEventMessage:
        """Construct from a ``PassingEvent`` (avoid hard import of shared module)."""
        return cls(
            ts=float(event.ts),          # type: ignore[attr-defined]
            track_id=int(event.track_id),  # type: ignore[attr-defined]
            cls=str(event.cls),            # type: ignore[attr-defined]
            zone=str(event.zone),          # type: ignore[attr-defined]
            zone_id=str(getattr(event, "zone_id", "")),
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
    # STABLE zone identity — mirrors ``PassingEventMessage`` so the image
    # topic segment tracks the zone by id, not name. Defaulted "" for legacy.
    zone_id: str = ""
    url: str = Field(..., description="URL of the saved JPEG snapshot (no image bytes)")


class ZoneObject(BaseModel):
    """One tracked object currently inside a zone (element of ``ZoneStateMessage``)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    track_id: int = Field(..., ge=0)
    cls: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    xy_m: tuple[float, float]
    # Pallet occupancy — RETIRED FROM THE WIRE 2026-08-19 (they were null/0
    # noise on most objects; the stabilised verdict lives in `decision`).
    # Kept as accepted-but-never-serialised fields so retained/old payloads
    # that still carry the keys parse cleanly (extra="forbid" would otherwise
    # reject them); `exclude=True` keeps every new dump free of them.
    occupancy_state: str | None = Field(default=None, exclude=True)
    occupancy_content: str | None = Field(default=None, exclude=True)
    occupancy_confidence: float = Field(default=0.0, ge=0.0, le=1.0, exclude=True)


class ZoneDecisionModel(BaseModel):
    """The ``PalletStateManager`` verdict for one zone (additive within v6).

    ``palette_state`` is the communication enum —
    ``no_data | no_palette | palette_empty | palette_loaded`` — decided from
    per-camera detection evidence (any camera's positive detection is proof),
    with hysteresis so it cannot flap. ``content`` lists the load classes when
    loaded (e.g. ``("carton",)``); ``counts`` is the per-class in-zone count
    (max across cameras, never summed). Renderers map the enum to text and
    stop re-deriving zone state from the occupants list.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    palette_state: str          # no_data|no_palette|palette_empty|palette_loaded
    content: tuple[str, ...] = ()
    counts: dict[str, int] = Field(default_factory=dict)

    @classmethod
    def from_decision(cls, decision: object) -> ZoneDecisionModel:
        """Construct from a ``ZoneDecision`` (avoid hard import of the
        homography module — same duck-typed bridge as ``from_state``)."""
        return cls(
            palette_state=str(decision.palette_state),   # type: ignore[attr-defined]
            content=tuple(str(c) for c in decision.content),  # type: ignore[attr-defined]
            counts={str(k): int(v)
                    for k, v in decision.counts.items()},  # type: ignore[attr-defined]
        )


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
    # STABLE zone identity — see ``PassingEventMessage.zone_id``. On MQTT the
    # topic segment is derived from this id (config ``metadata.mqtt_topic_zone``)
    # so a rename never orphans the zone's retained state. Defaulted "" so a
    # legacy v6 payload still parses.
    zone_id: str = ""
    objects: tuple[ZoneObject, ...]
    count: int = Field(..., ge=0, description="len(objects) — consumer convenience")
    # ALWAYS-PRESENT class summary — THE simple consumer key (AGV): with the
    # decision attached it carries the STABILISED per-class presence
    # (hysteresis, cross-camera union), [] when the zone is empty; without a
    # decision it mirrors objects[].cls. Additive + defaulted, no schema bump.
    cls: tuple[str, ...] = ()
    # Max detection confidence per PRESENT class (any camera), held through
    # dropouts with the presence itself; {} when the zone is empty. The
    # companion to `cls` for consumers that want scores. Added 2026-08-19.
    cls_confidence: dict[str, float] = Field(default_factory=dict)
    # Short-lived legacy spellings (same day, renamed on AGV request):
    # accepted on parse so retained old payloads don't reject, never emitted.
    classes: tuple[str, ...] = Field(default=(), exclude=True)
    class_confidence: dict[str, float] = Field(default_factory=dict, exclude=True)
    # OPTIONAL PalletStateManager verdict (additive within v6, default None so
    # a payload from a pre-decision Backbone still parses — mixed-version
    # rollout: gateway rebuilds BEFORE the Backbone starts emitting it).
    decision: ZoneDecisionModel | None = None

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
            zone_id=str(getattr(state, "zone_id", "")),
            objects=objects,
            count=len(objects),
            cls=tuple(o.cls for o in objects),
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
    # Pose keypoints ((u, v, conf) x K, frame coords) — persons only. Lets
    # display consumers render people without running their own pose model
    # (ONE perception applies to persons too).
    keypoints_uv: tuple[tuple[float, float, float], ...] | None = None


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


class WireDetection(BaseModel):
    """One detection inside a ``DetectionSetMessage`` (perception → metric).

    Coordinates are pixels in the producer's declared ``frame_wh`` space.
    ``confidence`` is the producer's RAW score — the metric engine's ByteTrack
    needs the low-confidence band (``conf_low``), so producers must NOT apply
    a display threshold here. Persons are ordinary detections with
    ``cls="person"`` plus ``keypoints_uv``; there is no occupancy field on
    purpose — occupancy is a METRIC verdict, computed downstream.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    cls: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    foot_uv: tuple[float, float]
    keypoints_uv: tuple[tuple[float, float, float], ...] | None = None
    mask_poly: tuple[tuple[float, float], ...] | None = None


class DetectionSetMessage(BaseModel):
    """Per-camera detections flowing INTO the Backbone (Direction 1).

    The inverse of ``ObservationsMessage``: an external perception producer
    (the dashboard's perception loop, later a DeepStream probe) publishes one
    of these per camera per perception tick on the Backbone's dedicated
    ingest port (``ingestion.points.listen_port``) — never on the outbound
    bus or MQTT. The Backbone in ``ingestion.mode: points`` pairs them by
    ``ts`` and runs its metric pipeline exactly as if a detector had produced
    them.

    Contract rules:
      * ``ts`` is the ``capture_ts`` of the SOURCE FRAME the detections were
        computed on (the single KPI clock), not the send time.
      * One message per camera per tick EVEN WHEN ``dets`` IS EMPTY — the
        explicit-empty heartbeat is what lets the Backbone tell "empty scene"
        from "dead producer" (silence ⇒ runtime degradation, as if the camera
        died).
      * ``seq`` is a per-camera monotonic counter: gaps = UDP loss, visible
        in diagnostics instead of silent.
      * ``config_fingerprint`` lets the Backbone detect producer/engine
        config drift (model, zones, calibration) and flag it — warn, never
        drop.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(default=SCHEMA_VERSION, ge=1)
    type: Literal[MessageType.DETECTION_SET] = MessageType.DETECTION_SET
    ts: float = Field(..., description="capture_ts of the source frame")
    camera_id: str
    frame_wh: tuple[int, int]
    seq: int = Field(..., ge=0)
    producer_id: str = "monitor_web"
    config_fingerprint: str | None = None
    dets: tuple[WireDetection, ...]


class EtagereCellState(BaseModel):
    """One shelf cell (row r, col c, 1-based) of an étagère grid."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    r: int = Field(..., ge=1)
    c: int = Field(..., ge=1)
    state: Literal["filled", "empty", "unknown"]
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class EtagereStateMessage(BaseModel):
    """Occupancy matrix of one étagère (bin rack) as seen by one camera.

    Produced RAW by the perception producer (isistream) per tick on the
    Backbone's points ingest port (``stabilized=False``), and re-published
    STABILISED by the Backbone (``stabilized=True``) on the metadata sinks —
    on MQTT retained at ``{prefix}/etagere/{zone_id}``. ``cells`` holds
    exactly ``rows*cols`` entries in reading order (r1c1, r1c2, …).
    ``unknown`` = no confident detection for that cell.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(default=SCHEMA_VERSION, ge=1)
    type: Literal[MessageType.ETAGERE_STATE] = MessageType.ETAGERE_STATE
    ts: float = Field(..., description="capture_ts of the source frame")
    camera_id: str
    zone_id: str
    name: str = ""
    rows: int = Field(3, ge=1)
    cols: int = Field(3, ge=1)
    cells: tuple[EtagereCellState, ...]
    seq: int = Field(0, ge=0)
    producer_id: str = ""
    config_fingerprint: str | None = None
    stabilized: bool = False


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
    sources: dict[str, str]   # camera_id → "alive" | "exited" | "crashed" | "waiting" (points mode, pre-first-set)
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
    # STABLE zone identity — makes the retained ConfigMessage advert
    # self-describing by id, so a gateway can join zone_state/passing topics
    # (keyed by id) back to a named, polygon-carrying zone. Defaulted "" so a
    # legacy retained advert on the broker still parses.
    zone_id: str = ""
    kind: str
    type: str
    severity: str
    polygon: list[list[float]]   # [[x, y], ...] in meters
    # Height (meters) of the plane this zone's polygon lives on — mirrors
    # ``Zone.z_base_m`` (0.0 = floor). Additive within v6, defaulted so a
    # legacy retained advert (pre-z_base_m Backbone) still parses.
    z_base_m: float = 0.0


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
    | DetectionSetMessage
    | EtagereStateMessage
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
    if msg_type == MessageType.DETECTION_SET.value:
        return DetectionSetMessage.model_validate(data)
    if msg_type == MessageType.ETAGERE_STATE.value:
        return EtagereStateMessage.model_validate(data)
    raise ValueError(f"unknown message type: {msg_type!r}")
