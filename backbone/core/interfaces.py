"""The five plugin seams of the Backbone — and only these.

Each seam is an ABC plus a module-level `Registry` instance. Everywhere else
in the codebase, prefer concrete code: ABCs here are justified by genuine
multiplicity (RTSP vs replay; YOLO vs YOLO-pose; ByteTrack vs SORT; 2-cam DLT
vs N-cam aniposelib; UDP vs MQTT vs S7).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

import numpy as np

from .registry import Registry
from .types import Detection, Frame, FramePair, Track2D, Track3D


class FrameSource(ABC):
    """Yields `Frame` objects from one camera. Each instance owns one stream."""

    @property
    @abstractmethod
    def camera_id(self) -> str: ...

    @abstractmethod
    def frames(self) -> Iterator[Frame]:
        """Iterate frames until the source is stopped or exhausted."""

    @abstractmethod
    def stop(self) -> None: ...


class Detector(ABC):
    """Runs detection on a synchronized `FramePair` (or larger batch in the future).

    Implementations must be batched — the 2-camera node feeds both cameras in
    a single inference call (batch=2) on the same CUDA stream.
    """

    @abstractmethod
    def warmup(self) -> None: ...

    @abstractmethod
    def detect(self, pair: FramePair) -> dict[str, list[Detection]]:
        """Return detections keyed by camera_id."""


class Tracker(ABC):
    """Maintains identity across frames.

    Operates in **metric (meters)** space, not pixel space — see the architecture
    principle "one identity space". `update` is called once per FramePair with
    pre-fused metric observations.
    """

    @abstractmethod
    def update(
        self,
        capture_ts: float,
        observations: list[tuple[str, tuple[float, float], float, tuple[str, ...]]],
    ) -> list[Track2D]:
        """Update with `(cls, xy_m, confidence, cameras_seeing)` observations."""


class Triangulator(ABC):
    """Lifts 2D observations across views to 3D `(X, Y, Z)`.

    Identity is supplied by the caller — the triangulator never re-IDs. Every
    output must have already passed a reprojection-error gate.

    Pose-mode (``triangulate_keypoints``) is part of the architecture but lands
    in S5.5 — when the Sécurité module's fall-detection subscription needs
    per-keypoint 3D. The current single method covers all v1 use cases.
    """

    @abstractmethod
    def triangulate_point(
        self,
        observations: dict[str, tuple[float, float]],
    ) -> np.ndarray | None:
        """Return XYZ in meters, or None if geometry is degenerate."""


class MetadataSink(ABC):
    """Publishes ``Track2D``, ``Track3D``, and event envelopes to the outside world."""

    @abstractmethod
    def publish_track_2d(self, track: Track2D) -> None: ...

    @abstractmethod
    def publish_track_3d(self, track: Track3D) -> None: ...

    def publish_event(self, event: object) -> None:
        """Publish a ``PassingEvent`` (or future event type).

        Non-abstract default no-op: existing sink implementations and test mocks
        that don't override this still satisfy the ABC, and the five-seam count
        is unchanged. Override in sink plugins to emit zone-crossing events.
        """
        return None

    def publish_image_ref(
        self,
        track_id: int,
        cls: str,
        zone: str,
        ts: float,
        url: str,
        zone_id: str = "",
    ) -> None:
        """Publish an image-reference URL for a zone-passing snapshot.

        Non-abstract default no-op so existing sink implementations that do
        not override this remain valid.  Override in sink plugins to emit
        the ``ImageRefMessage`` (URL only — never raw bytes). ``zone_id`` is
        the STABLE zone identity (defaulted for back-compat callers).
        """
        return None

    def publish_zone_state(self, msg: object) -> None:
        """Publish a ``ZoneStateMessage`` — one zone's current object list.

        Non-abstract default no-op: existing sink implementations and test
        mocks that don't override this still satisfy the ABC, and the
        five-seam count is unchanged.  MQTT sinks override this to publish
        retained on the per-zone topic.
        """
        return None

    def publish_etagere_state(self, msg: object) -> None:
        """Publish an ``EtagereStateMessage`` — one shelf rack's cell matrix.

        Non-abstract default no-op (same rationale as ``publish_zone_state``);
        MQTT sinks override to publish retained on ``{prefix}/etagere/{zone_id}``.
        """
        return None

    def publish_observations(self, msg: object) -> None:
        """Publish an ``ObservationsMessage`` — per-camera raw detections.

        Non-abstract default no-op (same rationale as ``publish_zone_state``).
        Display concern: only the UDP sink implements it; MQTT sinks keep the
        no-op so the broker never carries per-frame detections.
        """
        return None

    def publish_proximity(self, msg: object) -> None:
        """Publish a ``ProximityMessage`` — person↔object floor distances.

        Non-abstract default no-op (same rationale as ``publish_zone_state``):
        existing sinks/mocks stay valid; the five-seam count is unchanged.
        MQTT sinks override this to publish retained on ``{prefix}/proximity``.
        """
        return None

    def publish_diagnostics(self, msg: object) -> None:
        """Publish a ``DiagnosticsMessage`` heartbeat (Phase 1 distributed).

        Non-abstract default no-op: existing sink implementations and test
        mocks that don't override this still satisfy the ABC, and the
        five-seam count is unchanged.  Override in sink plugins.
        """
        return None

    def publish_config(self, msg: object) -> None:
        """Publish a retained ``ConfigMessage`` advertisement (Phase 1 distributed).

        Non-abstract default no-op so existing implementations remain valid.
        MQTT sinks override this to publish with ``retain=True``.
        """
        return None

    def advertise_zones(self, zones: list[tuple[str, str]]) -> None:
        """Announce the ACTIVE zone set as ``(name, zone_id)`` pairs, once at startup.

        Non-abstract default no-op (same rationale as ``publish_zone_state``):
        existing sinks/mocks stay valid; the five-seam count is unchanged.
        The MQTT sink overrides this to reconcile its retained per-zone topics
        against the active set — zones deleted from config would otherwise
        leave stale retained ``ZoneStateMessage``s on the broker forever.
        """
        return None

    @abstractmethod
    def close(self) -> None: ...


frame_source_registry: Registry[FrameSource] = Registry("FrameSource")
detector_registry: Registry[Detector] = Registry("Detector")
tracker_registry: Registry[Tracker] = Registry("Tracker")
triangulator_registry: Registry[Triangulator] = Registry("Triangulator")
metadata_sink_registry: Registry[MetadataSink] = Registry("MetadataSink")
