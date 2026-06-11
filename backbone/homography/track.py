"""Internal per-track state used by the tracker + stabilizer.

Carries everything the homography layer needs about a tracked object:

* A 2D constant-velocity Kalman filter on ``[X, Y, vX, vY]`` (FilterPy).
* Identity (``track_id``).
* Lifecycle state (``NEW`` → ``TRACKED`` → ``LOST`` → ``REMOVED``).
* Class history for the temporal stabilizer's majority vote.
* Bookkeeping (last_seen_ts, hit_streak, lost_frames).

The same ``InternalTrack`` instance is shared between ``ByteTrackMeters``
(uses the Kalman for motion prediction during matching) and
``TemporalStabilizer`` (reads Kalman state + class history when emitting
``Track2D`` for the bus). Single source of truth per object.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from filterpy.kalman import KalmanFilter


class TrackState(str, Enum):
    NEW = "new"           # spawned this frame, not yet confirmed
    TRACKED = "tracked"   # confirmed, currently observed
    LOST = "lost"         # confirmed but not seen this frame
    REMOVED = "removed"   # given up — will not match again


@dataclass(slots=True)
class TrackConfig:
    """Per-Kalman + lifecycle tunables. Defaults match a 30 FPS RTSP rig."""

    process_noise: float = 0.1          # m^2 / s^2 — acceleration noise
    measurement_noise_m: float = 0.05   # m std — derives R = (this)^2 * I
    min_hits_to_confirm: int = 3        # NEW → TRACKED after this many consecutive hits
    max_lost_frames: int = 30           # TRACKED/LOST → REMOVED after this many missed updates
    class_history_window: int = 5       # frames retained for majority vote


def _build_kalman(initial_xy: tuple[float, float], cfg: TrackConfig) -> KalmanFilter:
    """Constant-velocity 2D Kalman initialized at the observed position."""
    kf = KalmanFilter(dim_x=4, dim_z=2)
    # State: [X, Y, vX, vY]; measurement: [X, Y].
    kf.x = np.array([initial_xy[0], initial_xy[1], 0.0, 0.0], dtype=np.float64)
    # F is rebuilt per predict() with the actual dt — see InternalTrack.predict.
    kf.F = np.eye(4)
    kf.H = np.array(
        [[1.0, 0.0, 0.0, 0.0],
         [0.0, 1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    r = cfg.measurement_noise_m ** 2
    kf.R = np.array([[r, 0.0], [0.0, r]], dtype=np.float64)
    # Initial state covariance — wide on velocity (we don't know it yet),
    # tight on position (we just measured it).
    kf.P = np.diag([r, r, 1.0, 1.0])
    # Q rebuilt per predict() based on dt and process_noise.
    kf.Q = np.eye(4) * cfg.process_noise
    return kf


def _discrete_white_noise_q(dt: float, sigma2: float) -> np.ndarray:
    """Continuous-time process-noise integral for the 4D constant-velocity model."""
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt3 * dt
    pos = dt4 / 4.0
    pv = dt3 / 2.0
    vel = dt2
    return sigma2 * np.array(
        [[pos, 0.0, pv,  0.0],
         [0.0, pos, 0.0, pv ],
         [pv,  0.0, vel, 0.0],
         [0.0, pv,  0.0, vel]],
        dtype=np.float64,
    )


@dataclass(slots=True)
class InternalTrack:
    """Mutable internal track record."""

    track_id: int
    cls: str
    confidence: float
    cameras_seeing: tuple[str, ...]
    capture_ts: float
    cfg: TrackConfig
    kf: KalmanFilter = field(init=False)
    state: TrackState = TrackState.NEW
    hit_streak: int = 1
    lost_frames: int = 0
    class_history: deque[str] = field(default_factory=lambda: deque(maxlen=5))

    def __post_init__(self) -> None:
        # The dataclass receives `initial_xy` indirectly via __init__; we
        # require callers to construct via `InternalTrack.create()`.
        if not hasattr(self, "kf") or self.kf is None:
            raise RuntimeError(
                "InternalTrack must be created via InternalTrack.create(); "
                "direct construction is not supported."
            )

    @classmethod
    def create(
        cls,
        *,
        track_id: int,
        cls_label: str,
        xy_m: tuple[float, float],
        confidence: float,
        cameras_seeing: tuple[str, ...],
        capture_ts: float,
        cfg: TrackConfig | None = None,
    ) -> InternalTrack:
        config = cfg or TrackConfig()
        kf = _build_kalman(xy_m, config)
        history: deque[str] = deque(maxlen=config.class_history_window)
        history.append(cls_label)
        obj = cls.__new__(cls)
        obj.track_id = track_id
        obj.cls = cls_label
        obj.confidence = float(confidence)
        obj.cameras_seeing = tuple(cameras_seeing)
        obj.capture_ts = float(capture_ts)
        obj.cfg = config
        obj.kf = kf
        # If a single hit is enough to confirm, the track is already TRACKED
        # on creation (matches the "min_hits_to_confirm=1 → publishable on
        # frame 1" semantics used by ByteTrack callers).
        obj.state = (
            TrackState.TRACKED if config.min_hits_to_confirm <= 1 else TrackState.NEW
        )
        obj.hit_streak = 1
        obj.lost_frames = 0
        obj.class_history = history
        return obj

    # ----- Kalman driving -----

    def predict(self, dt: float) -> None:
        """Advance the Kalman state forward by ``dt`` seconds."""
        if dt < 0.0:
            dt = 0.0
        self.kf.F = np.array(
            [[1.0, 0.0, dt,  0.0],
             [0.0, 1.0, 0.0, dt ],
             [0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        self.kf.Q = _discrete_white_noise_q(dt, self.cfg.process_noise)
        self.kf.predict()

    def update_with(
        self,
        *,
        xy_m: tuple[float, float],
        cls_label: str,
        confidence: float,
        cameras_seeing: tuple[str, ...],
        capture_ts: float,
    ) -> None:
        """Apply a measurement and refresh lifecycle bookkeeping."""
        self.kf.update(np.asarray(xy_m, dtype=np.float64))
        self.class_history.append(cls_label)
        self.confidence = float(confidence)
        self.cameras_seeing = tuple(cameras_seeing)
        self.capture_ts = float(capture_ts)
        self.cls = cls_label   # latest class; stabilizer can override at publish
        self.hit_streak += 1
        self.lost_frames = 0
        if self.state == TrackState.NEW and self.hit_streak >= self.cfg.min_hits_to_confirm:
            self.state = TrackState.TRACKED
        elif self.state == TrackState.LOST:
            self.state = TrackState.TRACKED

    def mark_missed(self) -> None:
        """Lifecycle update when no observation matched this frame."""
        self.hit_streak = 0
        self.lost_frames += 1
        if self.state == TrackState.NEW:
            # Unconfirmed tracks die immediately on a miss — flicker suppression.
            self.state = TrackState.REMOVED
        elif self.state == TrackState.TRACKED:
            self.state = TrackState.LOST
        if self.lost_frames > self.cfg.max_lost_frames:
            self.state = TrackState.REMOVED

    # ----- Read-out (for ByteTrack matching + stabilizer publish) -----

    def xy(self) -> tuple[float, float]:
        return float(self.kf.x[0]), float(self.kf.x[1])

    def vxy(self) -> tuple[float, float]:
        return float(self.kf.x[2]), float(self.kf.x[3])

    def position_covariance(self) -> np.ndarray:
        """2x2 position covariance, used by Mahalanobis matching."""
        return self.kf.P[:2, :2].copy()

    @property
    def is_active(self) -> bool:
        """Tracks still eligible for matching — NEW, TRACKED, or LOST.

        LOST tracks must be matchable so they can recover within
        ``max_lost_frames``; otherwise brief occlusions would spawn a
        new track_id each time the detection comes back.
        """
        return self.state in (TrackState.NEW, TrackState.TRACKED, TrackState.LOST)

    @property
    def is_publishable(self) -> bool:
        """Confirmed tracks (NEW after min_hits → TRACKED) are publishable.

        LOST tracks are not published — they're alive internally but the bus
        should not see them until they recover to TRACKED.
        """
        return self.state == TrackState.TRACKED
