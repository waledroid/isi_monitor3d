"""``Tracker3D`` — per-track 3D Kalman, keyed by 2D ``track_id``.

Identity is **inherited** from the homography tracker. ``Tracker3D`` never
spawns or renames tracks — it only adds Z plus a velocity estimate to a
``track_id`` that S4 already minted.

State per track: ``[X, Y, Z, vX, vY, vZ]``, constant-velocity model, FilterPy
``KalmanFilter`` with discrete-white-noise process covariance scaled by
``dt``. Same flavor as the 2D Kalman in ``backbone.homography.track`` —
generalized to 3D.

Lifecycle: the orchestrator (S6) is responsible for calling ``gc()`` after
each frame with the set of currently-active 2D ``track_id``s. Any 3D Kalman
whose ``track_id`` is not in that set is dropped — keeps state proportional
to the live 2D tracker, no zombie 3D filters.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from filterpy.kalman import KalmanFilter

from backbone.core.types import Track3D


@dataclass(slots=True)
class Track3DConfig:
    """Per-Kalman tunables. Defaults assume a 30 FPS pipeline."""

    process_noise: float = 0.1            # m^2 / s^2 acceleration noise
    measurement_noise_m: float = 0.05     # std of triangulated XYZ measurement
    initial_velocity_variance: float = 1.0


def _discrete_white_noise_q_3d(dt: float, sigma2: float) -> np.ndarray:
    """Continuous-time white-noise integral for a 6D constant-velocity model."""
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt3 * dt
    pos = dt4 / 4.0
    pv = dt3 / 2.0
    vel = dt2
    q = np.zeros((6, 6), dtype=np.float64)
    for i in range(3):
        q[i, i] = pos
        q[i, i + 3] = pv
        q[i + 3, i] = pv
        q[i + 3, i + 3] = vel
    return sigma2 * q


def _build_kalman_3d(xyz_initial: np.ndarray, cfg: Track3DConfig) -> KalmanFilter:
    kf = KalmanFilter(dim_x=6, dim_z=3)
    kf.x = np.array(
        [xyz_initial[0], xyz_initial[1], xyz_initial[2], 0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    kf.F = np.eye(6)
    kf.H = np.zeros((3, 6), dtype=np.float64)
    kf.H[0, 0] = 1.0
    kf.H[1, 1] = 1.0
    kf.H[2, 2] = 1.0
    r = cfg.measurement_noise_m ** 2
    kf.R = np.eye(3) * r
    kf.P = np.diag([r, r, r,
                    cfg.initial_velocity_variance,
                    cfg.initial_velocity_variance,
                    cfg.initial_velocity_variance])
    kf.Q = np.eye(6) * cfg.process_noise
    return kf


@dataclass(slots=True)
class _Track3DState:
    track_id: int
    kf: KalmanFilter
    cls: str
    cameras_seeing: tuple[str, ...]
    last_capture_ts: float
    last_max_reproj_error_px: float
    single_view: bool = False
    confidence: float = 1.0


class Tracker3D:
    """3D Kalman manager keyed by 2D ``track_id``."""

    def __init__(self, cfg: Track3DConfig | None = None) -> None:
        self._cfg = cfg or Track3DConfig()
        self._states: dict[int, _Track3DState] = {}

    @property
    def active_track_ids(self) -> tuple[int, ...]:
        return tuple(self._states.keys())

    def update(
        self,
        *,
        track_id: int,
        xyz_obs: np.ndarray,
        capture_ts: float,
        cameras_seeing: tuple[str, ...],
        cls: str,
        max_reproj_error_px: float,
        single_view: bool = False,
        confidence: float = 1.0,
    ) -> Track3D:
        """Apply a measurement and emit a ``Track3D``.

        Spawns a new internal Kalman the first time a ``track_id`` is seen.
        ``single_view`` + ``confidence`` flow through to the emitted ``Track3D`` —
        the single-view floor fallback feeds a ``(X, Y, 0)`` measurement so the same
        Kalman smooths it (and seamlessly resumes once 2 views return).
        """
        xyz = np.asarray(xyz_obs, dtype=np.float64).reshape(3)
        state = self._states.get(track_id)
        if state is None:
            kf = _build_kalman_3d(xyz, self._cfg)
            state = _Track3DState(
                track_id=track_id,
                kf=kf,
                cls=cls,
                cameras_seeing=tuple(cameras_seeing),
                last_capture_ts=float(capture_ts),
                last_max_reproj_error_px=float(max_reproj_error_px),
                single_view=single_view,
                confidence=float(confidence),
            )
            self._states[track_id] = state
            return self._emit(state)

        dt = max(0.0, float(capture_ts) - state.last_capture_ts)
        self._advance_kalman(state.kf, dt)
        state.kf.update(xyz)
        state.cls = cls
        state.cameras_seeing = tuple(cameras_seeing)
        state.last_capture_ts = float(capture_ts)
        state.last_max_reproj_error_px = float(max_reproj_error_px)
        state.single_view = single_view
        state.confidence = float(confidence)
        return self._emit(state)

    def gc(self, active_track_ids: Iterable[int]) -> None:
        """Drop Kalmans for tracks no longer active in the 2D tracker."""
        keep = set(active_track_ids)
        for track_id in list(self._states):
            if track_id not in keep:
                del self._states[track_id]

    def drop(self, track_id: int) -> None:
        self._states.pop(track_id, None)

    # ---- internals ----

    def _advance_kalman(self, kf: KalmanFilter, dt: float) -> None:
        kf.F = np.eye(6)
        for i in range(3):
            kf.F[i, i + 3] = dt
        kf.Q = _discrete_white_noise_q_3d(dt, self._cfg.process_noise)
        kf.predict()

    def _emit(self, state: _Track3DState) -> Track3D:
        x = state.kf.x
        return Track3D(
            track_id=state.track_id,
            cls=state.cls,
            capture_ts=state.last_capture_ts,
            xyz_m=(float(x[0]), float(x[1]), float(x[2])),
            vxyz_m=(float(x[3]), float(x[4]), float(x[5])),
            contributing_cameras=state.cameras_seeing,
            max_reprojection_error_px=state.last_max_reproj_error_px,
            keypoints_xyz=None,
            single_view=state.single_view,
            confidence=state.confidence,
        )


# Re-exported for callers that want to construct the configuration explicitly.
__all__ = ["Track3DConfig", "Tracker3D"]
