"""3D-localization axis overlay for the CAM views.

Mirrors the floor map's axis gizmo (`static/js/floor_map_3d.js`) on the live
camera panels: for every object with a FRESH two-view ``Track3D`` fix on the
UDP bus, draw a small XYZ axis at the object's floor point plus the measured
height in the same white rounded badge the distance lines use — projected
into the camera through its FULL model (``cv2.projectPoints`` with K, D and
the camera←world extrinsic inverted from the stored world←camera pose).

Consumer-side only: data comes from the dashboard's :class:`BusSubscriber`
snapshot (``last_track3d_by_id``), calibration from the same rig the warp
views load. Mode-1 calibrations carry placeholder extrinsics (K=I, R=I, t=0)
— the overlay detects that and stays silently disabled for such a camera.
The map's gizmo is unchanged; this is the cam-view twin of it.
"""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from .overlay import _draw_text_badge

logger = logging.getLogger(__name__)

FRESH_S = 1.5          # 3D fix older than this vs the newest bus ts ⇒ stale
AXIS_M = 0.45          # metres — X/Y arrow length (same as the map gizmo)
Z_MIN_M, Z_MAX_M = 0.3, 2.5   # Z arrow length clamp (visible even at z≈0)

# BGR — match the map gizmo colours (0xef4444 / 0x22c55e / 0x3b82f6).
_X_COLOR = (68, 68, 239)      # red
_Y_COLOR = (94, 197, 34)      # green
_Z_COLOR = (246, 130, 59)     # blue


def has_metric_extrinsics(view) -> bool:
    """True iff ``view`` carries a usable metric camera model.

    Mode-1 single-cam calibrations store placeholders (``K=I, D=0, R=I, t=0``)
    plus a real ``H`` — projecting 3D points through those is meaningless, so
    the axis overlay must skip such cameras gracefully."""
    try:
        K = np.asarray(view.K, dtype=np.float64)
        R = np.asarray(view.R, dtype=np.float64)
        t = np.asarray(view.t, dtype=np.float64).reshape(-1)
    except Exception:
        return False
    if K.shape != (3, 3) or R.shape != (3, 3) or t.shape != (3,):
        return False
    if np.allclose(K, np.eye(3)):
        return False
    return not (np.allclose(R, np.eye(3)) and np.allclose(t, 0.0))


def fresh_tracks3d(snapshot, *, now_ts: float | None = None,
                   fresh_s: float = FRESH_S) -> list:
    """The FRESH, two-view ``Track3DMessage``s from a :class:`BusState` snapshot.

    Freshness is a SAME-CLOCK comparison on ``capture_ts``: the reference is
    the newest capture ts on the bus (observations + Track2D — the streams
    that keep flowing when the 3D subscription lapses), so a stale
    ``last_track3d_by_id`` leftover never keeps the axis alive (same rule as
    the map gizmo). Falls back to wall clock when the bus carries nothing
    else (same host; pipeline latency ≪ ``fresh_s``). ``single_view`` fixes
    are floor-pinned by design — no axis."""
    by3 = getattr(snapshot, "last_track3d_by_id", None) or {}
    if not by3:
        return []
    ref = now_ts
    if ref is None:
        others = [m.ts for m in (getattr(snapshot, "observations_by_camera", None) or {}).values()]
        others += [m.ts for m in (getattr(snapshot, "last_track2d_by_id", None) or {}).values()]
        ref = max(others) if others else time.time()
    return [t3 for t3 in by3.values()
            if not t3.single_view and (ref - t3.ts) < fresh_s]


class CamAxisOverlay:
    """Per-camera renderer: XYZ axis + height badge at every fresh Track3D fix.

    Built ONCE per stream build (`build_cam_stream`) — the camera←world
    ``rvec``/``tvec`` are inverted and cached at construction, never per
    frame. ``draw`` never raises: any failure logs (throttled) and leaves the
    frame untouched, so the video stream cannot die on a bad fix.
    """

    def __init__(self, view, bus_getter) -> None:
        self._get_bus = bus_getter
        self._enabled = False
        self._warn_ts = 0.0
        if view is None or not has_metric_extrinsics(view):
            return                       # Mode 1 / uncalibrated ⇒ silently off
        R = np.asarray(view.R, dtype=np.float64)
        t = np.asarray(view.t, dtype=np.float64).reshape(3)
        # Stored (R, t) are the camera POSE (world←camera — see
        # backbone.shared.geometry.projection_from_K_R_t); cv2.projectPoints
        # wants the camera←world extrinsic, i.e. the inverse.
        self._R_cw = R.T
        self._t_cw = -self._R_cw @ t
        self._rvec, _ = cv2.Rodrigues(self._R_cw)
        self._K = np.asarray(view.K, dtype=np.float64)
        self._D = np.asarray(view.D, dtype=np.float64)
        self._cal_wh = (int(view.image_size_wh[0]), int(view.image_size_wh[1]))
        self._enabled = True

    def project(self, world3: np.ndarray, frame_wh: tuple[int, int]):
        """Project Nx3 world metres → Nx2 DISPLAY-frame pixels + a per-point
        in-front-of-camera mask. Scales calibration-frame pixels to the live
        frame size (the same guard the distance overlay applies to ``H``)."""
        world3 = np.asarray(world3, dtype=np.float64).reshape(-1, 3)
        cam_z = (self._R_cw @ world3.T).T[:, 2] + self._t_cw[2]
        uv, _ = cv2.projectPoints(world3, self._rvec, self._t_cw, self._K, self._D)
        uv = uv.reshape(-1, 2)
        sx = frame_wh[0] / float(self._cal_wh[0])
        sy = frame_wh[1] / float(self._cal_wh[1])
        return uv * (sx, sy), cam_z > 1e-6

    def draw(self, image) -> None:
        """Overlay every fresh fix onto ``image`` in place (BGR). No-op when
        disabled, the bus is absent, or nothing fresh is on it."""
        if not self._enabled:
            return
        try:
            bus = self._get_bus()
            if bus is None:
                return
            tracks = fresh_tracks3d(bus.snapshot())
            for t3 in tracks:
                self._draw_one(image, t3)
        except Exception:
            now = time.monotonic()
            if now - self._warn_ts > 30.0:
                self._warn_ts = now
                logger.warning("3D axis overlay failed (throttled 30s)", exc_info=True)

    def _draw_one(self, image, t3) -> None:
        h, w = image.shape[:2]
        x, y = float(t3.xyz_m[0]), float(t3.xyz_m[1])
        z = max(0.0, float(t3.xyz_m[2]))
        z_len = max(Z_MIN_M, min(Z_MAX_M, z))       # clamp like the map gizmo
        world = [(x, y, 0.0),                        # origin (floor point)
                 (x + AXIS_M, y, 0.0),               # X tip
                 (x, y + AXIS_M, 0.0),               # Y tip
                 (x, y, z_len)]                      # Z tip
        uv, in_front = self.project(world, (w, h))
        if not in_front[0]:
            return
        ox, oy = uv[0]
        if not (-0.25 * w <= ox < 1.25 * w and -0.25 * h <= oy < 1.25 * h):
            return                                   # anchor far off-screen
        origin = (round(float(ox)), round(float(oy)))
        for i, color in ((1, _X_COLOR), (2, _Y_COLOR), (3, _Z_COLOR)):
            if not in_front[i]:
                continue
            ex, ey = uv[i]
            # Skip endpoints that exploded wildly outside the frame (distortion
            # polynomial divergence / near-degenerate geometry).
            if not (-w <= ex < 2 * w and -h <= ey < 2 * h):
                continue
            tip = (round(float(ex)), round(float(ey)))
            cv2.arrowedLine(image, origin, tip, color, 2, cv2.LINE_AA, tipLength=0.25)
            if i == 3:                               # height badge at the Z tip
                _draw_text_badge(image, tip, f"{z:.2f} m")
