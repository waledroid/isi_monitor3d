"""3D-localization axis + height badge overlay on the CAM views.

Hermetic: a synthetic overhead camera (simple K, involutory R, t = camera at
3 m above the origin) makes world→pixel projection analytic; the bus is a
stub whose ``snapshot()`` returns a plain namespace shaped like ``BusState``.
No model, no camera, no network.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
from backbone.comms.schemas import Track3DMessage

from monitor_web.track3d_overlay import (
    CamAxisOverlay,
    fresh_tracks3d,
    has_metric_extrinsics,
)


def _view(wh: tuple[int, int] = (640, 480)) -> SimpleNamespace:
    """Camera at (0, 0, 3) looking straight down: R_pose = R_cw = diag-ish
    involutory matrix, so world (X, Y, Z) → cam (X, -Y, 3 - Z)."""
    R = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    return SimpleNamespace(
        K=np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]]),
        D=np.zeros(5),
        R=R,                     # world←camera pose (its own inverse here)
        t=np.array([0.0, 0.0, 3.0]),
        image_size_wh=wh,
    )


def _mode1_view() -> SimpleNamespace:
    """Mode-1 single-cam placeholder extrinsics: K=I, D=0, R=I, t=0."""
    return SimpleNamespace(K=np.eye(3), D=np.zeros(5), R=np.eye(3),
                           t=np.zeros(3), image_size_wh=(640, 480))


def _t3(ts: float, *, xyz=(0.5, 0.4, 1.2), single_view: bool = False,
        track_id: int = 7) -> Track3DMessage:
    return Track3DMessage(
        ts=ts, track_id=track_id, cls="palette", xyz_m=xyz,
        vxyz_m=(0.0, 0.0, 0.0), contributing_cameras=("cam_a", "cam_b"),
        max_reprojection_error_px=1.0, single_view=single_view,
    )


def _snap(tracks3d, ref_ts: float | None = None) -> SimpleNamespace:
    obs = {}
    if ref_ts is not None:
        obs["cam_a"] = SimpleNamespace(ts=ref_ts)   # freshness reference clock
    return SimpleNamespace(
        last_track3d_by_id={t.track_id: t for t in tracks3d},
        observations_by_camera=obs,
        last_track2d_by_id={},
    )


def _bus(snap) -> SimpleNamespace:
    return SimpleNamespace(snapshot=lambda: snap)


# ---- extrinsics / freshness gates ----


def test_metric_extrinsics_detected() -> None:
    assert has_metric_extrinsics(_view())
    assert not has_metric_extrinsics(_mode1_view())          # Mode-1 placeholders
    assert not has_metric_extrinsics(SimpleNamespace())      # no arrays at all


def test_fresh_tracks3d_gates_stale_and_single_view() -> None:
    now = 1000.0
    fresh = _t3(now - 0.2, track_id=1)
    stale = _t3(now - 3.0, track_id=2)
    solo = _t3(now - 0.2, single_view=True, track_id=3)
    out = fresh_tracks3d(_snap([fresh, stale, solo], ref_ts=now))
    assert [t.track_id for t in out] == [1]


def test_fresh_tracks3d_reference_is_bus_capture_clock() -> None:
    # A 3D leftover from a lapsed subscription must go stale as the 2D/obs
    # streams keep advancing the reference ts — even though wall clock says
    # nothing (same-clock comparison).
    old = _t3(1000.0)
    assert fresh_tracks3d(_snap([old], ref_ts=1000.5)) == [old]
    assert fresh_tracks3d(_snap([old], ref_ts=1003.0)) == []


# ---- projection ----


def test_axis_endpoints_project_analytically() -> None:
    ov = CamAxisOverlay(_view(), lambda: None)
    uv, in_front = ov.project([(0.0, 0.0, 0.0), (0.45, 0.0, 0.0),
                               (0.0, 0.45, 0.0), (0.0, 0.0, 1.2)], (640, 480))
    assert in_front.all()
    # cam (X, -Y, 3-Z), f=800, c=(320,240): origin → centre; X tip → +120 px u;
    # Y tip → -120 px v; Z tip stays centred (straight-down camera).
    np.testing.assert_allclose(uv[0], (320.0, 240.0), atol=1e-6)
    np.testing.assert_allclose(uv[1], (440.0, 240.0), atol=1e-6)
    np.testing.assert_allclose(uv[2], (320.0, 120.0), atol=1e-6)
    np.testing.assert_allclose(uv[3], (320.0, 240.0), atol=1e-6)


def test_project_scales_to_display_frame() -> None:
    ov = CamAxisOverlay(_view(wh=(640, 480)), lambda: None)
    uv, _ = ov.project([(0.0, 0.0, 0.0)], (1280, 960))     # 2x display frame
    np.testing.assert_allclose(uv[0], (640.0, 480.0), atol=1e-6)


# ---- drawing ----


def test_fresh_fix_draws_axis_and_badge() -> None:
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    ov = CamAxisOverlay(_view(), lambda: _bus(_snap([_t3(time.time())])))
    ov.draw(img)
    assert img.any()                                  # something was drawn
    b, g, r = img[..., 0], img[..., 1], img[..., 2]
    assert (r.astype(int) - b > 100).any()            # a red-ish X axis pixel
    assert (g.astype(int) - b > 80).any()             # a green-ish Y axis pixel
    # the white rounded badge (255,255,255) with black text on it
    white = (b == 255) & (g == 255) & (r == 255)
    assert white.sum() > 20


def test_stale_fix_draws_nothing() -> None:
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    ov = CamAxisOverlay(_view(), lambda: _bus(_snap([_t3(1000.0)], ref_ts=1005.0)))
    ov.draw(img)
    assert not img.any()


def test_single_view_fix_draws_nothing() -> None:
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    ov = CamAxisOverlay(
        _view(), lambda: _bus(_snap([_t3(time.time(), single_view=True)])))
    ov.draw(img)
    assert not img.any()


def test_mode1_placeholders_disable_overlay_without_crash() -> None:
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    ov = CamAxisOverlay(_mode1_view(), lambda: _bus(_snap([_t3(time.time())])))
    ov.draw(img)
    assert not img.any()


def test_missing_view_and_bad_bus_never_raise() -> None:
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    CamAxisOverlay(None, lambda: None).draw(img)             # no calibration
    ov = CamAxisOverlay(_view(), lambda: (_ for _ in ()).throw(RuntimeError))
    ov.draw(img)                                             # bus getter blows up
    assert not img.any()


def test_offscreen_anchor_draws_nothing() -> None:
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    far = _t3(time.time(), xyz=(50.0, 50.0, 1.0))            # projects far off-frame
    ov = CamAxisOverlay(_view(), lambda: _bus(_snap([far])))
    ov.draw(img)
    assert not img.any()
