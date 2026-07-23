"""Producer-side One Euro pose smoothing (isistream/pose_smooth.py).

Pins the two properties that make the filter correct for the wire:
jitter suppression at rest with low lag in motion, and the hard constraint
that only person ``keypoints_uv`` change — ``foot_uv`` (the geometry input)
and non-person detections pass through bit-identical.
"""

from __future__ import annotations

import numpy as np
import pytest

from backbone.core.types import Detection
from isistream.pose_smooth import OneEuro, PoseSmoother

FPS = 15.0
DT = 1.0 / FPS


def _person(foot, kps, conf=0.9):
    kps = np.asarray(kps, dtype=np.float32).reshape(-1, 3)
    x, y = float(foot[0]), float(foot[1])
    return Detection(
        camera_id="cam_a", cls="person", confidence=conf,
        bbox_xyxy=(x - 40, y - 160, x + 40, y),
        foot_uv=(x, y), capture_ts=0.0, keypoints_uv=kps,
    )


def _object(foot=(300.0, 300.0)):
    return Detection(
        camera_id="cam_a", cls="palette", confidence=0.8,
        bbox_xyxy=(0, 0, 50, 50), foot_uv=foot, capture_ts=0.0,
    )


# ---------------------------------------------------------------- OneEuro

def test_one_euro_suppresses_noise_on_static_signal():
    rng = np.random.default_rng(0)
    f = OneEuro()
    xs = 100.0 + rng.normal(0.0, 2.0, size=90)
    out = [f.filter(float(x), i * DT) for i, x in enumerate(xs)]
    assert np.var(out[15:]) < 0.25 * np.var(xs[15:])


def test_one_euro_low_lag_on_ramp():
    f = OneEuro()
    speed = 100.0  # px/s
    lag = 0.0
    for i in range(60):
        t = i * DT
        x = speed * t
        lag = x - f.filter(x, t)
    assert 0.0 <= lag < 15.0


def test_one_euro_first_sample_passthrough():
    f = OneEuro()
    assert f.filter(42.0, 0.0) == pytest.approx(42.0)


def test_one_euro_converges_after_step():
    f = OneEuro()
    f.filter(0.0, 0.0)
    out = 0.0
    for i in range(1, 46):
        out = f.filter(100.0, i * DT)
    assert out == pytest.approx(100.0, abs=1.0)


# ------------------------------------------------------------ PoseSmoother

def test_new_person_first_output_is_raw():
    s = PoseSmoother()
    kps = [[10, 20, 0.9], [30, 40, 0.8]]
    out = s.smooth("cam_a", [_person((100, 400), kps)], t=0.0)
    np.testing.assert_allclose(out[0].keypoints_uv, np.asarray(kps, np.float32))


def test_keypoints_smoothed_but_foot_and_conf_raw():
    s = PoseSmoother()
    rng = np.random.default_rng(1)
    outs = []
    for i in range(30):
        noisy = [[10 + rng.normal(0, 3), 20 + rng.normal(0, 3), 0.7]]
        d = _person((100.0, 400.0), noisy)
        out = s.smooth("cam_a", [d], t=i * DT)[0]
        assert out.foot_uv == d.foot_uv                      # geometry untouched
        assert out.keypoints_uv[0, 2] == pytest.approx(0.7)  # conf raw
        outs.append(out.keypoints_uv[0, 0])
    assert np.var(outs[10:]) < 0.5 * 9.0  # smoothed ≪ input variance (σ²=9)


def test_non_person_dets_pass_through_identically():
    s = PoseSmoother()
    obj = _object()
    out = s.smooth("cam_a", [obj, _person((100, 400), [[1, 2, 0.5]])], t=0.0)
    assert out[0] is obj


def test_association_keeps_filters_separate_per_person():
    s = PoseSmoother()
    a0, b0 = [[100, 100, 0.9]], [[500, 500, 0.9]]
    s.smooth("cam_a", [_person((100, 100), a0), _person((500, 500), b0)], t=0.0)
    # order flips in the next frame; slots must follow the nearer foot
    out = s.smooth(
        "cam_a",
        [_person((502, 502), [[502, 502, 0.9]]), _person((102, 102), [[102, 102, 0.9]])],
        t=DT,
    )
    by_foot = {round(o.foot_uv[0]): o for o in out}
    # each smoothed keypoint stays near its own history, not the other person's
    assert abs(by_foot[502].keypoints_uv[0, 0] - 500) < 5
    assert abs(by_foot[102].keypoints_uv[0, 0] - 100) < 5


def test_stale_slot_pruned_and_reseeded():
    s = PoseSmoother(stale_s=2.0)
    s.smooth("cam_a", [_person((100, 400), [[10, 20, 0.9]])], t=0.0)
    # absent for > stale_s, then reappears far away: fresh seed = raw output
    out = s.smooth("cam_a", [_person((100, 400), [[90, 80, 0.9]])], t=5.0)
    np.testing.assert_allclose(
        out[0].keypoints_uv, np.asarray([[90, 80, 0.9]], np.float32))


def test_cameras_have_independent_state():
    s = PoseSmoother()
    s.smooth("cam_a", [_person((100, 400), [[10, 20, 0.9]])], t=0.0)
    out_b = s.smooth("cam_b", [_person((100, 400), [[70, 60, 0.9]])], t=DT)
    np.testing.assert_allclose(
        out_b[0].keypoints_uv, np.asarray([[70, 60, 0.9]], np.float32))
