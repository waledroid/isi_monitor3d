"""Pose overlay — annotate_frame hook + display smoothing/extrapolation.

The CPU branch has no in-dashboard pose inference (PoseEngine/AsyncPoseRunner
are gone): skeletons render from the wire's person observations. What remains
to pin is the drawing hook and the pure smoothing math.
"""

from __future__ import annotations

import numpy as np

from monitor_web import detection_overlay as do


class _FakePose:
    """Stand-in pose source: records that annotate_frame invoked it."""

    def __init__(self) -> None:
        self.called = 0

    def predict(self, image):
        return ["pose"]

    def draw(self, image, poses):
        self.called += 1
        image[:] = 0   # visible side effect


def test_annotate_frame_invokes_pose_detector() -> None:
    """With a pose_detector (and no object detector), annotate_frame still runs +
    draws pose — so people render even before a detection model is set."""
    img = np.full((48, 64, 3), 200, dtype=np.uint8)
    fake = _FakePose()
    out = do.annotate_frame(img, detector=None, cam_id="cam", pose_detector=fake)
    assert fake.called == 1
    assert int(out.mean()) == 0   # _FakePose.draw zeroed it → proof it ran


def test_pose_display_smoothing_converges_and_snaps() -> None:
    """predict() blends the drawn skeleton toward the newest result per rendered
    frame (no more N-fps stepping) but SNAPS for far jumps / new persons."""
    import numpy as np

    from monitor_web.pose_overlay import Pose, _advance_smoothing

    def pose_at(x):
        return Pose(box_xyxy=np.array([x, 0.0, x + 50, 100.0]), score=0.9,
                    keypoints=np.array([[x, 10.0, 0.8]] * 3),
                    foot_uv=(x, 100.0))

    prev, target = [pose_at(100.0)], [pose_at(140.0)]
    # A 55 ms frame at tau=0.12 → alpha ≈ 0.37: moves toward, doesn't jump.
    out = _advance_smoothing(prev, target, 0.055, tau_s=0.12, snap_px=120.0)
    assert 100.0 < out[0].keypoints[0, 0] < 140.0
    assert 100.0 < out[0].foot_uv[0] < 140.0
    # Iterating converges to the target.
    cur = prev
    for _ in range(30):
        cur = _advance_smoothing(cur, target, 0.055, tau_s=0.12, snap_px=120.0)
    assert abs(cur[0].keypoints[0, 0] - 140.0) < 1.0
    # Far jump (> snap_px) snaps instead of rubber-banding.
    out = _advance_smoothing([pose_at(100.0)], [pose_at(600.0)], 0.055,
                             tau_s=0.12, snap_px=120.0)
    assert out[0].keypoints[0, 0] == 600.0
    # Empty target clears; empty prev snaps to target.
    assert _advance_smoothing([pose_at(1.0)], [], 0.055, tau_s=0.12, snap_px=120.0) == []
    assert _advance_smoothing([], [pose_at(5.0)], 0.055, tau_s=0.12,
                              snap_px=120.0)[0].keypoints[0, 0] == 5.0


def test_extrapolation_projects_skeleton_to_now() -> None:
    """A person moving +400 px/s whose newest result is 100 ms old must render
    ~+40 px AHEAD of that result — on the body, not behind it."""
    import numpy as np

    from monitor_web.pose_overlay import Pose, _extrapolate

    def pose_at(x):
        return Pose(box_xyxy=np.array([x, 0.0, x + 50, 100.0]), score=0.9,
                    keypoints=np.array([[x, 10.0, 0.8]] * 3), foot_uv=(x, 100.0))

    prev, latest = [pose_at(100.0)], [pose_at(140.0)]   # +40 px over the pair
    out = _extrapolate(latest, prev, dt_pair_s=0.1, age_s=0.1,
                       max_age_s=0.35, snap_px=120.0)
    assert abs(out[0].keypoints[0, 0] - 180.0) < 1e-6    # 140 + 40*(0.1/0.1)
    assert abs(out[0].foot_uv[0] - 180.0) < 1e-6
    assert abs(out[0].box_xyxy[0] - 180.0) < 1e-6


def test_extrapolation_age_is_clamped() -> None:
    import numpy as np

    from monitor_web.pose_overlay import Pose, _extrapolate

    def pose_at(x):
        return Pose(box_xyxy=np.array([x, 0.0, x + 50, 100.0]), score=0.9,
                    keypoints=np.array([[x, 10.0, 0.8]] * 3), foot_uv=(x, 100.0))

    # 2 s old result: extrapolate at most max_age_s (0.35 s) of motion —
    # velocity is 400 px/s → cap at +140 px past the newest, never +800.
    out = _extrapolate([pose_at(140.0)], [pose_at(100.0)], dt_pair_s=0.1,
                       age_s=2.0, max_age_s=0.35, snap_px=120.0)
    assert abs(out[0].keypoints[0, 0] - (140.0 + 400.0 * 0.35)) < 1e-6


def test_extrapolation_falls_back_without_trustworthy_velocity() -> None:
    import numpy as np

    from monitor_web.pose_overlay import Pose, _extrapolate

    def pose_at(x):
        return Pose(box_xyxy=np.array([x, 0.0, x + 50, 100.0]), score=0.9,
                    keypoints=np.array([[x, 10.0, 0.8]] * 3), foot_uv=(x, 100.0))

    latest = [pose_at(140.0)]
    # No previous result → pass through unchanged.
    assert _extrapolate(latest, [], 0.1, 0.1, max_age_s=0.35,
                        snap_px=120.0)[0].keypoints[0, 0] == 140.0
    # Degenerate pair interval (dt too small) → pass through.
    assert _extrapolate(latest, [pose_at(100.0)], 0.001, 0.1, max_age_s=0.35,
                        snap_px=120.0)[0].keypoints[0, 0] == 140.0
    # Match beyond snap_px (teleport / different person) → pass through.
    assert _extrapolate(latest, [pose_at(600.0)], 0.1, 0.1, max_age_s=0.35,
                        snap_px=120.0)[0].keypoints[0, 0] == 140.0
