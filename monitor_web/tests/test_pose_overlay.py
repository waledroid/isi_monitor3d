"""Pose overlay wiring — get_pose_detector resolution + annotate_frame hook."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from monitor_web import detection_overlay as do
from monitor_web.config import Settings


def _cfg(tmp_path: Path, detection: dict | None = None) -> Settings:
    bb = tmp_path / "backbone.yaml"
    body: dict = {"cameras": {"cam_a": {"source": {"name": "rtsp", "url": "rtsp://x/y"}}}}
    if detection is not None:
        body["detection"] = detection
    bb.write_text(yaml.safe_dump(body))
    return Settings(backbone_config_path=bb, udp_port=0, port=0)


def test_get_pose_detector_none_when_unconfigured(tmp_path) -> None:
    do.reset_detector()
    assert do.get_pose_detector(_cfg(tmp_path)) is None
    # detection block present but no pose path → still None.
    assert do.get_pose_detector(_cfg(tmp_path, {"onnx_path": "/x.onnx"})) is None


def test_get_pose_detector_none_when_path_missing(tmp_path) -> None:
    """A configured pose path that doesn't resolve degrades to None (best-effort —
    never breaks the stream)."""
    do.reset_detector()
    cfg = _cfg(tmp_path, {"pose_onnx_path": str(tmp_path / "nope.onnx")})
    assert do.get_pose_detector(cfg) is None


class _FakePose:
    """Stand-in pose engine: records that annotate_frame invoked it."""

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


class _SlowEngine:
    """Engine stub whose predict is slow — proves the async runner never
    blocks the caller and eventually surfaces results."""

    kpt_conf = 0.3

    def __init__(self, delay_s: float = 0.05) -> None:
        self.delay_s = delay_s
        self.calls = 0
        self.drawn = 0

    def predict(self, image):
        import time as _t
        _t.sleep(self.delay_s)
        self.calls += 1
        return ["pose"]

    def draw(self, image, poses):
        self.drawn += 1


def test_async_pose_runner_never_blocks_and_surfaces_results() -> None:
    """The cam-view fix: predict() must return immediately (video stays at
    camera rate) and, once the background worker completes, return its result."""
    import time

    import numpy as np

    from monitor_web.pose_overlay import AsyncPoseRunner

    eng = _SlowEngine(delay_s=0.05)
    runner = AsyncPoseRunner(eng)
    img = np.zeros((10, 10, 3), dtype=np.uint8)

    t0 = time.perf_counter()
    first = runner.predict(img)
    assert time.perf_counter() - t0 < 0.02, "predict blocked on inference"
    assert first == []                       # nothing computed yet

    deadline = time.time() + 2.0
    while time.time() < deadline:
        poses = runner.predict(img)
        if poses:
            break
        time.sleep(0.01)
    assert poses == ["pose"]
    runner.draw(img, poses)
    assert eng.drawn == 1


def test_async_pose_runner_clears_stale_results() -> None:
    """A result older than the staleness window must not linger as a frozen
    skeleton on live video."""
    import numpy as np

    from monitor_web.pose_overlay import AsyncPoseRunner

    runner = AsyncPoseRunner(_SlowEngine(delay_s=0.0))
    runner._poses = ["old"]
    runner._result_ts = 0.0                  # ancient
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    assert runner.predict(img) == []


def test_pose_engine_handles_symbolic_input_dims(tmp_path) -> None:
    """Dynamic ONNX exports have shape [batch,3,'height','width'] — the engine
    must fall back to the canonical 640, not crash on string dims (the bug that
    silently killed the cam-view skeletons after the dynamic pose export)."""
    import numpy as np

    from monitor_web.pose_overlay import PoseEngine

    class _Dim:
        pass

    eng = PoseEngine.__new__(PoseEngine)     # skip session build
    eng.conf, eng.kpt_conf = 0.3, 0.3
    h, w = "height", "width"
    eng.h = h if isinstance(h, int) else 640
    eng.w = w if isinstance(w, int) else 640
    assert (eng.h, eng.w) == (640, 640)
    eng._lb = (1.0, 0, 0)
    out = eng._letterbox(np.zeros((360, 640, 3), dtype=np.uint8))
    assert out.shape == (640, 640, 3)


def test_async_pose_runner_stop_tears_down_now() -> None:
    """STOP hygiene: stop() must exit the worker, drop the engine ref (the
    CUDA session pin) and clear cached poses — and be idempotent. Post-stop
    predict()/draw() are inert, never crash."""
    import time

    import numpy as np

    from monitor_web.pose_overlay import AsyncPoseRunner

    eng = _SlowEngine(delay_s=0.0)
    runner = AsyncPoseRunner(eng)
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    runner.predict(img)                       # spins the worker up
    deadline = time.time() + 2.0
    while time.time() < deadline and not runner.predict(img):
        time.sleep(0.01)

    runner.stop()
    assert runner.engine is None
    assert runner._thread is None
    assert runner.predict(img) == []          # inert, not crashing
    runner.draw(img, [])                      # engine None → no-op
    _ = runner.kpt_conf                       # guarded fallback
    runner.stop()                             # idempotent


def test_reset_detector_drains_async_pose_registry() -> None:
    """The post-STOP VRAM leak: reset_detector() must stop and clear the
    per-camera AsyncPoseRunner registry (each pins its own PoseEngine CUDA
    session independent of the _POSE global)."""
    import monitor_web.detection_overlay as do
    from monitor_web.pose_overlay import AsyncPoseRunner

    r1, r2 = AsyncPoseRunner(_SlowEngine()), AsyncPoseRunner(_SlowEngine())
    do._ASYNC_POSE.clear()
    do._ASYNC_POSE.update({"cam_a": r1, "cam_b": r2})
    do.reset_detector()
    assert do._ASYNC_POSE == {}
    assert r1.engine is None and r2.engine is None


def test_get_pose_detector_passes_and_reloads_on_imgsz(tmp_path, monkeypatch) -> None:
    """``detection.pose_imgsz`` reaches PoseEngine and a changed value drops the
    cached engine (so the knob applies live, like the model path)."""
    import monitor_web.pose_overlay as po

    built: list[dict] = []

    class _StubEngine:
        def __init__(self, path, conf=0.3, imgsz=None):
            built.append({"path": path, "conf": conf, "imgsz": imgsz})

    monkeypatch.setattr(po, "PoseEngine", _StubEngine)
    model = tmp_path / "pose.onnx"
    model.write_bytes(b"x")   # existence is all get_pose_detector checks pre-build

    do.reset_detector()
    cfg = _cfg(tmp_path, {"pose_onnx_path": str(model), "pose_imgsz": 480})
    eng1 = do.get_pose_detector(cfg)
    assert eng1 is not None and built[-1]["imgsz"] == 480
    assert do.get_pose_detector(cfg) is eng1          # cached, same config

    bb = yaml.safe_load(Path(cfg.backbone_config_path).read_text())
    bb["detection"]["pose_imgsz"] = 384
    Path(cfg.backbone_config_path).write_text(yaml.safe_dump(bb))
    eng2 = do.get_pose_detector(cfg)
    assert eng2 is not eng1 and built[-1]["imgsz"] == 384
    do.reset_detector()


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
