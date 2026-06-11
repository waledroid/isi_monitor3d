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
