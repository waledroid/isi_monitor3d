"""``Publisher`` — fan-out semantics + error isolation between sinks."""

from __future__ import annotations

from backbone.core.interfaces import MetadataSink
from backbone.core.types import Track2D, Track3D
from backbone.metadata.publisher import Publisher


class _RecordingSink(MetadataSink):
    def __init__(self, *, raise_on_2d: bool = False, raise_on_3d: bool = False) -> None:
        self.track_2d: list[Track2D] = []
        self.track_3d: list[Track3D] = []
        self.closed = False
        self._raise_on_2d = raise_on_2d
        self._raise_on_3d = raise_on_3d

    def publish_track_2d(self, track: Track2D) -> None:
        if self._raise_on_2d:
            raise RuntimeError("simulated sink failure")
        self.track_2d.append(track)

    def publish_track_3d(self, track: Track3D) -> None:
        if self._raise_on_3d:
            raise RuntimeError("simulated sink failure")
        self.track_3d.append(track)

    def close(self) -> None:
        self.closed = True


def _t2() -> Track2D:
    return Track2D(
        track_id=1, cls="person", capture_ts=0.0,
        xy_m=(1.0, 1.0), vxy_m=(0.0, 0.0),
        confidence=0.9, cameras_seeing=("cam_a",),
    )


def _t3() -> Track3D:
    return Track3D(
        track_id=1, cls="person", capture_ts=0.0,
        xyz_m=(1.0, 1.0, 0.0), vxyz_m=(0.0, 0.0, 0.0),
        contributing_cameras=("cam_a", "cam_b"),
        max_reprojection_error_px=0.5,
        keypoints_xyz=None,
    )


def test_fan_out_to_multiple_sinks() -> None:
    a, b = _RecordingSink(), _RecordingSink()
    pub = Publisher([a, b])
    pub.publish_track_2d(_t2())
    assert len(a.track_2d) == 1
    assert len(b.track_2d) == 1


def test_sink_failure_does_not_starve_others() -> None:
    bad = _RecordingSink(raise_on_2d=True)
    good = _RecordingSink()
    pub = Publisher([bad, good])
    pub.publish_track_2d(_t2())
    assert len(bad.track_2d) == 0
    assert len(good.track_2d) == 1


def test_publish_after_close_is_silent_noop() -> None:
    sink = _RecordingSink()
    pub = Publisher([sink])
    pub.close()
    pub.publish_track_2d(_t2())
    pub.publish_track_3d(_t3())
    assert sink.track_2d == []
    assert sink.track_3d == []
    assert sink.closed


def test_close_propagates_to_all_sinks() -> None:
    a, b = _RecordingSink(), _RecordingSink()
    pub = Publisher([a, b])
    pub.close()
    assert a.closed
    assert b.closed


def test_close_swallows_sink_errors() -> None:
    class _BadCloser(_RecordingSink):
        def close(self) -> None:
            raise RuntimeError("close failed")
    good = _RecordingSink()
    pub = Publisher([_BadCloser(), good])
    pub.close()    # must not raise
    assert good.closed


def test_track_3d_fan_out() -> None:
    a, b = _RecordingSink(), _RecordingSink()
    pub = Publisher([a, b])
    pub.publish_track_3d(_t3())
    assert len(a.track_3d) == 1
    assert len(b.track_3d) == 1
