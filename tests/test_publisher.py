"""``Publisher`` — fan-out semantics + error isolation between sinks."""

from __future__ import annotations

from backbone.comms.publisher import Publisher
from backbone.comms.schemas import (
    CalibrationFactCheck,
    ConfigMessage,
    DiagnosticsMessage,
    LatencyStats,
)
from backbone.core.interfaces import MetadataSink
from backbone.core.types import Track2D, Track3D


class _RecordingSink(MetadataSink):
    def __init__(
        self,
        *,
        raise_on_2d: bool = False,
        raise_on_3d: bool = False,
        raise_on_diag: bool = False,
        raise_on_config: bool = False,
        raise_on_zone_state: bool = False,
    ) -> None:
        self.track_2d: list[Track2D] = []
        self.track_3d: list[Track3D] = []
        self.diagnostics: list[object] = []
        self.configs: list[object] = []
        self.zone_states: list[object] = []
        self.closed = False
        self._raise_on_2d = raise_on_2d
        self._raise_on_3d = raise_on_3d
        self._raise_on_diag = raise_on_diag
        self._raise_on_config = raise_on_config
        self._raise_on_zone_state = raise_on_zone_state

    def publish_track_2d(self, track: Track2D) -> None:
        if self._raise_on_2d:
            raise RuntimeError("simulated sink failure")
        self.track_2d.append(track)

    def publish_track_3d(self, track: Track3D) -> None:
        if self._raise_on_3d:
            raise RuntimeError("simulated sink failure")
        self.track_3d.append(track)

    def publish_diagnostics(self, msg: object) -> None:
        if self._raise_on_diag:
            raise RuntimeError("simulated diag failure")
        self.diagnostics.append(msg)

    def publish_config(self, msg: object) -> None:
        if self._raise_on_config:
            raise RuntimeError("simulated config failure")
        self.configs.append(msg)

    def publish_zone_state(self, msg: object) -> None:
        if self._raise_on_zone_state:
            raise RuntimeError("simulated zone-state failure")
        self.zone_states.append(msg)

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


# ---------------------------------------------------------------------------
# publish_diagnostics fan-out
# ---------------------------------------------------------------------------

def _make_diag() -> DiagnosticsMessage:
    return DiagnosticsMessage(
        ts=1.0,
        node_id="z",
        mode="single_cam_homography",
        sources={"cam_a": "alive"},
        frame_count=0,
        fps=0.0,
        latency_ms=LatencyStats(),
        zones=0,
        subscriptions=0,
        calibration=CalibrationFactCheck(loaded=True, rms_ok=True, mode=1),
    )


def _make_cfg() -> ConfigMessage:
    return ConfigMessage(
        ts=1.0,
        node_id="z",
        area="Zone A",
        mode="single_cam_homography",
        cameras=["cam_a"],
        zones=[],
        calibration=CalibrationFactCheck(loaded=True, rms_ok=True, mode=1),
    )


def test_publish_diagnostics_fan_out() -> None:
    a, b = _RecordingSink(), _RecordingSink()
    pub = Publisher([a, b])
    pub.publish_diagnostics(_make_diag())
    assert len(a.diagnostics) == 1
    assert len(b.diagnostics) == 1


def test_publish_diagnostics_raising_sink_swallowed() -> None:
    bad = _RecordingSink(raise_on_diag=True)
    good = _RecordingSink()
    pub = Publisher([bad, good])
    pub.publish_diagnostics(_make_diag())   # must not raise
    assert len(good.diagnostics) == 1


def test_publish_diagnostics_noop_after_close() -> None:
    sink = _RecordingSink()
    pub = Publisher([sink])
    pub.close()
    pub.publish_diagnostics(_make_diag())
    assert sink.diagnostics == []


# ---------------------------------------------------------------------------
# publish_config fan-out
# ---------------------------------------------------------------------------

def test_publish_config_fan_out() -> None:
    a, b = _RecordingSink(), _RecordingSink()
    pub = Publisher([a, b])
    pub.publish_config(_make_cfg())
    assert len(a.configs) == 1
    assert len(b.configs) == 1


def test_publish_config_raising_sink_swallowed() -> None:
    bad = _RecordingSink(raise_on_config=True)
    good = _RecordingSink()
    pub = Publisher([bad, good])
    pub.publish_config(_make_cfg())   # must not raise
    assert len(good.configs) == 1


def test_publish_config_noop_after_close() -> None:
    sink = _RecordingSink()
    pub = Publisher([sink])
    pub.close()
    pub.publish_config(_make_cfg())
    assert sink.configs == []


# ---------------------------------------------------------------------------
# publish_zone_state fan-out
# ---------------------------------------------------------------------------

def _make_zone_state():
    from backbone.comms.schemas import ZoneObject, ZoneStateMessage
    return ZoneStateMessage(
        ts=1.0,
        zone="B3D",
        objects=(ZoneObject(track_id=1, cls="palette", confidence=0.9, xy_m=(1.0, 1.0)),),
        count=1,
    )


def test_publish_zone_state_fan_out() -> None:
    a, b = _RecordingSink(), _RecordingSink()
    pub = Publisher([a, b])
    pub.publish_zone_state(_make_zone_state())
    assert len(a.zone_states) == 1
    assert len(b.zone_states) == 1


def test_publish_zone_state_raising_sink_swallowed() -> None:
    bad = _RecordingSink(raise_on_zone_state=True)
    good = _RecordingSink()
    pub = Publisher([bad, good])
    pub.publish_zone_state(_make_zone_state())   # must not raise
    assert len(good.zone_states) == 1


def test_publish_zone_state_noop_after_close() -> None:
    sink = _RecordingSink()
    pub = Publisher([sink])
    pub.close()
    pub.publish_zone_state(_make_zone_state())
    assert sink.zone_states == []


# ---------------------------------------------------------------------------
# advertise_zones fan-out
# ---------------------------------------------------------------------------

def test_advertise_zones_fan_out() -> None:
    class _ZoneRecordingSink(_RecordingSink):
        def __init__(self) -> None:
            super().__init__()
            self.advertised: list[list[tuple[str, str]]] = []

        def advertise_zones(self, zones: list[tuple[str, str]]) -> None:
            self.advertised.append(zones)

    a, b = _ZoneRecordingSink(), _ZoneRecordingSink()
    pub = Publisher([a, b])
    pub.advertise_zones([("Loading Bay", "z1")])
    assert a.advertised == [[("Loading Bay", "z1")]]
    assert b.advertised == [[("Loading Bay", "z1")]]


def test_advertise_zones_default_noop_and_raising_sink_swallowed() -> None:
    class _BadAdvertiser(_RecordingSink):
        def advertise_zones(self, zones: list[tuple[str, str]]) -> None:
            raise RuntimeError("simulated advertise failure")

    good = _RecordingSink()   # exercises the ABC's default no-op
    pub = Publisher([_BadAdvertiser(), good])
    pub.advertise_zones([("Loading Bay", "z1")])   # must not raise
