"""``DiagnosticsPublisher`` — hermetic tests; no real orchestrator or thread needed.

Uses a stub orchestrator so ``build_message()`` can be verified synchronously
without spinning up the background thread.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import pytest

from backbone.comms.diagnostics_publisher import DiagnosticsPublisher
from backbone.comms.schemas import (
    SCHEMA_VERSION,
    DiagnosticsMessage,
    MessageType,
)
from backbone.shared.timestamps import LatencyMeter

# ---------------------------------------------------------------------------
# Stub helpers
# ---------------------------------------------------------------------------

@dataclass
class _StubCameraView:
    """Minimal camera view exposing only reprojection_rms_px."""
    reprojection_rms_px: float


class _StubRig:
    """Minimal CameraRig surface used by DiagnosticsPublisher."""

    def __init__(self, cameras: dict[str, float]) -> None:
        self._cameras = {k: _StubCameraView(v) for k, v in cameras.items()}

    @property
    def camera_ids(self) -> tuple[str, ...]:
        return tuple(self._cameras.keys())

    def items(self) -> dict[str, _StubCameraView]:
        return self._cameras


class _StubOrchestrator:
    """Minimal orchestrator surface used by DiagnosticsPublisher."""

    def __init__(
        self,
        *,
        mode: str = "single_cam_homography",
        source_status: dict[str, str] | None = None,
        frame_count: int = 0,
        rig: _StubRig | None = None,
        zone_count: int = 0,
        subscription_count: int = 0,
    ) -> None:
        self.mode = mode
        self.source_status = source_status or {"cam_a": "alive"}
        self.frame_count = frame_count
        self.rig = rig or _StubRig({"cam_a": 0.5})
        self.zone_count = zone_count
        self.subscription_count = subscription_count
        self.latency_meter = LatencyMeter("capture_to_publish", window=64)


class _RecordingPublisher:
    """Publisher stub that records published messages."""

    def __init__(self) -> None:
        self.diagnostics: list[DiagnosticsMessage] = []

    def publish_diagnostics(self, msg: object) -> None:
        assert isinstance(msg, DiagnosticsMessage)
        self.diagnostics.append(msg)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_build_message_returns_diagnostics_message() -> None:
    orch = _StubOrchestrator()
    pub = _RecordingPublisher()
    dp = DiagnosticsPublisher(orch, pub, node_id="zone_a")
    msg = dp.build_message()
    assert isinstance(msg, DiagnosticsMessage)
    assert msg.type == MessageType.DIAGNOSTICS
    assert msg.schema_version == SCHEMA_VERSION


def test_build_message_node_id_propagated() -> None:
    orch = _StubOrchestrator()
    pub = _RecordingPublisher()
    dp = DiagnosticsPublisher(orch, pub, node_id="warehouse_b")
    msg = dp.build_message()
    assert msg.node_id == "warehouse_b"


def test_build_message_fps_zero_on_first_call() -> None:
    """First call must return fps=0.0 because there is no prior tick."""
    orch = _StubOrchestrator(frame_count=999)
    pub = _RecordingPublisher()
    dp = DiagnosticsPublisher(orch, pub, node_id="z")
    msg = dp.build_message()
    assert msg.fps == pytest.approx(0.0)


def test_build_message_fps_positive_on_second_call() -> None:
    """Second call with an increased frame_count must yield fps > 0."""
    orch = _StubOrchestrator(frame_count=0)
    pub = _RecordingPublisher()
    dp = DiagnosticsPublisher(orch, pub, node_id="z", interval_sec=5.0)

    dp.build_message()                # first call — seeds the tick
    time.sleep(0.05)                  # let a tiny dt accumulate
    orch.frame_count = 10             # bump the counter
    msg = dp.build_message()          # second call — should compute fps

    assert msg.fps > 0.0


def test_build_message_sources_matches_orchestrator() -> None:
    orch = _StubOrchestrator(source_status={"cam_a": "alive", "cam_b": "crashed"})
    pub = _RecordingPublisher()
    dp = DiagnosticsPublisher(orch, pub, node_id="z")
    msg = dp.build_message()
    assert msg.sources == {"cam_a": "alive", "cam_b": "crashed"}


def test_build_message_calibration_rms_ok_within_gate() -> None:
    rig = _StubRig({"cam_a": 1.5, "cam_b": 1.8})  # both < 2.0 gate
    orch = _StubOrchestrator(rig=rig)
    pub = _RecordingPublisher()
    dp = DiagnosticsPublisher(orch, pub, node_id="z", rms_gate_px=2.0)
    msg = dp.build_message()
    assert msg.calibration.loaded is True
    assert msg.calibration.rms_ok is True


def test_build_message_calibration_rms_fails_above_gate() -> None:
    rig = _StubRig({"cam_a": 1.5, "cam_b": 3.5})  # cam_b exceeds gate
    orch = _StubOrchestrator(rig=rig)
    pub = _RecordingPublisher()
    dp = DiagnosticsPublisher(orch, pub, node_id="z", rms_gate_px=2.0)
    msg = dp.build_message()
    assert msg.calibration.rms_ok is False


def test_build_message_calibration_mode_single_cam() -> None:
    orch = _StubOrchestrator(mode="single_cam_homography")
    pub = _RecordingPublisher()
    dp = DiagnosticsPublisher(orch, pub, node_id="z")
    msg = dp.build_message()
    assert msg.calibration.mode == 1


def test_build_message_calibration_mode_dual_cam() -> None:
    rig = _StubRig({"cam_a": 0.5, "cam_b": 0.6})
    orch = _StubOrchestrator(
        mode="dual_cam_homography_triangulation",
        rig=rig,
    )
    pub = _RecordingPublisher()
    dp = DiagnosticsPublisher(orch, pub, node_id="z")
    msg = dp.build_message()
    assert msg.calibration.mode == 2


def test_build_message_zone_and_subscription_counts() -> None:
    orch = _StubOrchestrator(zone_count=5, subscription_count=2)
    pub = _RecordingPublisher()
    dp = DiagnosticsPublisher(orch, pub, node_id="z")
    msg = dp.build_message()
    assert msg.zones == 5
    assert msg.subscriptions == 2


def test_build_message_latency_stats_from_meter() -> None:
    orch = _StubOrchestrator()
    # Seed the latency meter with known samples.
    for ms in [10.0, 20.0, 30.0]:
        orch.latency_meter.record_ms(ms)
    pub = _RecordingPublisher()
    dp = DiagnosticsPublisher(orch, pub, node_id="z")
    msg = dp.build_message()
    assert msg.latency_ms.n == 3
    assert msg.latency_ms.p50 > 0.0


def test_build_message_mode_in_message() -> None:
    orch = _StubOrchestrator(mode="dual_cam_homography_triangulation")
    pub = _RecordingPublisher()
    dp = DiagnosticsPublisher(orch, pub, node_id="z")
    msg = dp.build_message()
    assert msg.mode == "dual_cam_homography_triangulation"


def test_start_stop_does_not_raise() -> None:
    """start()/stop() lifecycle must complete without error (no publish asserted)."""
    orch = _StubOrchestrator()
    pub = _RecordingPublisher()
    dp = DiagnosticsPublisher(orch, pub, node_id="z", interval_sec=60.0)
    dp.start()
    dp.stop()   # must not raise or hang


def test_stop_before_start_is_safe() -> None:
    """stop() called without start() must be a no-op."""
    orch = _StubOrchestrator()
    pub = _RecordingPublisher()
    dp = DiagnosticsPublisher(orch, pub, node_id="z")
    dp.stop()   # must not raise
