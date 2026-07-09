"""``BusSubscriber`` — UDP listener decodes envelopes and updates state."""

from __future__ import annotations

import json
import socket
import time

from backbone.comms.schemas import Track2DMessage, Track3DMessage
from backbone.core.types import Track2D, Track3D

from monitor_web.bus_subscriber import BusSubscriber


def _bind_subscriber() -> tuple[BusSubscriber, int]:
    """Bind to an OS-assigned localhost port and return (subscriber, port)."""
    # Reserve a port by binding briefly, close, then let BusSubscriber re-bind.
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    sub = BusSubscriber("127.0.0.1", port)
    return sub, port


def _t2(track_id: int = 1) -> Track2D:
    return Track2D(
        track_id=track_id, cls="person", capture_ts=time.time(),
        xy_m=(1.0, 2.0), vxy_m=(0.0, 0.0),
        confidence=0.9, cameras_seeing=("cam_a", "cam_b"),
    )


def _send_track2d(port: int, track_id: int = 1) -> None:
    msg = Track2DMessage.from_track(_t2(track_id))
    payload = msg.model_dump_json().encode("utf-8")
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.sendto(payload, ("127.0.0.1", port))
    finally:
        s.close()


def _wait_for(predicate, timeout: float = 2.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_starts_and_receives_one_envelope() -> None:
    sub, port = _bind_subscriber()
    sub.start()
    try:
        _send_track2d(port, track_id=42)
        assert _wait_for(lambda: sub.snapshot().received >= 1)
        snap = sub.snapshot()
        assert snap.last_track2d_by_id[42].xy_m == (1.0, 2.0)
        assert sub.is_fresh(threshold_s=2.0)
    finally:
        sub.stop()


def test_malformed_json_dropped_without_crash() -> None:
    sub, port = _bind_subscriber()
    sub.start()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.sendto(b"not json", ("127.0.0.1", port))
        finally:
            s.close()
        assert _wait_for(lambda: sub.snapshot().dropped_malformed >= 1)
        assert sub.snapshot().received == 0
    finally:
        sub.stop()


def test_wrong_schema_version_dropped() -> None:
    sub, port = _bind_subscriber()
    sub.start()
    try:
        bad = {"schema_version": 0, "type": "track_2d"}
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.sendto(json.dumps(bad).encode(), ("127.0.0.1", port))
        finally:
            s.close()
        assert _wait_for(lambda: sub.snapshot().dropped_version >= 1)
        assert sub.snapshot().received == 0
    finally:
        sub.stop()


def test_track3d_stored_under_same_track_id() -> None:
    sub, port = _bind_subscriber()
    sub.start()
    try:
        t3 = Track3D(
            track_id=7, cls="person", capture_ts=time.time(),
            xyz_m=(1.0, 2.0, 0.0), vxyz_m=(0.0, 0.0, 0.0),
            contributing_cameras=("cam_a", "cam_b"),
            max_reprojection_error_px=0.3, keypoints_xyz=None,
        )
        payload = Track3DMessage.from_track(t3).model_dump_json().encode()
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.sendto(payload, ("127.0.0.1", port))
        finally:
            s.close()
        assert _wait_for(lambda: sub.snapshot().received >= 1)
        snap = sub.snapshot()
        assert snap.last_track3d_by_id[7].xyz_m == (1.0, 2.0, 0.0)
    finally:
        sub.stop()


def test_is_fresh_flips_to_false_after_threshold() -> None:
    sub, port = _bind_subscriber()
    sub.start()
    try:
        _send_track2d(port)
        assert _wait_for(lambda: sub.snapshot().received >= 1)
        assert sub.is_fresh(threshold_s=2.0)
        # No new packets for 0.2s; threshold 0.1s → stale.
        time.sleep(0.2)
        assert not sub.is_fresh(threshold_s=0.1)
    finally:
        sub.stop()


def test_stop_is_idempotent() -> None:
    sub, _ = _bind_subscriber()
    sub.start()
    sub.stop()
    sub.stop()   # must not raise


def test_zone_state_stored_by_zone_name() -> None:
    """zone_state envelopes populate snapshot().zone_state_by_zone — the local
    data source for the COMMUNICATION panel's zone cards."""
    from backbone.comms.schemas import ZoneObject, ZoneStateMessage

    sub, port = _bind_subscriber()
    sub.start()
    try:
        msg = ZoneStateMessage(
            ts=time.time(), zone="dock",
            objects=(
                ZoneObject(track_id=1, cls="palette", confidence=0.9,
                           xy_m=(1.0, 1.0), occupancy_state="empty"),
                ZoneObject(track_id=2, cls="carton", confidence=0.8, xy_m=(1.2, 1.1)),
            ),
            count=2,
        )
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.sendto(msg.model_dump_json().encode("utf-8"), ("127.0.0.1", port))
        finally:
            s.close()
        assert _wait_for(lambda: "dock" in sub.snapshot().zone_state_by_zone)
        stored = sub.snapshot().zone_state_by_zone["dock"]
        assert stored.count == 2
        assert stored.objects[0].occupancy_state == "empty"
        assert stored.objects[1].cls == "carton"
    finally:
        sub.stop()


def test_diagnostics_fps_stored_for_status_panel() -> None:
    """Diagnostics heartbeats populate per-camera + pipeline fps in the
    snapshot — the STATUS panel's camera-performance rows."""
    from backbone.comms.schemas import (
        CalibrationFactCheck,
        DiagnosticsMessage,
        LatencyStats,
    )

    sub, port = _bind_subscriber()
    sub.start()
    try:
        msg = DiagnosticsMessage(
            ts=time.time(), node_id="n1", mode="dual_cam_homography_triangulation",
            sources={"cam_a": "alive", "cam_b": "alive"},
            frame_count=100, fps=6.1,
            fps_by_camera={"cam_a": 12.3, "cam_b": 11.8},
            latency_ms=LatencyStats(), zones=2, subscriptions=1,
            calibration=CalibrationFactCheck(loaded=True, rms_ok=True, mode=2),
        )
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.sendto(msg.model_dump_json().encode("utf-8"), ("127.0.0.1", port))
        finally:
            s.close()
        assert _wait_for(lambda: sub.snapshot().fps_by_camera)
        snap = sub.snapshot()
        assert snap.fps_by_camera == {"cam_a": 12.3, "cam_b": 11.8}
        assert snap.pipeline_fps == 6.1
        assert snap.diagnostics_ts > 0.0
    finally:
        sub.stop()


def test_broadcast_offer_drops_oldest_when_full() -> None:
    """With no /ws/tracks client draining, the broadcast queue fills — the
    offer must evict the OLDEST and keep the newest, silently. (The old direct
    put_nowait raised QueueFull unhandled on the event loop, and uvloop logged
    the entire queue repr to the console per message.)"""
    import asyncio

    from monitor_web.bus_subscriber import _offer_broadcast

    async def run():
        q: asyncio.Queue = asyncio.Queue(maxsize=2)
        for i in range(5):
            _offer_broadcast(q, i)       # never raises
        assert q.qsize() == 2
        assert q.get_nowait() == 3       # oldest evicted…
        assert q.get_nowait() == 4       # …newest kept

    asyncio.run(run())


def test_clear_live_state_empties_caches_keeps_counters():
    """STOP calls this so the map/cards/panels blank immediately — live caches
    empty, received/latency counters intact."""
    from backbone.comms.schemas import ObservationsMessage

    sub = BusSubscriber(host="127.0.0.1", port=0)
    with sub._lock:
        sub._state.received = 42
        sub._state.last_track2d_by_id[1] = object()
        sub._state.zone_state_by_zone["Z"] = object()
        sub._state.observations_by_camera["cam_a"] = ObservationsMessage(
            ts=1.0, camera_id="cam_a", frame_wh=(10, 10), dets=())
        sub._state.fps_by_camera = {"cam_a": 18.0}
        sub._state.pipeline_fps = 25.0
    sub.clear_live_state()
    snap = sub.snapshot()
    assert snap.received == 42
    assert not snap.last_track2d_by_id and not snap.zone_state_by_zone
    assert not snap.observations_by_camera and not snap.fps_by_camera
    assert snap.pipeline_fps is None


def test_fragmented_observations_reassemble_into_state():
    """UdpSink slices >MTU payloads into fragment datagrams (WSL2 mirrored
    networking drops big loopback UDP) — the subscriber must reassemble them
    and store the observations as if they had arrived whole."""
    import json as _json

    from backbone.comms.schemas import ObservationDet, ObservationsMessage

    poly = tuple((float(i % 1920), float(i % 1080)) for i in range(400))
    obs = ObservationsMessage(
        ts=3.0, camera_id="cam_a", frame_wh=(1920, 1080),
        dets=(ObservationDet(cls="palette", confidence=0.9,
                             bbox_xyxy=(0.0, 0.0, 50.0, 50.0),
                             foot_uv=(25.0, 50.0), mask_poly=poly),))
    text = obs.model_dump_json()
    assert len(text) > 1300
    chunk = 1100
    chunks = [text[i:i + chunk] for i in range(0, len(text), chunk)]

    sub = BusSubscriber(host="127.0.0.1", port=0)
    for i, c in enumerate(chunks):
        sub._handle_payload(_json.dumps(
            {"schema_version": 6, "type": "fragment", "fid": "f1",
             "i": i, "n": len(chunks), "data": c}).encode())
    snap = sub.snapshot()
    stored = snap.observations_by_camera["cam_a"]
    assert stored == obs
    assert stored.dets[0].mask_poly == poly
    # Only the COMPLETED message counts as received (fragments are transport).
    assert snap.received == 1
