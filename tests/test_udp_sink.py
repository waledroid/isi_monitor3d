"""``UdpSink`` — verify packets land on a loopback socket as expected JSON."""

from __future__ import annotations

import json
import socket

import pytest

from backbone.comms.schemas import (
    SCHEMA_VERSION,
    CalibrationFactCheck,
    ConfigMessage,
    DiagnosticsMessage,
    FragmentBuffer,
    FragmentMessage,
    LatencyStats,
    MessageType,
    ObservationDet,
    ObservationsMessage,
    parse_envelope,
)
from backbone.comms.udp_sink import UdpSink
from backbone.core.interfaces import metadata_sink_registry
from backbone.core.types import Track2D, Track3D
from backbone.shared.zone_transitions import PassingEvent


def _bind_receiver() -> tuple[socket.socket, int]:
    """Bind a UDP socket on 127.0.0.1 to an OS-assigned port."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    sock.settimeout(2.0)
    _, port = sock.getsockname()
    return sock, port


def _t2() -> Track2D:
    return Track2D(
        track_id=42, cls="person", capture_ts=12345.678,
        xy_m=(1.0, 2.0), vxy_m=(0.0, 0.0),
        confidence=0.8, cameras_seeing=("cam_a", "cam_b"),
    )


def _t3() -> Track3D:
    return Track3D(
        track_id=42, cls="person", capture_ts=12345.678,
        xyz_m=(1.0, 2.0, 0.0), vxyz_m=(0.0, 0.0, 0.0),
        contributing_cameras=("cam_a", "cam_b"),
        max_reprojection_error_px=2.5,
        keypoints_xyz=None,
    )


def test_plugin_registered_under_udp() -> None:
    import backbone.comms  # noqa: F401

    assert "udp" in metadata_sink_registry


def test_construction_via_registry() -> None:
    sock, port = _bind_receiver()
    try:
        sink = metadata_sink_registry.create("udp", host="127.0.0.1", port=port)
        assert isinstance(sink, UdpSink)
        sink.close()
    finally:
        sock.close()


def test_publish_track_2d_arrives_as_json() -> None:
    sock, port = _bind_receiver()
    try:
        sink = UdpSink(host="127.0.0.1", port=port)
        sink.publish_track_2d(_t2())
        payload, _ = sock.recvfrom(8192)
        msg = json.loads(payload.decode("utf-8"))
        assert msg["schema_version"] == SCHEMA_VERSION
        assert msg["type"] == MessageType.TRACK_2D.value
        assert msg["track_id"] == 42
        assert msg["xy_m"] == [1.0, 2.0]
        assert msg["cameras_seeing"] == ["cam_a", "cam_b"]
        sink.close()
    finally:
        sock.close()


def test_publish_track_3d_arrives_as_json() -> None:
    sock, port = _bind_receiver()
    try:
        sink = UdpSink(host="127.0.0.1", port=port)
        sink.publish_track_3d(_t3())
        payload, _ = sock.recvfrom(8192)
        msg = json.loads(payload.decode("utf-8"))
        assert msg["type"] == MessageType.TRACK_3D.value
        assert msg["xyz_m"] == [1.0, 2.0, 0.0]
        assert msg["max_reprojection_error_px"] == 2.5
        sink.close()
    finally:
        sock.close()


def test_port_validation() -> None:
    with pytest.raises(ValueError, match="port"):
        UdpSink(host="127.0.0.1", port=0)
    with pytest.raises(ValueError, match="port"):
        UdpSink(host="127.0.0.1", port=65536)


def test_publish_event_arrives_as_json_with_passing_type() -> None:
    """publish_event sends a PassingEventMessage datagram with type=='passing'."""
    sock, port = _bind_receiver()
    try:
        sink = UdpSink(host="127.0.0.1", port=port)
        ev = PassingEvent(track_id=42, cls="palette", zone="B3D", direction="enter", ts=1.0)
        sink.publish_event(ev)
        payload, _ = sock.recvfrom(8192)
        msg = json.loads(payload.decode("utf-8"))
        assert msg["type"] == MessageType.PASSING.value
        assert msg["schema_version"] == SCHEMA_VERSION
        assert msg["track_id"] == 42
        assert msg["zone"] == "B3D"
        assert msg["direction"] == "enter"
        assert msg["cls"] == "palette"
        sink.close()
    finally:
        sock.close()


def test_publish_image_ref_arrives_as_json() -> None:
    """publish_image_ref sends an ImageRefMessage datagram with type=='image_ref'."""
    sock, port = _bind_receiver()
    try:
        sink = UdpSink(host="127.0.0.1", port=port)
        sink.publish_image_ref(
            track_id=42,
            cls="palette",
            zone="B3D",
            ts=5.0,
            url="file:///var/lib/isi_monitor3d/snapshots/5000_B3D_42.jpg",
        )
        payload, _ = sock.recvfrom(8192)
        msg = json.loads(payload.decode("utf-8"))
        assert msg["type"] == MessageType.IMAGE_REF.value
        assert msg["schema_version"] == SCHEMA_VERSION
        assert msg["track_id"] == 42
        assert msg["zone"] == "B3D"
        assert msg["url"] == "file:///var/lib/isi_monitor3d/snapshots/5000_B3D_42.jpg"
        # No raw image bytes
        assert "image" not in msg
        assert "image_bytes" not in msg
        sink.close()
    finally:
        sock.close()


def test_close_is_idempotent() -> None:
    sink = UdpSink(host="127.0.0.1", port=12345)
    sink.close()
    sink.close()   # must not raise


# ---------------------------------------------------------------------------
# publish_diagnostics
# ---------------------------------------------------------------------------

def _make_diag_msg() -> DiagnosticsMessage:
    return DiagnosticsMessage(
        ts=1_700_000_000.0,
        node_id="zone_a",
        mode="single_cam_homography",
        sources={"cam_a": "alive"},
        frame_count=10,
        fps=5.0,
        latency_ms=LatencyStats(p50=8.0, p95=15.0, p99=20.0, n=50),
        zones=2,
        subscriptions=0,
        calibration=CalibrationFactCheck(loaded=True, rms_ok=True, mode=1),
    )


def _make_config_msg() -> ConfigMessage:
    return ConfigMessage(
        ts=1_700_000_000.0,
        node_id="zone_a",
        area="Zone A",
        mode="single_cam_homography",
        cameras=["cam_a"],
        zones=[],
        calibration=CalibrationFactCheck(loaded=True, rms_ok=True, mode=1),
    )


def test_publish_diagnostics_arrives_with_correct_type() -> None:
    sock, port = _bind_receiver()
    try:
        sink = UdpSink(host="127.0.0.1", port=port)
        sink.publish_diagnostics(_make_diag_msg())
        payload, _ = sock.recvfrom(16384)
        msg = json.loads(payload.decode("utf-8"))
        assert msg["type"] == MessageType.DIAGNOSTICS.value
        assert msg["schema_version"] == SCHEMA_VERSION
        assert msg["node_id"] == "zone_a"
        assert msg["fps"] == pytest.approx(5.0)
        assert msg["latency_ms"]["p95"] == pytest.approx(15.0)
        sink.close()
    finally:
        sock.close()


def test_publish_config_arrives_with_correct_type() -> None:
    sock, port = _bind_receiver()
    try:
        sink = UdpSink(host="127.0.0.1", port=port)
        sink.publish_config(_make_config_msg())
        payload, _ = sock.recvfrom(16384)
        msg = json.loads(payload.decode("utf-8"))
        assert msg["type"] == MessageType.CONFIG.value
        assert msg["schema_version"] == SCHEMA_VERSION
        assert msg["node_id"] == "zone_a"
        assert msg["area"] == "Zone A"
        assert msg["cameras"] == ["cam_a"]
        sink.close()
    finally:
        sock.close()




# ---- application-layer fragmentation (WSL2 mirrored networking drops
# loopback UDP datagrams over ~1.5 KB — large messages must be sliced) ----


def _recv_all(sock: socket.socket, n: int) -> list[dict]:
    out = []
    for _ in range(n):
        payload, _ = sock.recvfrom(65535)
        out.append(json.loads(payload))
    return out


def test_small_payload_is_not_fragmented() -> None:
    sock, port = _bind_receiver()
    sink = UdpSink(host="127.0.0.1", port=port)
    obs = ObservationsMessage(ts=1.0, camera_id="cam_a", frame_wh=(1920, 1080), dets=())
    sink.publish_observations(obs)
    (msg,) = _recv_all(sock, 1)
    assert msg["type"] == "observations"
    sock.close()
    sink.close()


def test_large_payload_fragments_and_reassembles() -> None:
    sock, port = _bind_receiver()
    sink = UdpSink(host="127.0.0.1", port=port)
    # A mask polygon big enough to exceed the fragmentation threshold.
    poly = tuple((float(i % 1920), float(i % 1080)) for i in range(400))
    dets = tuple(
        ObservationDet(cls="palette", confidence=0.9,
                       bbox_xyxy=(0.0, 0.0, 100.0, 100.0), foot_uv=(50.0, 100.0),
                       mask_poly=poly)
        for _ in range(3)
    )
    obs = ObservationsMessage(ts=2.0, camera_id="cam_b", frame_wh=(1920, 1080), dets=dets)
    original = obs.model_dump_json()
    assert len(original) > 1300, "test payload must actually exceed the threshold"

    sink.publish_observations(obs)
    (first,) = _recv_all(sock, 1)
    assert first["type"] == "fragment"
    frags = [first, *_recv_all(sock, first["n"] - 1)]

    # Every fragment fits a 1500-byte MTU with headroom.
    assert all(len(json.dumps(f, separators=(",", ":"))) < 1450 for f in frags)
    # Reassembly through the consumer-side helper restores the exact message.
    buf = FragmentBuffer()
    text = None
    for f in frags:
        text = buf.add(FragmentMessage.model_validate(f), now=0.0) or text
    assert text == original
    restored = parse_envelope(json.loads(text))
    assert isinstance(restored, ObservationsMessage)
    assert restored == obs
    sock.close()
    sink.close()


def test_fragment_buffer_prunes_stale_and_rejects_inconsistent() -> None:
    buf = FragmentBuffer(max_age_s=1.0)
    f0 = FragmentMessage(fid="aaa", i=0, n=2, data="{}")
    assert buf.add(f0, now=0.0) is None
    # Stale: the second part arrives after max_age_s — group was pruned,
    # so it re-opens and still doesn't complete.
    f1 = FragmentMessage(fid="aaa", i=1, n=2, data="x")
    assert buf.add(f1, now=5.0) is None
    # Inconsistent n drops the group.
    assert buf.add(FragmentMessage(fid="bbb", i=0, n=2, data="a"), now=5.0) is None
    assert buf.add(FragmentMessage(fid="bbb", i=1, n=3, data="b"), now=5.0) is None
    assert buf.add(FragmentMessage(fid="bbb", i=1, n=2, data="b"), now=5.0) is None


def test_publish_etagere_state_arrives_as_json() -> None:
    from backbone.comms.schemas import EtagereCellState, EtagereStateMessage
    sock, port = _bind_receiver()
    try:
        sink = UdpSink(host="127.0.0.1", port=port)
        sink.publish_etagere_state(EtagereStateMessage(
            ts=1.0, camera_id="cam_a", zone_id="et_1", rows=1, cols=1,
            cells=(EtagereCellState(r=1, c=1, state="empty"),), stabilized=True))
        payload, _ = sock.recvfrom(8192)
        msg = json.loads(payload.decode("utf-8"))
        assert msg["type"] == "etagere_state" and msg["zone_id"] == "et_1"
        assert msg["cells"][0]["state"] == "empty" and msg["stabilized"] is True
        sink.close()
    finally:
        sock.close()
