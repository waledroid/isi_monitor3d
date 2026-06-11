"""``UdpSink`` — verify packets land on a loopback socket as expected JSON."""

from __future__ import annotations

import json
import socket

import pytest

from backbone.core.interfaces import metadata_sink_registry
from backbone.core.types import Track2D, Track3D
from backbone.metadata.schemas import SCHEMA_VERSION, MessageType
from backbone.metadata.udp_sink import UdpSink


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
    import backbone.metadata  # noqa: F401

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


def test_close_is_idempotent() -> None:
    sink = UdpSink(host="127.0.0.1", port=12345)
    sink.close()
    sink.close()   # must not raise


