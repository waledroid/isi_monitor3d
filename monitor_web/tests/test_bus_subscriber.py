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
