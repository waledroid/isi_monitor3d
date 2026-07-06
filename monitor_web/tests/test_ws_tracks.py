"""`/ws/tracks` serialization — only track envelopes go to the floor map."""

from __future__ import annotations

import time

from backbone.comms.schemas import (
    CalibrationFactCheck,
    DiagnosticsMessage,
    LatencyStats,
    Track2DMessage,
    ZoneStateMessage,
)
from backbone.core.types import Track2D

from monitor_web.api.routes_ws import _serialize


def _track2d() -> Track2DMessage:
    return Track2DMessage.from_track(Track2D(
        track_id=1, cls="person", capture_ts=time.time(),
        xy_m=(1.0, 2.0), vxy_m=(0.0, 0.0),
        confidence=0.9, cameras_seeing=("cam_a",),
    ))


def test_serialize_tracks_to_dict() -> None:
    out = _serialize(_track2d())
    assert isinstance(out, dict) and out["type"] == "track_2d"


def test_serialize_skips_non_track_envelopes() -> None:
    """The broadcast queue fans out EVERY bus envelope; non-track types must
    be skipped (None), not passed through raw — a raw DiagnosticsMessage made
    send_json raise TypeError and killed the socket (observed live)."""
    diag = DiagnosticsMessage(
        ts=time.time(), node_id="n", mode="single_cam_homography",
        sources={"cam_a": "alive"}, frame_count=1, fps=1.0,
        latency_ms=LatencyStats(), zones=0, subscriptions=0,
        calibration=CalibrationFactCheck(loaded=True, rms_ok=True, mode=1),
    )
    zone = ZoneStateMessage(ts=time.time(), zone="dock", objects=(), count=0)
    assert _serialize(diag) is None
    assert _serialize(zone) is None
