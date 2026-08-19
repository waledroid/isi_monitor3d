"""Shared test fixtures for the ISI Gateway test suite.

Design: the app is created with an in-memory Settings (no env vars, no broker).
Data is injected directly via ``subscriber.update_from_message()`` — no MQTT
broker needed, no sockets opened.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest
from backbone.comms.schemas import (
    CalibrationFactCheck,
    ConfigMessage,
    DiagnosticsMessage,
    LatencyStats,
    PassingEventMessage,
    Track2DMessage,
    Track3DMessage,
    ZoneSpec,
)
from fastapi.testclient import TestClient

from isicomms.app import create_app
from isicomms.config import Settings


@pytest.fixture(autouse=True)
def _no_real_broker(monkeypatch):
    """Replace the paho client so the suite never opens a socket or spawns a
    network thread — tests inject data via ``update_from_message`` directly, so
    a real loop would only add a multi-second teardown join on the dead broker."""
    monkeypatch.setattr(
        "isicomms.mqtt_subscriber.mqtt.Client",
        lambda *a, **k: MagicMock(),   # ignore the CallbackAPIVersion arg; unspec'd mock
    )
    yield


def _settings(**kwargs) -> Settings:
    """Build a Settings with safe defaults for tests (no real broker port)."""
    defaults = {
        "mqtt_host": "127.0.0.1",
        "mqtt_port": 1884,          # no real broker; start() retries silently
        "node_stale_after_s": 5.0,
        "passings_buffer": 10,
    }
    defaults.update(kwargs)
    return Settings(**defaults)


@pytest.fixture
def client():
    """TestClient backed by a gateway app with no broker and no auth."""
    app = create_app(_settings())
    with TestClient(app) as c:
        yield c


@pytest.fixture
def authed_client():
    """TestClient for an app that requires a bearer token."""
    app = create_app(_settings(api_token="secret"))
    with TestClient(app) as c:
        yield c


# ---- message builders (re-used across test modules) -------------------------

def make_track2d(
    track_id: int = 1,
    cls: str = "palette",
    xy_m: tuple = (1.0, 2.0),
    ts: float | None = None,
) -> Track2DMessage:
    return Track2DMessage(
        ts=ts if ts is not None else time.time(),
        track_id=track_id,
        cls=cls,
        xy_m=xy_m,
        vxy_m=(0.0, 0.0),
        confidence=0.9,
        cameras_seeing=("cam_a",),
    )


def make_track3d(
    track_id: int = 1,
    cls: str = "palette",
    xyz_m: tuple = (1.0, 2.0, 0.0),
    ts: float | None = None,
) -> Track3DMessage:
    return Track3DMessage(
        ts=ts if ts is not None else time.time(),
        track_id=track_id,
        cls=cls,
        xyz_m=xyz_m,
        vxyz_m=(0.0, 0.0, 0.0),
        contributing_cameras=("cam_a", "cam_b"),
        max_reprojection_error_px=1.2,
    )


def make_passing(
    track_id: int = 1,
    cls: str = "palette",
    zone: str = "rack_a",
    direction: str = "enter",
    ts: float | None = None,
) -> PassingEventMessage:
    return PassingEventMessage(
        ts=ts if ts is not None else time.time(),
        track_id=track_id,
        cls=cls,
        zone=zone,
        direction=direction,
    )


def make_diagnostics(node_id: str = "node_a") -> DiagnosticsMessage:
    return DiagnosticsMessage(
        ts=time.time(),
        node_id=node_id,
        mode="single_cam_homography",
        sources={"cam_a": "alive"},
        frame_count=100,
        fps=25.0,
        latency_ms=LatencyStats(p50=40.0, p95=80.0, p99=120.0, n=100),
        zones=2,
        subscriptions=1,
        calibration=CalibrationFactCheck(loaded=True, rms_ok=True, mode=1),
    )


def make_config(
    node_id: str = "node_a",
    area: str = "zone_a",
    zones: list | None = None,
) -> ConfigMessage:
    if zones is None:
        zones = [
            ZoneSpec(
                name="rack_a",
                kind="palette",
                type="storage",
                severity="info",
                polygon=[[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [0.0, 2.0]],
            )
        ]
    return ConfigMessage(
        ts=time.time(),
        node_id=node_id,
        area=area,
        mode="single_cam_homography",
        cameras=["cam_a"],
        zones=zones,
        calibration=CalibrationFactCheck(loaded=True, rms_ok=True, mode=1),
    )


def make_zone_state(
    zone: str = "rack_a",
    objects: list | None = None,
    ts: float | None = None,
    decision=None,
):
    from backbone.comms.schemas import ZoneObject, ZoneStateMessage
    if objects is None:
        objects = [
            ZoneObject(track_id=1, cls="palette", confidence=0.9, xy_m=(1.0, 1.0)),
        ]
    return ZoneStateMessage(
        ts=ts if ts is not None else time.time(),
        zone=zone,
        objects=tuple(objects),
        count=len(objects),
        cls=tuple(o.cls for o in objects),
        decision=decision,
    )


def make_zone_decision(
    palette_state: str = "palette_loaded",
    content: tuple = ("carton",),
    counts: dict | None = None,
):
    from backbone.comms.schemas import ZoneDecisionModel
    return ZoneDecisionModel(
        palette_state=palette_state,
        content=content,
        counts=counts if counts is not None else {"palette": 1, "carton": 1},
    )
