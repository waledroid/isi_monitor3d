"""`/ws/overlays` — the per-camera observation feed for client-side overlays.

Hermetic: the app's bus is swapped for a fake whose snapshot the test controls.
Pins the protocol: subscribe with ``{"cameras": [...]}``; one JSON payload per
camera per NEW ``ts`` (silent while unchanged); unsubscribed cameras never
sent; optional det fields omitted when absent.
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from backbone.comms.schemas import ObservationDet, ObservationsMessage
from fastapi.testclient import TestClient

from monitor_web.app import create_app
from monitor_web.config import Settings


class _FakeBus:
    def __init__(self) -> None:
        self.observations: dict[str, ObservationsMessage] = {}

    def snapshot(self):
        return SimpleNamespace(observations_by_camera=dict(self.observations))


def _obs(camera_id: str, ts: float, *, with_optionals: bool = False) -> ObservationsMessage:
    kwargs = {}
    if with_optionals:
        kwargs["mask_poly"] = ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))
        kwargs["keypoints_uv"] = tuple((10.0 + i, 20.0 + i, 0.5) for i in range(17))
    det = ObservationDet(
        cls="person" if with_optionals else "palette",
        confidence=0.9,
        bbox_xyxy=(10.0, 20.0, 110.0, 220.0),
        foot_uv=(60.0, 220.0),
        **kwargs,
    )
    return ObservationsMessage(
        ts=ts, camera_id=camera_id, frame_wh=(1920, 1080), dets=(det,),
    )


@pytest.fixture
def client(tmp_path: Path):
    backbone_yaml = tmp_path / "backbone.yaml"
    backbone_yaml.write_text(yaml.safe_dump({
        "cameras": {"cam_a": {"source": {"name": "replay", "frames": []}}},
        "metadata": {"sinks": [{"plugin": "udp", "host": "127.0.0.1", "port": 0}]},
    }))
    cfg = Settings(backbone_config_path=backbone_yaml, udp_port=0, port=0)
    with TestClient(create_app(cfg)) as c:
        yield c


def test_overlays_push_on_ts_change_only(client):
    fake = _FakeBus()
    client.app.state.bus = fake
    fake.observations["cam_a"] = _obs("cam_a", 1.0, with_optionals=True)
    fake.observations["cam_b"] = _obs("cam_b", 1.0)   # never subscribed

    with client.websocket_connect("/ws/overlays") as ws:
        ws.send_json({"cameras": ["cam_a"]})
        msg = ws.receive_json()
        assert msg["camera_id"] == "cam_a"
        assert msg["ts"] == 1.0
        assert msg["frame_wh"] == [1920, 1080]
        det = msg["dets"][0]
        assert det["cls"] == "person"
        assert det["bbox_xyxy"] == [10.0, 20.0, 110.0, 220.0]
        assert det["foot_uv"] == [60.0, 220.0]
        assert len(det["keypoints_uv"]) == 17
        assert len(det["mask_poly"]) == 3

        # Many poll cycles with an UNCHANGED ts → silent. Then bump the ts:
        # the very NEXT message must be the new one (no duplicate of ts=1.0
        # queued in between — FIFO order proves the silence), and never cam_b.
        time.sleep(0.3)
        fake.observations["cam_a"] = _obs("cam_a", 2.0)
        msg2 = ws.receive_json()
        assert msg2["camera_id"] == "cam_a"
        assert msg2["ts"] == 2.0
        det2 = msg2["dets"][0]
        assert "mask_poly" not in det2 and "keypoints_uv" not in det2


def test_overlays_silent_with_no_observations(client):
    """No bus traffic (Backbone stopped) → the socket stays open and silent;
    a resubscribe message is accepted without error."""
    fake = _FakeBus()
    client.app.state.bus = fake
    with client.websocket_connect("/ws/overlays") as ws:
        ws.send_json({"cameras": ["cam_a", "cam_b"]})
        time.sleep(0.2)
        # Change the subscription set, then feed cam_b only: proves the second
        # message was applied and nothing was pushed while the bus was empty.
        ws.send_json({"cameras": ["cam_b"]})
        time.sleep(0.3)      # let the server apply the new set first
        fake.observations["cam_a"] = _obs("cam_a", 5.0)
        fake.observations["cam_b"] = _obs("cam_b", 6.0)
        msg = ws.receive_json()
        assert msg["camera_id"] == "cam_b"
        assert msg["ts"] == 6.0
