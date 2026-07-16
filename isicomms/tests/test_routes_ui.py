"""The probe surface: /recent (raw tail + counters) and /ui (the page shell).

The tail must show EVERY arriving message — malformed included (that is the
point of a probe) — newest last, bounded by the ring buffer. The page shell
carries no data, so it stays reachable without a token even when the API is
Bearer-protected; its JS supplies the token to the data endpoints.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

from fastapi.testclient import TestClient

from isicomms.app import create_app
from isicomms.config import Settings
from isicomms.mqtt_subscriber import MqttSubscriber


def _msg(topic: str, payload: bytes) -> SimpleNamespace:
    return SimpleNamespace(topic=topic, payload=payload)


def _diag_payload(node="n1") -> bytes:
    return json.dumps({
        "schema_version": 6, "type": "diagnostics", "ts": time.time(),
        "node_id": node, "mode": "m", "sources": {}, "frame_count": 1,
        "fps": 1.0, "fps_by_camera": {},
        "latency_ms": {"p50": 1.0, "p95": 2.0, "p99": 3.0, "n": 1},
        "zones": 0, "subscriptions": 0,
        "calibration": {"loaded": True, "rms_ok": True, "mode": 2},
    }).encode()


# ---- ring buffer ------------------------------------------------------------


def test_recent_records_valid_and_malformed_newest_last():
    sub = MqttSubscriber(host="unused", port=1883, base="isiMonitor3D", recent_buffer=10)
    sub._on_message(None, None, _msg("isiMonitor3D/v1/n1/diagnostics/heartbeat",
                                     _diag_payload()))
    sub._on_message(None, None, _msg("garbage-topic", b"not json at all"))
    got = sub.recent(10)
    assert len(got) == 2
    assert got[0]["topic"].endswith("heartbeat")
    assert got[1]["topic"] == "garbage-topic"       # malformed still tailed
    assert got[1]["payload"] == "not json at all"
    assert got[1]["bytes"] == len(b"not json at all")


def test_recent_ring_is_bounded_and_limit_slices():
    sub = MqttSubscriber(host="unused", port=1883, base="isiMonitor3D", recent_buffer=5)
    for i in range(9):
        sub._on_message(None, None, _msg(f"t/{i}", b"{}"))
    assert [m["topic"] for m in sub.recent(100)] == [f"t/{i}" for i in range(4, 9)]
    assert [m["topic"] for m in sub.recent(2)] == ["t/7", "t/8"]


# ---- endpoints --------------------------------------------------------------


def _client(token=None):
    cfg = Settings(mqtt_host="unused", api_token=token)
    app = create_app(cfg)
    return TestClient(app)


def test_recent_endpoint_shape_and_limit():
    with _client() as c:
        sub = c.app.state.subscriber
        for i in range(4):
            sub._on_message(None, None, _msg(f"t/{i}", b"{}"))
        r = c.get("/recent?limit=3")
        assert r.status_code == 200
        body = r.json()
        assert body["count"] == 3
        assert [m["topic"] for m in body["messages"]] == ["t/1", "t/2", "t/3"]
        assert set(body["stats"]) == {"received", "dropped_malformed",
                                      "dropped_version"}
        # versioned alias too
        assert c.get("/v1/recent").status_code == 200


def test_recent_requires_token_when_set():
    with _client(token="s3cret") as c:
        assert c.get("/recent").status_code == 401
        assert c.get("/recent",
                     headers={"Authorization": "Bearer s3cret"}).status_code == 200


def test_ui_shell_is_tokenless_html():
    with _client(token="s3cret") as c:
        r = c.get("/ui")                       # no Authorization header
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/html")
        for probe in ("/nodes", "/zones", "/recent", "isicomms"):
            assert probe in r.text


def test_topics_map_latest_and_count():
    sub = MqttSubscriber(host="unused", port=1883, base="isiMonitor3D", recent_buffer=3)
    for i in range(5):                           # ring holds 3; topics map holds all
        sub._on_message(None, None, _msg("a/b", json.dumps({"i": i}).encode()))
    sub._on_message(None, None, _msg("a/c", b"{}"))
    topics = sub.topics()
    assert set(topics) == {"a/b", "a/c"}
    assert topics["a/b"]["count"] == 5           # counts every arrival
    assert json.loads(topics["a/b"]["payload"]) == {"i": 4}   # latest wins
    # a topic pushed out of the RING still lives in the tree map
    assert all(m["topic"] != "a/b" or json.loads(m["payload"])["i"] >= 2
               for m in sub.recent(10))


def test_recent_endpoint_includes_topics():
    with _client() as c:
        sub = c.app.state.subscriber
        sub._on_message(None, None, _msg("x/y", b"{}"))
        body = c.get("/recent").json()
        assert "x/y" in body["topics"]
        assert body["topics"]["x/y"]["count"] == 1
