"""GET /nodes — node listing with alive/stale status."""

from __future__ import annotations

import time

from tests.conftest import make_config, make_diagnostics, make_track2d


def test_nodes_empty_initially(client):
    r = client.get("/nodes")
    assert r.status_code == 200
    assert r.json() == {"nodes": [], "count": 0}


def test_nodes_shows_two_nodes(client):
    sub = client.app.state.subscriber
    sub.update_from_message("node_a", make_track2d())
    sub.update_from_message("node_b", make_track2d())

    r = client.get("/nodes")
    assert r.status_code == 200
    data = r.json()
    assert data["count"] == 2
    ids = {n["node_id"] for n in data["nodes"]}
    assert ids == {"node_a", "node_b"}


def test_nodes_alive_vs_stale(client):
    """node_a is freshly seen; node_b's last_seen is backdated past the threshold."""
    sub = client.app.state.subscriber
    settings = client.app.state.settings

    sub.update_from_message("node_a", make_track2d())
    sub.update_from_message("node_b", make_track2d())

    # Backdate node_b so it appears stale.
    with sub._lock:
        sub._nodes["node_b"].last_seen = time.time() - settings.node_stale_after_s - 1

    r = client.get("/nodes")
    assert r.status_code == 200
    nodes = {n["node_id"]: n for n in r.json()["nodes"]}
    assert nodes["node_a"]["status"] == "alive"
    assert nodes["node_b"]["status"] == "stale"


def test_nodes_populated_from_config_and_diagnostics(client):
    sub = client.app.state.subscriber
    sub.update_from_message("node_a", make_config("node_a", area="hall_1"))
    sub.update_from_message("node_a", make_diagnostics("node_a"))

    r = client.get("/nodes")
    node = r.json()["nodes"][0]
    assert node["area"] == "hall_1"
    assert node["mode"] == "single_cam_homography"
    assert "cam_a" in node["cameras"]
    assert node["fps"] == 25.0
    assert node["latency_ms"] == 80.0   # p95 from make_diagnostics


def test_nodes_no_config_fields_are_null(client):
    sub = client.app.state.subscriber
    sub.update_from_message("node_a", make_track2d())

    r = client.get("/nodes")
    node = r.json()["nodes"][0]
    assert node["area"] is None
    assert node["mode"] is None
    assert node["cameras"] == []
    assert node["latency_ms"] is None
    assert node["fps"] is None


def test_nodes_surface_topic_version(client):
    """/nodes reflects each node's topic_version."""
    sub = client.app.state.subscriber
    sub.update_from_message("node_a", make_track2d(), topic_version="v1")
    sub.update_from_message("node_b", make_track2d(), topic_version="v0")

    r = client.get("/nodes")
    nodes = {n["node_id"]: n for n in r.json()["nodes"]}
    assert nodes["node_a"]["topic_version"] == "v1"
    assert nodes["node_b"]["topic_version"] == "v0"


def test_nodes_v1_prefix_matches_bare_alias(client):
    """/v1/nodes returns the same shape as the bare /nodes alias."""
    sub = client.app.state.subscriber
    sub.update_from_message("node_a", make_track2d())

    bare = client.get("/nodes")
    versioned = client.get("/v1/nodes")
    assert versioned.status_code == 200
    assert versioned.json() == bare.json()
