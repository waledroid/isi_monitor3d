"""GET /config — per-node config advertisement."""

from __future__ import annotations

from tests.conftest import make_config, make_track2d


def test_config_empty_initially(client):
    r = client.get("/config")
    assert r.status_code == 200
    assert r.json() == {"nodes": [], "count": 0}


def test_config_returned_after_config_message(client):
    sub = client.app.state.subscriber
    cfg = make_config("node_a", area="hall_1")
    sub.update_from_message("node_a", cfg)

    r = client.get("/config")
    data = r.json()
    assert data["count"] == 1
    node = data["nodes"][0]
    assert node["node_id"] == "node_a"
    assert node["config"]["area"] == "hall_1"
    assert node["config"]["mode"] == "single_cam_homography"
    assert "cam_a" in node["config"]["cameras"]


def test_config_null_when_only_track_received(client):
    sub = client.app.state.subscriber
    sub.update_from_message("node_a", make_track2d())

    r = client.get("/config")
    data = r.json()
    assert data["count"] == 1
    assert data["nodes"][0]["config"] is None


def test_config_two_nodes(client):
    sub = client.app.state.subscriber
    sub.update_from_message("node_a", make_config("node_a", area="hall_1"))
    sub.update_from_message("node_b", make_config("node_b", area="hall_2"))

    r = client.get("/config")
    data = r.json()
    assert data["count"] == 2
    areas = {n["node_id"]: n["config"]["area"] for n in data["nodes"]}
    assert areas == {"node_a": "hall_1", "node_b": "hall_2"}
