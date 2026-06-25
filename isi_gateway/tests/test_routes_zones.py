"""GET /zones — union of all nodes' config zone advertisements."""

from __future__ import annotations

from backbone.comms.schemas import ZoneSpec

from tests.conftest import make_config


def test_zones_empty_initially(client):
    r = client.get("/zones")
    assert r.status_code == 200
    assert r.json() == {"zones": [], "count": 0}


def test_zones_from_single_node(client):
    sub = client.app.state.subscriber
    sub.update_from_message("node_a", make_config("node_a", area="hall_1"))

    r = client.get("/zones")
    data = r.json()
    assert data["count"] == 1
    z = data["zones"][0]
    assert z["node_id"] == "node_a"
    assert z["area"] == "hall_1"
    assert z["name"] == "rack_a"
    assert z["kind"] == "palette"
    assert z["type"] == "storage"
    assert z["severity"] == "info"
    assert isinstance(z["polygon"], list)


def test_zones_union_from_two_nodes(client):
    sub = client.app.state.subscriber

    cfg_a = make_config("node_a", area="hall_1", zones=[
        ZoneSpec(name="rack_a", kind="palette", type="storage", severity="info",
                 polygon=[[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [0.0, 2.0]]),
    ])
    cfg_b = make_config("node_b", area="hall_2", zones=[
        ZoneSpec(name="danger_north", kind="danger", type="danger", severity="critical",
                 polygon=[[5.0, 0.0], [9.0, 0.0], [9.0, 3.0], [5.0, 3.0]]),
    ])
    sub.update_from_message("node_a", cfg_a)
    sub.update_from_message("node_b", cfg_b)

    r = client.get("/zones")
    data = r.json()
    assert data["count"] == 2
    names = {z["name"] for z in data["zones"]}
    assert names == {"rack_a", "danger_north"}
    severities = {z["name"]: z["severity"] for z in data["zones"]}
    assert severities["danger_north"] == "critical"


def test_zones_node_without_config_omitted(client):
    sub = client.app.state.subscriber
    # node_b only has a track, no config.
    from tests.conftest import make_track2d
    sub.update_from_message("node_b", make_track2d())
    sub.update_from_message("node_a", make_config("node_a"))

    r = client.get("/zones")
    data = r.json()
    # Only node_a has a config — node_b is omitted.
    assert data["count"] == 1
    assert data["zones"][0]["node_id"] == "node_a"
