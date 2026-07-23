"""GET /zones — union of all nodes' config zone advertisements."""

from __future__ import annotations

from backbone.comms.schemas import ZoneSpec

from tests.conftest import make_config, make_zone_state


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


# ---- zone state enrichment (retained zone/<zone> payload over REST) ---------


def test_zones_enriched_with_zone_state(client):
    from tests.conftest import make_zone_state
    sub = client.app.state.subscriber
    sub.update_from_message("node_a", make_config("node_a", area="hall_1"))
    sub.update_from_message("node_a", make_zone_state(zone="rack_a", ts=123.0))

    r = client.get("/zones")
    z = r.json()["zones"][0]
    assert z["count"] == 1
    assert z["state_ts"] == 123.0
    obj = z["objects"][0]
    assert obj["track_id"] == 1
    assert obj["cls"] == "palette"
    assert obj["confidence"] == 0.9


def test_zones_without_state_have_null_objects(client):
    """No zone_state received yet → objects/count/state_ts are null, not []."""
    sub = client.app.state.subscriber
    sub.update_from_message("node_a", make_config("node_a"))

    r = client.get("/zones")
    z = r.json()["zones"][0]
    assert z["objects"] is None
    assert z["count"] is None
    assert z["state_ts"] is None


def test_zone_by_name_returns_spec_and_state(client):
    from tests.conftest import make_zone_state
    sub = client.app.state.subscriber
    sub.update_from_message("node_a", make_config("node_a", area="hall_1"))
    sub.update_from_message("node_a", make_zone_state(zone="rack_a"))

    r = client.get("/zones/rack_a")
    assert r.status_code == 200
    data = r.json()
    assert data["name"] == "rack_a"
    assert data["count"] == 1
    entry = data["zones"][0]
    assert entry["node_id"] == "node_a"
    assert entry["polygon"]
    assert entry["objects"][0]["cls"] == "palette"


def test_zone_by_name_unknown_zone_404(client):
    sub = client.app.state.subscriber
    sub.update_from_message("node_a", make_config("node_a"))
    r = client.get("/zones/nope")
    assert r.status_code == 404


def test_zone_state_explicit_empty_vs_unknown(client):
    """An explicit empty state (count=0) is distinct from no state (null)."""
    from tests.conftest import make_zone_state
    sub = client.app.state.subscriber
    sub.update_from_message("node_a", make_config("node_a"))
    sub.update_from_message("node_a", make_zone_state(zone="rack_a", objects=[]))

    r = client.get("/zones")
    z = r.json()["zones"][0]
    assert z["objects"] == []
    assert z["count"] == 0


def test_six_zones_per_node_all_served(client):
    """The site contract: up to 6 zones per node (zone1-zone6), each monitored.

    The gateway must cache all 6 per-zone states for a node and serve every
    one of them, enriched, on /v1/zones.
    """
    from backbone.comms.schemas import ZoneObject

    from tests.conftest import make_zone_state

    sub = client.app.state.subscriber
    specs = [
        ZoneSpec(name=f"zone{i}", kind="palette", type="storage", severity="info",
                 polygon=[[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [0.0, 2.0]])
        for i in range(1, 7)
    ]
    sub.update_from_message("node_a", make_config("node_a", zones=specs))
    for i in range(1, 7):
        sub.update_from_message("node_a", make_zone_state(
            zone=f"zone{i}",
            objects=[ZoneObject(track_id=i, cls="palette", confidence=0.9,
                                xy_m=(1.0, 1.0))],
            ts=float(i),
        ))

    r = client.get("/zones")
    data = r.json()
    assert data["count"] == 6
    by_name = {z["name"]: z for z in data["zones"]}
    assert set(by_name) == {f"zone{i}" for i in range(1, 7)}
    for i in range(1, 7):
        z = by_name[f"zone{i}"]
        assert z["count"] == 1
        assert z["objects"][0]["track_id"] == i
        assert z["state_ts"] == float(i)

    # And each is individually addressable.
    r = client.get("/zones/zone6")
    assert r.status_code == 200
    assert r.json()["zones"][0]["objects"][0]["track_id"] == 6


# ---- stable zone_id exposure for the /ui schema-tree annotation ----


def _config_with_ids(node_id="node_a"):
    from backbone.comms.schemas import ZoneSpec
    zones = [
        ZoneSpec(name="Sortie_1", zone_id="zp_aaa", kind="palette",
                 type="palette", severity="info",
                 polygon=[[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]]),
        ZoneSpec(name="Sortie_2", zone_id="zp_bbb", kind="palette",
                 type="palette", severity="info",
                 polygon=[[3.0, 0.0], [5.0, 0.0], [5.0, 2.0]]),
    ]
    return make_config(node_id=node_id, zones=zones)


def test_zones_expose_zone_id_and_name(client):
    """The /ui schema tree resolves id-keyed zone topics to display names via
    this pairing — both fields must be present per entry."""
    sub = client.app.state.subscriber
    sub.update_from_message("node_a", _config_with_ids())

    r = client.get("/zones")
    assert r.status_code == 200
    by_id = {z["zone_id"]: z for z in r.json()["zones"]}
    assert by_id["zp_aaa"]["name"] == "Sortie_1"
    assert by_id["zp_bbb"]["name"] == "Sortie_2"


def test_zone_state_joined_by_stable_id(client):
    """State published under the id key must attach to the named zone entry."""
    from backbone.comms.schemas import ZoneStateMessage
    sub = client.app.state.subscriber
    sub.update_from_message("node_a", _config_with_ids())
    st = make_zone_state(zone="Sortie_1")
    st = ZoneStateMessage(**{**st.model_dump(), "zone_id": "zp_aaa"})
    sub.update_from_message("node_a", st)

    r = client.get("/zones")
    entry = {z["zone_id"]: z for z in r.json()["zones"]}["zp_aaa"]
    assert entry["count"] == 1 and entry["objects"] is not None
