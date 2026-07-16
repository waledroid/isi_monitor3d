"""GET /tracks — track aggregation, node/cls/zone filters."""

from __future__ import annotations

from backbone.comms.schemas import ZoneSpec

from tests.conftest import make_config, make_track2d, make_track3d


def test_tracks_empty_initially(client):
    r = client.get("/tracks")
    assert r.status_code == 200
    assert r.json() == {"tracks": [], "count": 0}


def test_tracks_from_two_nodes_tagged(client):
    sub = client.app.state.subscriber
    sub.update_from_message("node_a", make_track2d(track_id=1, cls="palette"))
    sub.update_from_message("node_b", make_track2d(track_id=2, cls="palette"))

    r = client.get("/tracks")
    data = r.json()
    assert data["count"] == 2
    node_ids = {t["node_id"] for t in data["tracks"]}
    assert node_ids == {"node_a", "node_b"}


def test_tracks_filter_by_node(client):
    sub = client.app.state.subscriber
    sub.update_from_message("node_a", make_track2d(track_id=1))
    sub.update_from_message("node_b", make_track2d(track_id=2))

    r = client.get("/tracks?node=node_a")
    data = r.json()
    assert data["count"] == 1
    assert data["tracks"][0]["node_id"] == "node_a"


def test_tracks_filter_by_cls(client):
    sub = client.app.state.subscriber
    sub.update_from_message("node_a", make_track2d(track_id=1, cls="palette"))
    sub.update_from_message("node_a", make_track2d(track_id=2, cls="carton"))

    r = client.get("/tracks?cls=palette")
    data = r.json()
    assert data["count"] == 1
    assert data["tracks"][0]["cls"] == "palette"


def test_tracks_includes_track3d(client):
    sub = client.app.state.subscriber
    sub.update_from_message("node_a", make_track3d(track_id=5, xyz_m=(1.0, 1.0, 0.0)))

    r = client.get("/tracks")
    data = r.json()
    assert data["count"] == 1
    assert data["tracks"][0]["track_id"] == 5
    assert "xyz_m" in data["tracks"][0]


def test_tracks_filter_by_zone_inside(client):
    """Tracks whose xy_m lies inside the named zone are kept."""
    sub = client.app.state.subscriber
    # Zone: [0,0]→[4,0]→[4,2]→[0,2] (rectangle, area 8 m²).
    cfg = make_config("node_a", zones=[
        ZoneSpec(
            name="rack_a",
            kind="palette",
            type="storage",
            severity="info",
            polygon=[[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [0.0, 2.0]],
        )
    ])
    sub.update_from_message("node_a", cfg)

    # Inside.
    sub.update_from_message("node_a", make_track2d(track_id=1, xy_m=(2.0, 1.0)))
    # Outside.
    sub.update_from_message("node_a", make_track2d(track_id=2, xy_m=(10.0, 10.0)))

    r = client.get("/tracks?zone=rack_a")
    data = r.json()
    assert data["count"] == 1
    assert data["tracks"][0]["track_id"] == 1


def test_tracks_zone_filter_unknown_zone_returns_all_unfiltered(client):
    """Unknown zone name: zone_filter is None → no zone filtering applied."""
    sub = client.app.state.subscriber
    sub.update_from_message("node_a", make_track2d(track_id=1))
    sub.update_from_message("node_a", make_track2d(track_id=2))

    r = client.get("/tracks?zone=nonexistent")
    # Zone not found means zone_filter=None → no filter → all tracks returned.
    data = r.json()
    assert data["count"] == 2


def test_tracks_zone_filter_track3d_by_xy(client):
    """Zone filter uses xyz_m[:2] for 3D tracks."""
    sub = client.app.state.subscriber
    cfg = make_config("node_a", zones=[
        ZoneSpec(
            name="rack_b",
            kind="palette",
            type="storage",
            severity="info",
            polygon=[[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [0.0, 2.0]],
        )
    ])
    sub.update_from_message("node_a", cfg)
    sub.update_from_message("node_a", make_track3d(track_id=10, xyz_m=(1.0, 1.0, 1.5)))

    r = client.get("/tracks?zone=rack_b")
    data = r.json()
    assert data["count"] == 1


def test_tracks_node_and_cls_combined(client):
    sub = client.app.state.subscriber
    sub.update_from_message("node_a", make_track2d(track_id=1, cls="palette"))
    sub.update_from_message("node_a", make_track2d(track_id=2, cls="carton"))
    sub.update_from_message("node_b", make_track2d(track_id=3, cls="palette"))

    r = client.get("/tracks?node=node_a&cls=palette")
    data = r.json()
    assert data["count"] == 1
    assert data["tracks"][0]["node_id"] == "node_a"
    assert data["tracks"][0]["cls"] == "palette"
