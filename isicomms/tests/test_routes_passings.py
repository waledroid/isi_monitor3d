"""GET /passings — zone-passing events with limit and node filters."""

from __future__ import annotations

import time

from tests.conftest import make_passing


def test_passings_empty_initially(client):
    r = client.get("/passings")
    assert r.status_code == 200
    assert r.json() == {"passings": [], "count": 0}


def test_passings_tagged_with_node_id(client):
    sub = client.app.state.subscriber
    sub.update_from_message("node_a", make_passing(zone="rack_a"))
    sub.update_from_message("node_b", make_passing(zone="rack_b"))

    r = client.get("/passings")
    data = r.json()
    assert data["count"] == 2
    node_ids = {p["node_id"] for p in data["passings"]}
    assert node_ids == {"node_a", "node_b"}


def test_passings_filter_by_node(client):
    sub = client.app.state.subscriber
    sub.update_from_message("node_a", make_passing(zone="rack_a"))
    sub.update_from_message("node_b", make_passing(zone="rack_b"))

    r = client.get("/passings?node=node_a")
    data = r.json()
    assert data["count"] == 1
    assert data["passings"][0]["node_id"] == "node_a"


def test_passings_limit(client):
    sub = client.app.state.subscriber
    for i in range(8):
        sub.update_from_message("node_a", make_passing(track_id=i, ts=float(i)))

    r = client.get("/passings?limit=3")
    data = r.json()
    assert data["count"] == 3
    # newest-last: the last 3 are [5, 6, 7]
    track_ids = [p["track_id"] for p in data["passings"]]
    assert track_ids == [5, 6, 7]


def test_passings_sorted_newest_last(client):
    sub = client.app.state.subscriber
    now = time.time()
    sub.update_from_message("node_a", make_passing(track_id=2, ts=now + 1.0))
    sub.update_from_message("node_a", make_passing(track_id=1, ts=now))

    r = client.get("/passings")
    data = r.json()
    assert data["count"] == 2
    assert data["passings"][0]["track_id"] == 1
    assert data["passings"][1]["track_id"] == 2


def test_passings_clamped_to_buffer(client):
    """Buffer is 10 (from conftest Settings); sending 12 keeps only the last 10."""
    sub = client.app.state.subscriber
    for i in range(12):
        sub.update_from_message("node_a", make_passing(track_id=i, ts=float(i)))

    r = client.get("/passings")
    data = r.json()
    # The deque itself is capped at 10 (passings_buffer in NodeState).
    assert data["count"] == 10
