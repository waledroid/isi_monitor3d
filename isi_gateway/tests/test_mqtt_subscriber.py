"""MqttSubscriber — message routing and node cache, hermetic (no broker)."""

from __future__ import annotations

import time

import pytest
from backbone.metadata.schemas import ImageRefMessage

from isi_gateway.mqtt_subscriber import MqttSubscriber
from tests.conftest import (
    make_config,
    make_diagnostics,
    make_passing,
    make_track2d,
    make_track3d,
)


@pytest.fixture
def sub() -> MqttSubscriber:
    return MqttSubscriber("127.0.0.1", 1884, "isi", passings_buffer=5)


def test_track2d_cached_per_node(sub):
    msg = make_track2d(track_id=7, cls="palette", xy_m=(1.0, 2.0))
    sub.update_from_message("node_a", msg)
    nodes = sub.snapshot_nodes()
    assert "node_a" in nodes
    assert nodes["node_a"].last_track2d_by_id[7] is msg


def test_track2d_overwrites_same_id(sub):
    m1 = make_track2d(track_id=1, xy_m=(0.0, 0.0))
    m2 = make_track2d(track_id=1, xy_m=(5.0, 5.0))
    sub.update_from_message("node_a", m1)
    sub.update_from_message("node_a", m2)
    node = sub.snapshot_nodes()["node_a"]
    assert node.last_track2d_by_id[1].xy_m == (5.0, 5.0)


def test_track3d_cached_per_node(sub):
    msg = make_track3d(track_id=3)
    sub.update_from_message("node_b", msg)
    nodes = sub.snapshot_nodes()
    assert "node_b" in nodes
    assert nodes["node_b"].last_track3d_by_id[3] is msg


def test_passing_appended_to_deque(sub):
    for i in range(3):
        sub.update_from_message("node_a", make_passing(track_id=i, zone="rack_a"))
    node = sub.snapshot_nodes()["node_a"]
    assert len(node.passings) == 3


def test_passings_deque_respects_maxlen(sub):
    """passings_buffer=5 → oldest are dropped after 5 items."""
    for i in range(8):
        sub.update_from_message("node_a", make_passing(track_id=i))
    node = sub.snapshot_nodes()["node_a"]
    assert len(node.passings) == 5


def test_diagnostics_stored(sub):
    msg = make_diagnostics("node_a")
    sub.update_from_message("node_a", msg)
    node = sub.snapshot_nodes()["node_a"]
    assert node.last_diagnostics is msg


def test_config_stored(sub):
    msg = make_config("node_a", area="hall_1")
    sub.update_from_message("node_a", msg)
    node = sub.snapshot_nodes()["node_a"]
    assert node.config is msg
    assert node.config.area == "hall_1"


def test_image_ref_bumps_last_seen_only(sub):
    """ImageRefMessage is not cached explicitly — only last_seen is updated."""
    before = time.time()
    img_ref = ImageRefMessage(
        ts=before,
        track_id=1,
        cls="palette",
        zone="rack_a",
        url="file:///tmp/snap.jpg",
    )
    sub.update_from_message("node_a", img_ref)
    node = sub.snapshot_nodes()["node_a"]
    # No track2d/3d, no passings, no diagnostics, no config — just last_seen bumped.
    assert node.last_seen >= before
    assert len(node.last_track2d_by_id) == 0
    assert node.last_diagnostics is None


def test_multiple_nodes_isolated(sub):
    sub.update_from_message("node_a", make_track2d(track_id=1))
    sub.update_from_message("node_b", make_track2d(track_id=2))
    nodes = sub.snapshot_nodes()
    assert 1 in nodes["node_a"].last_track2d_by_id
    assert 2 in nodes["node_b"].last_track2d_by_id
    assert 1 not in nodes["node_b"].last_track2d_by_id


def test_node_alive_freshness(sub):
    sub.update_from_message("node_a", make_track2d())
    now = time.time()
    assert sub.node_alive("node_a", now, stale_after=5.0)
    assert not sub.node_alive("node_a", now + 100, stale_after=5.0)


def test_node_alive_unknown_node(sub):
    assert not sub.node_alive("ghost", time.time(), stale_after=5.0)


def test_stats_count_received(sub):
    sub.update_from_message("node_a", make_track2d())
    sub.update_from_message("node_a", make_track3d())
    assert sub.stats()["received"] == 2


def test_stop_is_idempotent_without_start(sub):
    """stop() on a never-started subscriber must not raise."""
    sub.stop()
    sub.stop()
