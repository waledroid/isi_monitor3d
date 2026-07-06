"""MqttSubscriber — message routing and node cache, hermetic (no broker)."""

from __future__ import annotations

import time

import pytest
from backbone.comms.schemas import ImageRefMessage

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
    return MqttSubscriber("127.0.0.1", 1884, "isiMonitor3D", passings_buffer=5)


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


# ---------------------------------------------------------------------------
# Version-aware topic parsing (versioned + legacy fallback)
# ---------------------------------------------------------------------------

def test_parse_topic_versioned(sub):
    """isiMonitor3D/v1/zone_a/track2d/person → node_id=zone_a, version=v1."""
    node_id, version = sub._parse_topic("isiMonitor3D/v1/zone_a/track2d/person")
    assert node_id == "zone_a"
    assert version == "v1"


def test_parse_topic_legacy_unversioned(sub):
    """Legacy isiMonitor3D/zone_a/track2d/person → node_id=zone_a, version=v0."""
    node_id, version = sub._parse_topic("isiMonitor3D/zone_a/track2d/person")
    assert node_id == "zone_a"
    assert version == "v0"


def test_parse_topic_malformed_returns_none(sub):
    assert sub._parse_topic("isiMonitor3D") is None
    assert sub._parse_topic("other/v1/zone_a/track2d") is None


def test_versioned_message_stores_topic_version(sub):
    """A versioned message recorded via the topic path sets topic_version on the node."""
    msg = make_track2d(track_id=7)
    sub.update_from_message("zone_a", msg, topic_version="v1")
    node = sub.snapshot_nodes()["zone_a"]
    assert node.topic_version == "v1"


def test_legacy_message_defaults_topic_version_v0(sub):
    msg = make_track2d(track_id=7)
    sub.update_from_message("zone_a", msg, topic_version="v0")
    node = sub.snapshot_nodes()["zone_a"]
    assert node.topic_version == "v0"


def test_update_from_message_default_topic_version(sub):
    """Tests that omit topic_version still work (default v1)."""
    sub.update_from_message("zone_a", make_track2d())
    node = sub.snapshot_nodes()["zone_a"]
    assert node.topic_version == "v1"


# ---------------------------------------------------------------------------
# TLS — ca_cert / tls_insecure wiring (paho mocked via autouse conftest)
# ---------------------------------------------------------------------------

def test_tls_with_ca_cert_calls_tls_set_with_ca_certs():
    """tls=True + ca_cert → client.tls_set(ca_certs=<path>) called in start()."""
    from isi_gateway.mqtt_subscriber import MqttSubscriber
    s = MqttSubscriber("127.0.0.1", 8883, "isiMonitor3D", tls=True, ca_cert="/c/ca.crt")
    s.start()
    client = s._client
    client.tls_set.assert_called_once_with(ca_certs="/c/ca.crt")
    client.tls_insecure_set.assert_not_called()
    s.stop()


def test_tls_insecure_calls_tls_insecure_set():
    """tls=True + tls_insecure=True → client.tls_insecure_set(True) called in start()."""
    from isi_gateway.mqtt_subscriber import MqttSubscriber
    s = MqttSubscriber(
        "127.0.0.1", 8883, "isiMonitor3D",
        tls=True, ca_cert="/c/ca.crt", tls_insecure=True,
    )
    s.start()
    client = s._client
    client.tls_set.assert_called_once_with(ca_certs="/c/ca.crt")
    client.tls_insecure_set.assert_called_once_with(True)
    s.stop()


def test_tls_false_skips_tls_set_and_tls_insecure_set():
    """tls=False → neither tls_set nor tls_insecure_set is called in start()."""
    from isi_gateway.mqtt_subscriber import MqttSubscriber
    s = MqttSubscriber("127.0.0.1", 1884, "isiMonitor3D", tls=False, tls_insecure=True)
    s.start()
    client = s._client
    client.tls_set.assert_not_called()
    client.tls_insecure_set.assert_not_called()
    s.stop()


def test_tls_no_ca_cert_uses_system_cas():
    """tls=True with default ca_cert=None → tls_set(ca_certs=None) for system CAs."""
    from isi_gateway.mqtt_subscriber import MqttSubscriber
    s = MqttSubscriber("127.0.0.1", 8883, "isiMonitor3D", tls=True)
    s.start()
    client = s._client
    client.tls_set.assert_called_once_with(ca_certs=None)
    client.tls_insecure_set.assert_not_called()
    s.stop()
