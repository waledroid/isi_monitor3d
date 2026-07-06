"""``MqttSink`` — hermetic tests; no real broker required.

All paho I/O is replaced by a ``MagicMock`` so tests run without a running
MQTT broker. Mirrors the structure of ``tests/test_udp_sink.py``.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from backbone.comms.schemas import (
    SCHEMA_VERSION,
    CalibrationFactCheck,
    ConfigMessage,
    DiagnosticsMessage,
    LatencyStats,
    MessageType,
)
from backbone.core.interfaces import metadata_sink_registry
from backbone.core.types import Track2D, Track3D
from backbone.shared.zone_transitions import PassingEvent

# ---------------------------------------------------------------------------
# Track factories — identical pattern to test_udp_sink.py
# ---------------------------------------------------------------------------

def _t2() -> Track2D:
    return Track2D(
        track_id=7, cls="palette", capture_ts=99999.0,
        xy_m=(3.0, 4.0), vxy_m=(0.1, -0.1),
        confidence=0.9, cameras_seeing=("cam_a",),
    )


def _t3() -> Track3D:
    return Track3D(
        track_id=7, cls="palette", capture_ts=99999.0,
        xyz_m=(3.0, 4.0, 1.5), vxyz_m=(0.0, 0.0, 0.0),
        contributing_cameras=("cam_a", "cam_b"),
        max_reprojection_error_px=3.0,
        keypoints_xyz=None,
    )


# ---------------------------------------------------------------------------
# Helper: build a MqttSink with paho patched out
# ---------------------------------------------------------------------------

def _make_sink(**kwargs):
    """Instantiate MqttSink with a mocked paho Client."""
    with patch("backbone.comms.mqtt_sink.mqtt.Client") as MockClient:
        mock_instance = MagicMock()
        MockClient.return_value = mock_instance
        # Import here so the module is already loaded; patch targets the
        # already-imported name in mqtt_sink's namespace.
        from backbone.comms.mqtt_sink import MqttSink
        sink = MqttSink(**kwargs)
        return sink, mock_instance


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_plugin_registered_under_mqtt() -> None:
    """``"mqtt"`` appears in the registry after importing backbone.comms."""
    import backbone.comms  # noqa: F401

    assert "mqtt" in metadata_sink_registry


def test_construction_via_registry_calls_loop_start() -> None:
    """Creating via the registry returns a MqttSink and starts the loop."""
    with patch("backbone.comms.mqtt_sink.mqtt.Client") as MockClient:
        mock_instance = MagicMock()
        MockClient.return_value = mock_instance

        from backbone.comms.mqtt_sink import MqttSink
        sink = metadata_sink_registry.create("mqtt", host="127.0.0.1", port=1883)
        assert isinstance(sink, MqttSink)
        mock_instance.loop_start.assert_called_once()
        sink.close()


def test_publish_track_2d_correct_topic_and_payload() -> None:
    """publish_track_2d calls client.publish with the per-class topic and valid JSON."""
    with patch("backbone.comms.mqtt_sink.mqtt.Client") as MockClient:
        mock_instance = MagicMock()
        MockClient.return_value = mock_instance

        from backbone.comms.mqtt_sink import MqttSink
        sink = MqttSink(host="127.0.0.1", port=1883, prefix="isiMonitor3D/v1/node")
        sink.publish_track_2d(_t2())

        mock_instance.publish.assert_called_once()
        call_args = mock_instance.publish.call_args
        topic = call_args[0][0]
        payload_bytes = call_args[0][1]

        assert topic == "isiMonitor3D/v1/node/track2d/palette"
        msg = json.loads(payload_bytes.decode("utf-8"))
        assert msg["schema_version"] == SCHEMA_VERSION
        assert msg["type"] == MessageType.TRACK_2D.value
        assert msg["track_id"] == 7
        assert msg["cls"] == "palette"

        sink.close()


def test_publish_track_3d_correct_topic_and_payload() -> None:
    """publish_track_3d calls client.publish with the per-class topic and xyz_m present."""
    with patch("backbone.comms.mqtt_sink.mqtt.Client") as MockClient:
        mock_instance = MagicMock()
        MockClient.return_value = mock_instance

        from backbone.comms.mqtt_sink import MqttSink
        sink = MqttSink(host="127.0.0.1", port=1883, prefix="isiMonitor3D/v1/node")
        sink.publish_track_3d(_t3())

        mock_instance.publish.assert_called_once()
        call_args = mock_instance.publish.call_args
        topic = call_args[0][0]
        payload_bytes = call_args[0][1]

        assert topic == "isiMonitor3D/v1/node/track3d/palette"
        msg = json.loads(payload_bytes.decode("utf-8"))
        assert msg["schema_version"] == SCHEMA_VERSION
        assert msg["type"] == MessageType.TRACK_3D.value
        assert "xyz_m" in msg
        assert msg["xyz_m"] == [3.0, 4.0, 1.5]

        sink.close()


def test_port_validation() -> None:
    """Port values 0 and 65536 must raise ValueError."""
    with patch("backbone.comms.mqtt_sink.mqtt.Client"):
        from backbone.comms.mqtt_sink import MqttSink

        with pytest.raises(ValueError, match="port"):
            MqttSink(host="127.0.0.1", port=0)
        with pytest.raises(ValueError, match="port"):
            MqttSink(host="127.0.0.1", port=65536)


def test_qos_validation() -> None:
    """QoS values outside {0, 1, 2} must raise ValueError."""
    with patch("backbone.comms.mqtt_sink.mqtt.Client"):
        from backbone.comms.mqtt_sink import MqttSink

        with pytest.raises(ValueError, match="qos"):
            MqttSink(host="127.0.0.1", port=1883, qos=3)
        with pytest.raises(ValueError, match="qos"):
            MqttSink(host="127.0.0.1", port=1883, qos=-1)


def test_publish_swallows_client_error() -> None:
    """A client.publish() exception must not propagate out of publish_track_2d."""
    with patch("backbone.comms.mqtt_sink.mqtt.Client") as MockClient:
        mock_instance = MagicMock()
        mock_instance.publish.side_effect = RuntimeError("broker gone")
        MockClient.return_value = mock_instance

        from backbone.comms.mqtt_sink import MqttSink
        sink = MqttSink(host="127.0.0.1", port=1883)
        # Must not raise, even though client.publish raises.
        sink.publish_track_2d(_t2())
        sink.close()


def test_publish_event_correct_topic_and_payload() -> None:
    """publish_event calls client.publish with the per-zone passings topic and valid JSON."""
    with patch("backbone.comms.mqtt_sink.mqtt.Client") as MockClient:
        mock_instance = MagicMock()
        MockClient.return_value = mock_instance

        from backbone.comms.mqtt_sink import MqttSink
        sink = MqttSink(host="127.0.0.1", port=1883, prefix="isiMonitor3D/v1/node")
        ev = PassingEvent(track_id=3, cls="palette", zone="B3D", direction="enter", ts=5.0)
        sink.publish_event(ev)

        mock_instance.publish.assert_called_once()
        call_args = mock_instance.publish.call_args
        topic = call_args[0][0]
        payload_bytes = call_args[0][1]

        assert topic == "isiMonitor3D/v1/node/zone/B3D/passings"
        msg = json.loads(payload_bytes.decode("utf-8"))
        assert msg["type"] == MessageType.PASSING.value
        assert msg["schema_version"] == SCHEMA_VERSION
        assert msg["zone"] == "B3D"
        assert msg["direction"] == "enter"
        assert msg["track_id"] == 3

        sink.close()


def test_publish_event_sanitises_zone_name() -> None:
    """Zone names containing MQTT wildcards are sanitised before topic formatting."""
    with patch("backbone.comms.mqtt_sink.mqtt.Client") as MockClient:
        mock_instance = MagicMock()
        MockClient.return_value = mock_instance

        from backbone.comms.mqtt_sink import MqttSink
        sink = MqttSink(host="127.0.0.1", port=1883, prefix="isiMonitor3D/v1/node")
        ev = PassingEvent(track_id=1, cls="person", zone="zone/A+B#C", direction="leave", ts=1.0)
        sink.publish_event(ev)

        call_args = mock_instance.publish.call_args
        topic = call_args[0][0]
        assert topic == "isiMonitor3D/v1/node/zone/zone_A_B_C/passings"
        sink.close()


def test_publish_image_ref_correct_topic_and_payload() -> None:
    """publish_image_ref sends an ImageRefMessage with the URL but no image bytes."""
    with patch("backbone.comms.mqtt_sink.mqtt.Client") as MockClient:
        mock_instance = MagicMock()
        MockClient.return_value = mock_instance

        from backbone.comms.mqtt_sink import MqttSink
        from backbone.comms.schemas import MessageType
        sink = MqttSink(host="127.0.0.1", port=1883, prefix="isiMonitor3D/v1/node")
        sink.publish_image_ref(
            track_id=42,
            cls="palette",
            zone="B3D",
            ts=5.0,
            url="file:///var/lib/isi_monitor3d/snapshots/5000_B3D_42.jpg",
        )

        mock_instance.publish.assert_called_once()
        call_args = mock_instance.publish.call_args
        topic = call_args[0][0]
        payload_bytes = call_args[0][1]

        # Topic must embed zone + track_id, inside the zone/ folder
        assert topic == "isiMonitor3D/v1/node/zone/B3D/images/42"

        msg = json.loads(payload_bytes.decode("utf-8"))
        assert msg["type"] == MessageType.IMAGE_REF.value
        assert msg["schema_version"] == SCHEMA_VERSION
        assert msg["track_id"] == 42
        assert msg["zone"] == "B3D"
        assert msg["url"] == "file:///var/lib/isi_monitor3d/snapshots/5000_B3D_42.jpg"
        # Crucially: no raw image bytes field
        assert "image" not in msg
        assert "image_bytes" not in msg
        assert "data" not in msg

        sink.close()


def test_close_is_idempotent() -> None:
    """Calling close() twice must not raise and must call loop_stop + disconnect."""
    with patch("backbone.comms.mqtt_sink.mqtt.Client") as MockClient:
        mock_instance = MagicMock()
        MockClient.return_value = mock_instance

        from backbone.comms.mqtt_sink import MqttSink
        sink = MqttSink(host="127.0.0.1", port=1883)
        sink.close()
        sink.close()  # second call must be a no-op

        # loop_stop and disconnect called exactly once (not twice)
        mock_instance.loop_stop.assert_called_once()
        mock_instance.disconnect.assert_called_once()


def test_close_disconnects_before_loop_stop() -> None:
    """close() must issue disconnect() BEFORE loop_stop() so the DISCONNECT
    packet is handed to a still-running loop (mirrors the gateway's stop())."""
    with patch("backbone.comms.mqtt_sink.mqtt.Client") as MockClient:
        mock_instance = MagicMock()
        MockClient.return_value = mock_instance

        from backbone.comms.mqtt_sink import MqttSink
        sink = MqttSink(host="127.0.0.1", port=1883)
        sink.close()

        # Inspect the shared mock_calls ledger and assert ordering: the
        # disconnect() call must appear before the loop_stop() call.
        names = [c[0] for c in mock_instance.mock_calls]
        assert "disconnect" in names and "loop_stop" in names
        assert names.index("disconnect") < names.index("loop_stop"), names


# ---------------------------------------------------------------------------
# Diagnostics heartbeat topic
# ---------------------------------------------------------------------------

def _make_diag() -> DiagnosticsMessage:
    return DiagnosticsMessage(
        ts=1_700_000_000.0,
        node_id="zone_a",
        mode="single_cam_homography",
        sources={"cam_a": "alive"},
        frame_count=0,
        fps=0.0,
        latency_ms=LatencyStats(),
        zones=0,
        subscriptions=0,
        calibration=CalibrationFactCheck(loaded=True, rms_ok=True, mode=1),
    )


def _make_config() -> ConfigMessage:
    return ConfigMessage(
        ts=1_700_000_000.0,
        node_id="zone_a",
        area="Zone A",
        mode="single_cam_homography",
        cameras=["cam_a"],
        zones=[],
        calibration=CalibrationFactCheck(loaded=True, rms_ok=True, mode=1),
    )


def test_publish_diagnostics_correct_topic_and_payload() -> None:
    """publish_diagnostics calls client.publish with the heartbeat topic and valid JSON."""
    with patch("backbone.comms.mqtt_sink.mqtt.Client") as MockClient:
        mock_instance = MagicMock()
        MockClient.return_value = mock_instance

        from backbone.comms.mqtt_sink import MqttSink
        sink = MqttSink(host="127.0.0.1", port=1883, prefix="isiMonitor3D/v1/zone_a")
        sink.publish_diagnostics(_make_diag())

        mock_instance.publish.assert_called_once()
        call_args = mock_instance.publish.call_args
        topic = call_args[0][0]
        payload_bytes = call_args[0][1]

        assert topic == "isiMonitor3D/v1/zone_a/diagnostics/heartbeat"
        msg = json.loads(payload_bytes.decode("utf-8"))
        assert msg["type"] == MessageType.DIAGNOSTICS.value
        assert msg["schema_version"] == SCHEMA_VERSION
        assert msg["node_id"] == "zone_a"
        sink.close()


def test_publish_diagnostics_custom_topic() -> None:
    """diag_topic parameter overrides the default heartbeat topic."""
    with patch("backbone.comms.mqtt_sink.mqtt.Client") as MockClient:
        mock_instance = MagicMock()
        MockClient.return_value = mock_instance

        from backbone.comms.mqtt_sink import MqttSink
        sink = MqttSink(
            host="127.0.0.1", port=1883, prefix="isi/z",
            diag_topic="{prefix}/hb",
        )
        sink.publish_diagnostics(_make_diag())

        call_args = mock_instance.publish.call_args
        assert call_args[0][0] == "isi/z/hb"
        sink.close()


# ---------------------------------------------------------------------------
# Config retained advertisement
# ---------------------------------------------------------------------------

def test_publish_config_uses_retain_true() -> None:
    """publish_config MUST call client.publish with retain=True unconditionally."""
    with patch("backbone.comms.mqtt_sink.mqtt.Client") as MockClient:
        mock_instance = MagicMock()
        MockClient.return_value = mock_instance

        from backbone.comms.mqtt_sink import MqttSink
        # Even when instance retain=False, config must be retained.
        sink = MqttSink(host="127.0.0.1", port=1883, prefix="isiMonitor3D/v1/zone_a", retain=False)
        sink.publish_config(_make_config())

        mock_instance.publish.assert_called_once()
        call_args = mock_instance.publish.call_args
        topic = call_args[0][0]
        payload_bytes = call_args[0][1]
        kwargs = call_args[1]

        assert topic == "isiMonitor3D/v1/zone_a/config"
        assert kwargs.get("retain") is True, "retain must be True for config messages"
        msg = json.loads(payload_bytes.decode("utf-8"))
        assert msg["type"] == MessageType.CONFIG.value
        assert msg["node_id"] == "zone_a"
        assert msg["area"] == "Zone A"
        sink.close()


def test_publish_config_custom_topic() -> None:
    """config_topic parameter overrides the default config topic."""
    with patch("backbone.comms.mqtt_sink.mqtt.Client") as MockClient:
        mock_instance = MagicMock()
        MockClient.return_value = mock_instance

        from backbone.comms.mqtt_sink import MqttSink
        sink = MqttSink(
            host="127.0.0.1", port=1883, prefix="isi/z",
            config_topic="{prefix}/node_config",
        )
        sink.publish_config(_make_config())

        call_args = mock_instance.publish.call_args
        assert call_args[0][0] == "isi/z/node_config"
        assert call_args[1].get("retain") is True
        sink.close()


def test_publish_config_retain_true_even_when_instance_retain_is_false() -> None:
    """Force-retain test: instance retain=False must not bleed into config publish."""
    with patch("backbone.comms.mqtt_sink.mqtt.Client") as MockClient:
        mock_instance = MagicMock()
        MockClient.return_value = mock_instance

        from backbone.comms.mqtt_sink import MqttSink
        sink = MqttSink(host="127.0.0.1", port=1883, retain=False)
        sink.publish_config(_make_config())

        call_args = mock_instance.publish.call_args
        assert call_args[1].get("retain") is True
        sink.close()


def test_config_advert_republished_on_connect() -> None:
    """The retained config advert must be re-published from on_connect.

    publish_config() at orchestrator startup races the async CONNACK; a QoS-0
    publish issued before the socket connects is dropped. The sink caches the
    advert and re-emits it when on_connect fires (rc=0), so a late-joining
    gateway always sees it (and it survives a broker restart that wipes
    retained state)."""
    with patch("backbone.comms.mqtt_sink.mqtt.Client") as MockClient:
        mock_instance = MagicMock()
        MockClient.return_value = mock_instance

        from backbone.comms.mqtt_sink import MqttSink
        sink = MqttSink(host="127.0.0.1", port=1883, prefix="isiMonitor3D/v1/zone_a")
        sink.publish_config(_make_config())
        publishes_after_first = mock_instance.publish.call_count

        # Simulate the broker connection completing (CONNACK rc=0).
        sink._on_connect(mock_instance, None, {}, 0)

        # A second retained publish to the config topic must have happened.
        assert mock_instance.publish.call_count == publishes_after_first + 1
        topic, *_ = mock_instance.publish.call_args[0]
        assert topic == "isiMonitor3D/v1/zone_a/config"
        assert mock_instance.publish.call_args[1].get("retain") is True
        sink.close()


def test_on_connect_without_config_does_not_publish() -> None:
    """on_connect must not publish anything if no config advert was cached."""
    with patch("backbone.comms.mqtt_sink.mqtt.Client") as MockClient:
        mock_instance = MagicMock()
        MockClient.return_value = mock_instance

        from backbone.comms.mqtt_sink import MqttSink
        sink = MqttSink(host="127.0.0.1", port=1883)
        sink._on_connect(mock_instance, None, {}, 0)

        assert mock_instance.publish.call_count == 0
        sink.close()


# ---------------------------------------------------------------------------
# TLS — ca_cert / tls_insecure wiring
# ---------------------------------------------------------------------------

def test_tls_with_ca_cert_calls_tls_set_with_ca_certs() -> None:
    """tls=True + ca_cert path → client.tls_set(ca_certs=<path>)."""
    with patch("backbone.comms.mqtt_sink.mqtt.Client") as MockClient:
        mock_instance = MagicMock()
        MockClient.return_value = mock_instance

        from backbone.comms.mqtt_sink import MqttSink
        sink = MqttSink(host="127.0.0.1", port=8883, tls=True, ca_cert="/c/ca.crt")

        mock_instance.tls_set.assert_called_once_with(ca_certs="/c/ca.crt")
        mock_instance.tls_insecure_set.assert_not_called()
        sink.close()


def test_tls_insecure_calls_tls_insecure_set() -> None:
    """tls=True + tls_insecure=True → client.tls_insecure_set(True) is called."""
    with patch("backbone.comms.mqtt_sink.mqtt.Client") as MockClient:
        mock_instance = MagicMock()
        MockClient.return_value = mock_instance

        from backbone.comms.mqtt_sink import MqttSink
        sink = MqttSink(
            host="127.0.0.1", port=8883,
            tls=True, ca_cert="/c/ca.crt", tls_insecure=True,
        )

        mock_instance.tls_set.assert_called_once_with(ca_certs="/c/ca.crt")
        mock_instance.tls_insecure_set.assert_called_once_with(True)
        sink.close()


def test_tls_false_skips_tls_set_and_tls_insecure_set() -> None:
    """tls=False → neither tls_set nor tls_insecure_set is called."""
    with patch("backbone.comms.mqtt_sink.mqtt.Client") as MockClient:
        mock_instance = MagicMock()
        MockClient.return_value = mock_instance

        from backbone.comms.mqtt_sink import MqttSink
        sink = MqttSink(host="127.0.0.1", port=1883, tls=False, tls_insecure=True)

        mock_instance.tls_set.assert_not_called()
        mock_instance.tls_insecure_set.assert_not_called()
        sink.close()


def test_tls_with_no_ca_cert_uses_system_cas() -> None:
    """tls=True with ca_cert=None (default) → tls_set(ca_certs=None) for system CAs."""
    with patch("backbone.comms.mqtt_sink.mqtt.Client") as MockClient:
        mock_instance = MagicMock()
        MockClient.return_value = mock_instance

        from backbone.comms.mqtt_sink import MqttSink
        sink = MqttSink(host="127.0.0.1", port=8883, tls=True)

        mock_instance.tls_set.assert_called_once_with(ca_certs=None)
        mock_instance.tls_insecure_set.assert_not_called()
        sink.close()


# ---------------------------------------------------------------------------
# Zone state — the retained per-zone object list (the WMS/FMS signal)
# ---------------------------------------------------------------------------

def _make_zone_state():
    from backbone.comms.schemas import ZoneObject, ZoneStateMessage
    return ZoneStateMessage(
        ts=1_700_000_000.0,
        zone="B3D",
        objects=(
            ZoneObject(track_id=7, cls="palette", confidence=0.91, xy_m=(3.0, 4.0)),
        ),
        count=1,
    )


def test_publish_zone_state_topic_retained_qos1() -> None:
    """publish_zone_state publishes to {prefix}/zone/{zone} with retain=True, qos=1."""
    with patch("backbone.comms.mqtt_sink.mqtt.Client") as MockClient:
        mock_instance = MagicMock()
        MockClient.return_value = mock_instance

        from backbone.comms.mqtt_sink import MqttSink
        sink = MqttSink(host="127.0.0.1", port=1883, prefix="isiMonitor3D/v1/zone_a",
                        retain=False)
        sink.publish_zone_state(_make_zone_state())

        mock_instance.publish.assert_called_once()
        call_args = mock_instance.publish.call_args
        topic = call_args[0][0]
        payload_bytes = call_args[0][1]
        kwargs = call_args[1]

        assert topic == "isiMonitor3D/v1/zone_a/zone/B3D"
        assert kwargs.get("retain") is True, "zone state must be retained"
        assert kwargs.get("qos") == 1, "zone state defaults to QoS 1"
        msg = json.loads(payload_bytes.decode("utf-8"))
        assert msg["type"] == "zone_state"
        assert msg["schema_version"] == SCHEMA_VERSION
        assert msg["zone"] == "B3D"
        assert msg["count"] == 1
        assert msg["objects"][0]["track_id"] == 7
        assert msg["objects"][0]["confidence"] == 0.91
        sink.close()


def test_publish_zone_state_sanitises_zone_name() -> None:
    """Zone names with MQTT wildcard chars are sanitised in the state topic."""
    with patch("backbone.comms.mqtt_sink.mqtt.Client") as MockClient:
        mock_instance = MagicMock()
        MockClient.return_value = mock_instance

        from backbone.comms.mqtt_sink import MqttSink
        from backbone.comms.schemas import ZoneStateMessage
        sink = MqttSink(host="127.0.0.1", port=1883, prefix="isiMonitor3D/v1/node")
        sink.publish_zone_state(
            ZoneStateMessage(ts=1.0, zone="zone/A+B#C", objects=(), count=0)
        )
        topic = mock_instance.publish.call_args[0][0]
        assert topic == "isiMonitor3D/v1/node/zone/zone_A_B_C"
        sink.close()


def test_publish_zone_state_custom_topic_and_qos() -> None:
    """zone_state_topic and zone_state_qos parameters override the defaults."""
    with patch("backbone.comms.mqtt_sink.mqtt.Client") as MockClient:
        mock_instance = MagicMock()
        MockClient.return_value = mock_instance

        from backbone.comms.mqtt_sink import MqttSink
        sink = MqttSink(
            host="127.0.0.1", port=1883, prefix="isi/z",
            zone_state_topic="{prefix}/zstate/{zone}", zone_state_qos=0,
        )
        sink.publish_zone_state(_make_zone_state())

        call_args = mock_instance.publish.call_args
        assert call_args[0][0] == "isi/z/zstate/B3D"
        assert call_args[1].get("qos") == 0
        assert call_args[1].get("retain") is True
        sink.close()


def test_zone_state_qos_validation() -> None:
    """zone_state_qos outside {0, 1, 2} must raise ValueError."""
    with patch("backbone.comms.mqtt_sink.mqtt.Client"):
        from backbone.comms.mqtt_sink import MqttSink
        with pytest.raises(ValueError, match="zone_state_qos"):
            MqttSink(host="127.0.0.1", port=1883, zone_state_qos=3)


def test_six_zone_topics_per_node() -> None:
    """zone1-zone6 each publish retained on their own {prefix}/zone/<name> topic."""
    with patch("backbone.comms.mqtt_sink.mqtt.Client") as MockClient:
        mock_instance = MagicMock()
        MockClient.return_value = mock_instance

        from backbone.comms.mqtt_sink import MqttSink
        from backbone.comms.schemas import ZoneStateMessage
        sink = MqttSink(host="127.0.0.1", port=1883, prefix="isiMonitor3D/v1/zone_a")
        for i in range(1, 7):
            sink.publish_zone_state(
                ZoneStateMessage(ts=1.0, zone=f"zone{i}", objects=(), count=0)
            )

        topics = [c[0][0] for c in mock_instance.publish.call_args_list]
        assert topics == [f"isiMonitor3D/v1/zone_a/zone/zone{i}" for i in range(1, 7)]
        assert all(c[1].get("retain") is True for c in mock_instance.publish.call_args_list)
        sink.close()
