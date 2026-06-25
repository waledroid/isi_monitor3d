"""``MqttSink`` — hermetic tests; no real broker required.

All paho I/O is replaced by a ``MagicMock`` so tests run without a running
MQTT broker. Mirrors the structure of ``tests/test_udp_sink.py``.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from backbone.core.interfaces import metadata_sink_registry
from backbone.core.types import Track2D, Track3D
from backbone.metadata.schemas import (
    SCHEMA_VERSION,
    CalibrationFactCheck,
    ConfigMessage,
    DiagnosticsMessage,
    LatencyStats,
    MessageType,
)
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
    with patch("backbone.metadata.mqtt_sink.mqtt.Client") as MockClient:
        mock_instance = MagicMock()
        MockClient.return_value = mock_instance
        # Import here so the module is already loaded; patch targets the
        # already-imported name in mqtt_sink's namespace.
        from backbone.metadata.mqtt_sink import MqttSink
        sink = MqttSink(**kwargs)
        return sink, mock_instance


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_plugin_registered_under_mqtt() -> None:
    """``"mqtt"`` appears in the registry after importing backbone.metadata."""
    import backbone.metadata  # noqa: F401

    assert "mqtt" in metadata_sink_registry


def test_construction_via_registry_calls_loop_start() -> None:
    """Creating via the registry returns a MqttSink and starts the loop."""
    with patch("backbone.metadata.mqtt_sink.mqtt.Client") as MockClient:
        mock_instance = MagicMock()
        MockClient.return_value = mock_instance

        from backbone.metadata.mqtt_sink import MqttSink
        sink = metadata_sink_registry.create("mqtt", host="127.0.0.1", port=1883)
        assert isinstance(sink, MqttSink)
        mock_instance.loop_start.assert_called_once()
        sink.close()


def test_publish_track_2d_correct_topic_and_payload() -> None:
    """publish_track_2d calls client.publish with the per-class topic and valid JSON."""
    with patch("backbone.metadata.mqtt_sink.mqtt.Client") as MockClient:
        mock_instance = MagicMock()
        MockClient.return_value = mock_instance

        from backbone.metadata.mqtt_sink import MqttSink
        sink = MqttSink(host="127.0.0.1", port=1883, prefix="isi/monitor3d")
        sink.publish_track_2d(_t2())

        mock_instance.publish.assert_called_once()
        call_args = mock_instance.publish.call_args
        topic = call_args[0][0]
        payload_bytes = call_args[0][1]

        assert topic == "isi/monitor3d/track2d/palette"
        msg = json.loads(payload_bytes.decode("utf-8"))
        assert msg["schema_version"] == SCHEMA_VERSION
        assert msg["type"] == MessageType.TRACK_2D.value
        assert msg["track_id"] == 7
        assert msg["cls"] == "palette"

        sink.close()


def test_publish_track_3d_correct_topic_and_payload() -> None:
    """publish_track_3d calls client.publish with the per-class topic and xyz_m present."""
    with patch("backbone.metadata.mqtt_sink.mqtt.Client") as MockClient:
        mock_instance = MagicMock()
        MockClient.return_value = mock_instance

        from backbone.metadata.mqtt_sink import MqttSink
        sink = MqttSink(host="127.0.0.1", port=1883, prefix="isi/monitor3d")
        sink.publish_track_3d(_t3())

        mock_instance.publish.assert_called_once()
        call_args = mock_instance.publish.call_args
        topic = call_args[0][0]
        payload_bytes = call_args[0][1]

        assert topic == "isi/monitor3d/track3d/palette"
        msg = json.loads(payload_bytes.decode("utf-8"))
        assert msg["schema_version"] == SCHEMA_VERSION
        assert msg["type"] == MessageType.TRACK_3D.value
        assert "xyz_m" in msg
        assert msg["xyz_m"] == [3.0, 4.0, 1.5]

        sink.close()


def test_port_validation() -> None:
    """Port values 0 and 65536 must raise ValueError."""
    with patch("backbone.metadata.mqtt_sink.mqtt.Client"):
        from backbone.metadata.mqtt_sink import MqttSink

        with pytest.raises(ValueError, match="port"):
            MqttSink(host="127.0.0.1", port=0)
        with pytest.raises(ValueError, match="port"):
            MqttSink(host="127.0.0.1", port=65536)


def test_qos_validation() -> None:
    """QoS values outside {0, 1, 2} must raise ValueError."""
    with patch("backbone.metadata.mqtt_sink.mqtt.Client"):
        from backbone.metadata.mqtt_sink import MqttSink

        with pytest.raises(ValueError, match="qos"):
            MqttSink(host="127.0.0.1", port=1883, qos=3)
        with pytest.raises(ValueError, match="qos"):
            MqttSink(host="127.0.0.1", port=1883, qos=-1)


def test_publish_swallows_client_error() -> None:
    """A client.publish() exception must not propagate out of publish_track_2d."""
    with patch("backbone.metadata.mqtt_sink.mqtt.Client") as MockClient:
        mock_instance = MagicMock()
        mock_instance.publish.side_effect = RuntimeError("broker gone")
        MockClient.return_value = mock_instance

        from backbone.metadata.mqtt_sink import MqttSink
        sink = MqttSink(host="127.0.0.1", port=1883)
        # Must not raise, even though client.publish raises.
        sink.publish_track_2d(_t2())
        sink.close()


def test_publish_event_correct_topic_and_payload() -> None:
    """publish_event calls client.publish with the per-zone passings topic and valid JSON."""
    with patch("backbone.metadata.mqtt_sink.mqtt.Client") as MockClient:
        mock_instance = MagicMock()
        MockClient.return_value = mock_instance

        from backbone.metadata.mqtt_sink import MqttSink
        sink = MqttSink(host="127.0.0.1", port=1883, prefix="isi/monitor3d")
        ev = PassingEvent(track_id=3, cls="palette", zone="B3D", direction="enter", ts=5.0)
        sink.publish_event(ev)

        mock_instance.publish.assert_called_once()
        call_args = mock_instance.publish.call_args
        topic = call_args[0][0]
        payload_bytes = call_args[0][1]

        assert topic == "isi/monitor3d/zones/B3D/passings"
        msg = json.loads(payload_bytes.decode("utf-8"))
        assert msg["type"] == MessageType.PASSING.value
        assert msg["schema_version"] == SCHEMA_VERSION
        assert msg["zone"] == "B3D"
        assert msg["direction"] == "enter"
        assert msg["track_id"] == 3

        sink.close()


def test_publish_event_sanitises_zone_name() -> None:
    """Zone names containing MQTT wildcards are sanitised before topic formatting."""
    with patch("backbone.metadata.mqtt_sink.mqtt.Client") as MockClient:
        mock_instance = MagicMock()
        MockClient.return_value = mock_instance

        from backbone.metadata.mqtt_sink import MqttSink
        sink = MqttSink(host="127.0.0.1", port=1883, prefix="isi/monitor3d")
        ev = PassingEvent(track_id=1, cls="person", zone="zone/A+B#C", direction="leave", ts=1.0)
        sink.publish_event(ev)

        call_args = mock_instance.publish.call_args
        topic = call_args[0][0]
        assert topic == "isi/monitor3d/zones/zone_A_B_C/passings"
        sink.close()


def test_publish_image_ref_correct_topic_and_payload() -> None:
    """publish_image_ref sends an ImageRefMessage with the URL but no image bytes."""
    with patch("backbone.metadata.mqtt_sink.mqtt.Client") as MockClient:
        mock_instance = MagicMock()
        MockClient.return_value = mock_instance

        from backbone.metadata.mqtt_sink import MqttSink
        from backbone.metadata.schemas import MessageType
        sink = MqttSink(host="127.0.0.1", port=1883, prefix="isi/monitor3d")
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

        # Topic must embed zone + track_id
        assert topic == "isi/monitor3d/images/B3D/42"

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
    with patch("backbone.metadata.mqtt_sink.mqtt.Client") as MockClient:
        mock_instance = MagicMock()
        MockClient.return_value = mock_instance

        from backbone.metadata.mqtt_sink import MqttSink
        sink = MqttSink(host="127.0.0.1", port=1883)
        sink.close()
        sink.close()  # second call must be a no-op

        # loop_stop and disconnect called exactly once (not twice)
        mock_instance.loop_stop.assert_called_once()
        mock_instance.disconnect.assert_called_once()


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
    with patch("backbone.metadata.mqtt_sink.mqtt.Client") as MockClient:
        mock_instance = MagicMock()
        MockClient.return_value = mock_instance

        from backbone.metadata.mqtt_sink import MqttSink
        sink = MqttSink(host="127.0.0.1", port=1883, prefix="isi/zone_a")
        sink.publish_diagnostics(_make_diag())

        mock_instance.publish.assert_called_once()
        call_args = mock_instance.publish.call_args
        topic = call_args[0][0]
        payload_bytes = call_args[0][1]

        assert topic == "isi/zone_a/diagnostics/heartbeat"
        msg = json.loads(payload_bytes.decode("utf-8"))
        assert msg["type"] == MessageType.DIAGNOSTICS.value
        assert msg["schema_version"] == SCHEMA_VERSION
        assert msg["node_id"] == "zone_a"
        sink.close()


def test_publish_diagnostics_custom_topic() -> None:
    """diag_topic parameter overrides the default heartbeat topic."""
    with patch("backbone.metadata.mqtt_sink.mqtt.Client") as MockClient:
        mock_instance = MagicMock()
        MockClient.return_value = mock_instance

        from backbone.metadata.mqtt_sink import MqttSink
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
    with patch("backbone.metadata.mqtt_sink.mqtt.Client") as MockClient:
        mock_instance = MagicMock()
        MockClient.return_value = mock_instance

        from backbone.metadata.mqtt_sink import MqttSink
        # Even when instance retain=False, config must be retained.
        sink = MqttSink(host="127.0.0.1", port=1883, prefix="isi/zone_a", retain=False)
        sink.publish_config(_make_config())

        mock_instance.publish.assert_called_once()
        call_args = mock_instance.publish.call_args
        topic = call_args[0][0]
        payload_bytes = call_args[0][1]
        kwargs = call_args[1]

        assert topic == "isi/zone_a/config"
        assert kwargs.get("retain") is True, "retain must be True for config messages"
        msg = json.loads(payload_bytes.decode("utf-8"))
        assert msg["type"] == MessageType.CONFIG.value
        assert msg["node_id"] == "zone_a"
        assert msg["area"] == "Zone A"
        sink.close()


def test_publish_config_custom_topic() -> None:
    """config_topic parameter overrides the default config topic."""
    with patch("backbone.metadata.mqtt_sink.mqtt.Client") as MockClient:
        mock_instance = MagicMock()
        MockClient.return_value = mock_instance

        from backbone.metadata.mqtt_sink import MqttSink
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
    with patch("backbone.metadata.mqtt_sink.mqtt.Client") as MockClient:
        mock_instance = MagicMock()
        MockClient.return_value = mock_instance

        from backbone.metadata.mqtt_sink import MqttSink
        sink = MqttSink(host="127.0.0.1", port=1883, retain=False)
        sink.publish_config(_make_config())

        call_args = mock_instance.publish.call_args
        assert call_args[1].get("retain") is True
        sink.close()
