"""``MqttSink`` — emit ``Track2D`` / ``Track3D`` as MQTT messages.

Each track becomes one MQTT message containing the JSON-serialized
``Track2DMessage`` / ``Track3DMessage`` from ``schemas.py``. Topics are
per-class by default so subscribers can filter at the broker level without
parsing JSON; the ``track_id`` is embedded in the payload.

Designed as a drop-in companion to ``UdpSink``: a dead or unreachable broker
at startup or publish time is logged and swallowed — it must never raise into
the pipeline.

Uses paho-mqtt 2.1.0 with ``CallbackAPIVersion.VERSION1`` so the simple
``on_connect``/``on_disconnect`` callbacks remain valid without migration to the
v2 callback signatures.

Topic scheme (defaults)::

    {prefix}/track2d/{cls}   — Track2DMessage JSON
    {prefix}/track3d/{cls}   — Track3DMessage JSON

The templates are user-configurable; ``{prefix}`` and ``{cls}`` are the only
substitution tokens. ``track_id`` intentionally does NOT appear in the topic by
default: per-class fan-out at the broker keeps subscriber cardinality O(classes),
not O(objects).
"""

from __future__ import annotations

import logging

import paho.mqtt.client as mqtt

from backbone.core.interfaces import MetadataSink, metadata_sink_registry
from backbone.core.types import Track2D, Track3D

from .schemas import PassingEventMessage, Track2DMessage, Track3DMessage

logger = logging.getLogger(__name__)

_TOPIC_SANITIZE = str.maketrans({"/": "_", "+": "_", "#": "_"})


def _sanitize_cls(cls: str) -> str:
    """Replace MQTT wildcard / topic-separator characters with underscores."""
    return cls.translate(_TOPIC_SANITIZE)


@metadata_sink_registry.register("mqtt")
class MqttSink(MetadataSink):
    """MQTT publisher. One instance, one paho client, one background loop thread."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 1883,
        prefix: str = "isi/monitor3d",
        qos: int = 0,
        retain: bool = False,
        keepalive: int = 60,
        client_id: str = "",
        username: str | None = None,
        password: str | None = None,
        tls: bool = False,
        track2d_topic: str = "{prefix}/track2d/{cls}",
        track3d_topic: str = "{prefix}/track3d/{cls}",
        event_topic: str = "{prefix}/zones/{zone}/passings",
    ) -> None:
        """Initialise and start the MQTT client background thread.

        A broker that is unreachable at construction time is handled gracefully:
        ``connect_async`` queues the connection attempt; the background loop
        retries with exponential back-off (1-30 s) until the broker appears.

        Args:
            host: Broker hostname or IP address.
            port: Broker TCP port (must be in 1-65535).
            prefix: Topic namespace prepended to every topic template.
            qos: MQTT QoS level — 0 (at most once), 1 (at least once),
                 or 2 (exactly once).
            retain: Whether the broker should retain the last message per topic.
            keepalive: MQTT keep-alive interval in seconds.
            client_id: MQTT client identifier; empty string lets the broker
                       assign one.
            username: Optional broker username.
            password: Optional broker password (used only when username is set).
            tls: If True, wrap the transport with TLS using system CAs.
            track2d_topic: Topic template for ``Track2DMessage``; supports
                           ``{prefix}`` and ``{cls}`` tokens.
            track3d_topic: Topic template for ``Track3DMessage``; supports
                           ``{prefix}`` and ``{cls}`` tokens.
            event_topic:   Topic template for ``PassingEventMessage``; supports
                           ``{prefix}`` and ``{zone}`` tokens. The zone name is
                           sanitised (``/``, ``+``, ``#`` → ``_``) before
                           substitution so it is safe as a MQTT topic segment.

        Raises:
            ValueError: If ``port`` is outside (0, 65536) or ``qos`` is not
                        one of 0, 1, 2.
        """
        if not (0 < port < 65536):
            raise ValueError(f"port must be in (0, 65535], got {port}")
        if qos not in (0, 1, 2):
            raise ValueError(f"qos must be 0, 1, or 2, got {qos}")

        self._host = host
        self._port = int(port)
        self._prefix = prefix
        self._qos = qos
        self._retain = retain
        self._track2d_topic = track2d_topic
        self._track3d_topic = track3d_topic
        self._event_topic = event_topic
        self._closed = False

        self._client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION1,
            client_id=client_id,
        )

        if username:
            self._client.username_pw_set(username, password)

        if tls:
            self._client.tls_set()

        self._client.reconnect_delay_set(min_delay=1, max_delay=30)
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect

        try:
            self._client.connect_async(host, int(port), keepalive)
            self._client.loop_start()
        except Exception:
            logger.warning(
                "MqttSink: could not start connection to %s:%s — will retry in background",
                host,
                port,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # MetadataSink implementation
    # ------------------------------------------------------------------

    def publish_track_2d(self, track: Track2D) -> None:
        """Publish a ``Track2DMessage`` to the configured track2d topic."""
        msg = Track2DMessage.from_track(track)
        topic = self._track2d_topic.format(
            prefix=self._prefix,
            cls=_sanitize_cls(track.cls),
        )
        self._publish(topic, msg.model_dump_json().encode("utf-8"))

    def publish_track_3d(self, track: Track3D) -> None:
        """Publish a ``Track3DMessage`` to the configured track3d topic."""
        msg = Track3DMessage.from_track(track)
        topic = self._track3d_topic.format(
            prefix=self._prefix,
            cls=_sanitize_cls(track.cls),
        )
        self._publish(topic, msg.model_dump_json().encode("utf-8"))

    def publish_event(self, event: object) -> None:
        """Publish a ``PassingEventMessage`` to the configured event topic.

        The ``{zone}`` token in the topic template is populated with the
        sanitised zone name so that MQTT wildcard chars can't appear in topics.
        """
        msg = PassingEventMessage.from_event(event)
        topic = self._event_topic.format(
            prefix=self._prefix,
            zone=_sanitize_cls(msg.zone),   # reuse the same sanitiser
        )
        self._publish(topic, msg.model_dump_json().encode("utf-8"))

    def close(self) -> None:
        """Stop the MQTT loop and disconnect. Idempotent."""
        if self._closed:
            return
        self._closed = True
        try:
            self._client.loop_stop()
        except Exception:
            logger.warning("MqttSink.close: loop_stop failed", exc_info=True)
        try:
            self._client.disconnect()
        except Exception:
            logger.warning("MqttSink.close: disconnect failed", exc_info=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _publish(self, topic: str, payload: bytes) -> None:
        """Publish one payload; swallow any error so the pipeline never stalls."""
        try:
            self._client.publish(topic, payload, qos=self._qos, retain=self._retain)
        except Exception:
            logger.warning(
                "MqttSink._publish failed on topic %r", topic, exc_info=True
            )

    def _on_connect(self, client: mqtt.Client, userdata: object, flags: dict, rc: int) -> None:
        if rc == 0:
            logger.info("MqttSink: connected to %s:%s", self._host, self._port)
        else:
            logger.warning(
                "MqttSink: connection refused by %s:%s (rc=%s)",
                self._host,
                self._port,
                rc,
            )

    def _on_disconnect(self, client: mqtt.Client, userdata: object, rc: int) -> None:
        if rc == 0:
            logger.debug("MqttSink: clean disconnect from %s:%s", self._host, self._port)
        else:
            logger.warning(
                "MqttSink: unexpected disconnect from %s:%s (rc=%s) — reconnecting",
                self._host,
                self._port,
                rc,
            )
