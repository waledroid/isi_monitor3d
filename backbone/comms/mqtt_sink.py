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

    {prefix}/track2d/{cls}              — Track2DMessage JSON
    {prefix}/track3d/{cls}              — Track3DMessage JSON
    {prefix}/zone/{zone}                — ZoneStateMessage JSON (retained, QoS 1)
    {prefix}/zone/{zone}/passings       — PassingEventMessage JSON
    {prefix}/zone/{zone}/images/{id}    — ImageRefMessage JSON
    {prefix}/diagnostics/heartbeat      — DiagnosticsMessage JSON
    {prefix}/config                     — ConfigMessage JSON (retained)

The templates are user-configurable; ``{prefix}`` and ``{cls}`` are the only
substitution tokens. ``track_id`` intentionally does NOT appear in the topic by
default: per-class fan-out at the broker keeps subscriber cardinality O(classes),
not O(objects).

**Zone topic segment (``{zone}``).** ``topic_zone`` selects what fills the
``{zone}`` slot on zone-state / passing / image topics: ``"id"`` (default) uses
the STABLE zone id so a rename never moves the topic (nor orphans retained
zone-state), ``"name"`` restores the legacy operator-label segment. Rollback for
existing name-keyed subscribers: set ``metadata.mqtt_topic_zone: name`` and
restart. Either way the id AND name are both inside the JSON payload.

**Retained-topic hygiene.** Zone-state (and, with ``retain: true``, image-ref)
topics are retained, so a zone deleted from config would leave a ghost retained
message on the broker forever. ``advertise_zones`` — called once at startup by
the orchestrator via the ``Publisher`` — reconciles ``{prefix}/zone/#`` against
the active zone set and clears every stale retained topic with a retained-empty
publish. Strictly scoped to this sink's own prefix.
"""

from __future__ import annotations

import logging
import threading

import paho.mqtt.client as mqtt

from backbone.core.interfaces import MetadataSink, metadata_sink_registry
from backbone.core.types import Track2D, Track3D

from .schemas import (
    ConfigMessage,
    DiagnosticsMessage,
    ImageRefMessage,
    PassingEventMessage,
    ProximityMessage,
    Track2DMessage,
    Track3DMessage,
    ZoneStateMessage,
)

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
        prefix: str = "isiMonitor3D/v1/node",
        qos: int = 0,
        retain: bool = False,
        keepalive: int = 60,
        client_id: str = "",
        username: str | None = None,
        password: str | None = None,
        tls: bool = False,
        ca_cert: str | None = None,
        tls_insecure: bool = False,
        track2d_topic: str = "{prefix}/track2d/{cls}",
        track3d_topic: str = "{prefix}/track3d/{cls}",
        event_topic: str = "{prefix}/zone/{zone}/passings",
        image_topic: str = "{prefix}/zone/{zone}/images/{track_id}",
        zone_state_topic: str = "{prefix}/zone/{zone}",
        zone_state_qos: int = 1,
        proximity_topic: str = "{prefix}/proximity",
        diag_topic: str = "{prefix}/diagnostics/heartbeat",
        config_topic: str = "{prefix}/config",
        topic_zone: str = "id",
        zone_reconcile_window_s: float = 1.5,
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
            tls: If True, wrap the transport with TLS.
            ca_cert: Path to a CA certificate file (PEM) used to verify the
                     broker's server certificate.  ``None`` (the default) falls
                     back to the system CA bundle — the same behaviour as before
                     this parameter existed.  Pass your own CA path when the
                     broker uses a self-signed certificate.
            tls_insecure: If True, skip hostname verification after ``tls_set``.
                          This is an escape hatch for development rigs and must
                          never be enabled in production.  Has no effect when
                          ``tls=False``.
            track2d_topic: Topic template for ``Track2DMessage``; supports
                           ``{prefix}`` and ``{cls}`` tokens.
            track3d_topic: Topic template for ``Track3DMessage``; supports
                           ``{prefix}`` and ``{cls}`` tokens.
            event_topic:   Topic template for ``PassingEventMessage``; supports
                           ``{prefix}`` and ``{zone}`` tokens. The zone name is
                           sanitised (``/``, ``+``, ``#`` → ``_``) before
                           substitution so it is safe as a MQTT topic segment.
            image_topic:   Topic template for ``ImageRefMessage``; supports
                           ``{prefix}``, ``{zone}``, and ``{track_id}`` tokens.
                           Zone is sanitised the same way as ``event_topic``.
                           Default: ``"{prefix}/zone/{zone}/images/{track_id}"``.
            zone_state_topic: Topic template for the retained ``ZoneStateMessage``
                           (one topic per zone — the WMS/FMS signal); supports
                           ``{prefix}`` and ``{zone}`` tokens (zone sanitised).
                           Always published with ``retain=True`` so late joiners
                           read every zone's current contents immediately.
                           Default: ``"{prefix}/zone/{zone}"``.
            zone_state_qos: QoS for zone-state publishes (default 1 — low-rate,
                           WMS-consequential; duplicates are harmless because
                           the payload is absolute state, not a delta).
            diag_topic:    Topic for ``DiagnosticsMessage`` heartbeats; supports
                           ``{prefix}``.  Published at the instance qos/retain.
                           Default: ``"{prefix}/diagnostics/heartbeat"``.
            config_topic:  Topic for the retained ``ConfigMessage`` advertisement;
                           supports ``{prefix}``.  Always published with
                           ``retain=True`` regardless of the instance ``retain``
                           flag.  Default: ``"{prefix}/config"``.
            topic_zone:    ``"id"`` (default) fills the ``{zone}`` topic segment
                           with the STABLE zone id so a rename never moves the
                           topic; ``"name"`` restores the legacy operator-label
                           segment (rollback for existing subscribers). The id
                           and name are always both in the JSON payload. When an
                           id is missing (legacy payload) it falls back to name.
            zone_reconcile_window_s: How long ``advertise_zones`` collects
                           retained messages under ``{prefix}/zone/#`` before
                           purging topics whose zone is no longer configured
                           (default 1.5 s; runs on the paho loop + a timer
                           thread, never blocking the pipeline).

        Raises:
            ValueError: If ``port`` is outside (0, 65536), ``qos`` is not one of
                        0, 1, 2, or ``topic_zone`` is not ``"id"``/``"name"``.
        """
        if not (0 < port < 65536):
            raise ValueError(f"port must be in (0, 65535], got {port}")
        if qos not in (0, 1, 2):
            raise ValueError(f"qos must be 0, 1, or 2, got {qos}")
        if zone_state_qos not in (0, 1, 2):
            raise ValueError(f"zone_state_qos must be 0, 1, or 2, got {zone_state_qos}")
        if topic_zone not in ("id", "name"):
            raise ValueError(f"topic_zone must be 'id' or 'name', got {topic_zone!r}")

        self._host = host
        self._port = int(port)
        self._prefix = prefix
        self._qos = qos
        self._retain = retain
        self._track2d_topic = track2d_topic
        self._track3d_topic = track3d_topic
        self._event_topic = event_topic
        self._image_topic = image_topic
        self._zone_state_topic = zone_state_topic
        self._zone_state_qos = zone_state_qos
        self._proximity_topic = proximity_topic
        self._diag_topic = diag_topic
        self._config_topic = config_topic
        self._topic_zone = topic_zone
        self._ca_cert = ca_cert
        self._tls_insecure = tls_insecure
        self._closed = False
        # Last retained config advert (topic, payload). Cached so it can be
        # re-published from ``_on_connect``: ``publish_config`` is called once
        # at orchestrator startup, which races the async CONNACK — a QoS-0
        # publish issued before the socket connects is silently dropped, so the
        # retained advert would never reach the broker. Re-publishing on every
        # (re)connect also restores the advert after a broker restart wipes
        # retained state.
        self._retained_config: tuple[str, bytes] | None = None

        # Retained zone-topic reconciliation state (see advertise_zones).
        # ``_zone_topic_head`` is the literal topic text preceding the {zone}
        # segment (e.g. "isi/node/zone/") — it scopes the reconcile strictly
        # to THIS sink's own prefix.
        head, _, _ = zone_state_topic.format(prefix=prefix, zone="\x00").partition("\x00")
        self._zone_topic_head = head
        self._zone_reconcile_window_s = float(zone_reconcile_window_s)
        self._active_zone_segments: frozenset[str] = frozenset()
        self._reconcile_pending = False
        self._reconcile_started = False
        self._reconcile_finished = False
        self._reconcile_topics: set[str] = set()
        self._reconcile_lock = threading.Lock()
        self._reconcile_timer: threading.Timer | None = None

        self._client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION1,
            client_id=client_id,
        )

        if username:
            self._client.username_pw_set(username, password)

        if tls:
            self._client.tls_set(ca_certs=ca_cert)
            if tls_insecure:
                self._client.tls_insecure_set(True)

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

    def _zone_segment(self, zone_name: str, zone_id: str) -> str:
        """Sanitised ``{zone}`` topic segment — id (stable) or name per config.

        Falls back to the name when ``topic_zone="id"`` but no id is present
        (a legacy payload), so a topic is always produced.
        """
        seg = zone_id if (self._topic_zone == "id" and zone_id) else zone_name
        return _sanitize_cls(seg)

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
            zone=self._zone_segment(msg.zone, msg.zone_id),
        )
        self._publish(topic, msg.model_dump_json().encode("utf-8"))

    def publish_image_ref(
        self,
        track_id: int,
        cls: str,
        zone: str,
        ts: float,
        url: str,
        zone_id: str = "",
    ) -> None:
        """Publish an ``ImageRefMessage`` (URL only, never raw bytes).

        The ``{zone}`` and ``{track_id}`` tokens in the topic template are
        populated; the zone segment follows ``topic_zone`` (id or name) and is
        sanitised so MQTT wildcard chars can't appear.
        """
        msg = ImageRefMessage(
            track_id=track_id, cls=cls, zone=zone, zone_id=zone_id, ts=ts, url=url
        )
        topic = self._image_topic.format(
            prefix=self._prefix,
            zone=self._zone_segment(zone, zone_id),
            track_id=track_id,
        )
        self._publish(topic, msg.model_dump_json().encode("utf-8"))

    def publish_zone_state(self, msg: object) -> None:
        """Publish a ``ZoneStateMessage`` retained on the per-zone topic.

        Retain is forced (like ``publish_config``) so the topic always holds
        the zone's *current* contents — a late-joining WMS/FMS reads every
        zone's occupancy immediately on subscribe. Published at
        ``zone_state_qos`` (default 1) rather than the instance ``qos``.
        """
        assert isinstance(msg, ZoneStateMessage)
        topic = self._zone_state_topic.format(
            prefix=self._prefix,
            zone=self._zone_segment(msg.zone, msg.zone_id),
        )
        payload = msg.model_dump_json().encode("utf-8")
        try:
            self._client.publish(topic, payload, qos=self._zone_state_qos, retain=True)
        except Exception:
            logger.warning(
                "MqttSink.publish_zone_state failed on topic %r", topic, exc_info=True
            )

    def publish_proximity(self, msg: object) -> None:
        """Publish a ``ProximityMessage`` retained on ``{prefix}/proximity``.

        Retained (like zone state): the topic always holds the CURRENT
        proximity picture — a late-joining safety consumer reads it on
        subscribe; the Backbone clears it with an explicit empty-``pairs``
        message. Published at ``zone_state_qos`` (same low-rate, state-full
        class of message).
        """
        assert isinstance(msg, ProximityMessage)
        topic = self._proximity_topic.format(prefix=self._prefix)
        payload = msg.model_dump_json().encode("utf-8")
        try:
            self._client.publish(topic, payload, qos=self._zone_state_qos, retain=True)
        except Exception:
            logger.warning(
                "MqttSink.publish_proximity failed on topic %r", topic, exc_info=True
            )

    def publish_diagnostics(self, msg: object) -> None:
        """Publish a ``DiagnosticsMessage`` to the diagnostics heartbeat topic."""
        assert isinstance(msg, DiagnosticsMessage)
        topic = self._diag_topic.format(prefix=self._prefix)
        self._publish(topic, msg.model_dump_json().encode("utf-8"))

    def publish_config(self, msg: object) -> None:
        """Publish a ``ConfigMessage`` with ``retain=True`` to the config topic.

        The retain flag is forced unconditionally here (overriding the instance
        ``retain`` setting) so that new subscribers always receive the most
        recent node config immediately on connection.
        """
        assert isinstance(msg, ConfigMessage)
        topic = self._config_topic.format(prefix=self._prefix)
        payload = msg.model_dump_json().encode("utf-8")
        # Cache for re-publish on (re)connect — see _on_connect.
        self._retained_config = (topic, payload)
        self._publish_retained(topic, payload)

    def advertise_zones(self, zones: list[tuple[str, str]]) -> None:
        """Reconcile retained zone topics against the ACTIVE ``(name, id)`` set.

        Zones only change across a Backbone restart, so a one-shot startup
        reconciliation suffices: subscribe once to ``{prefix}/zone/#``, collect
        retained messages for ``zone_reconcile_window_s``, then publish a
        retained-empty payload to every collected topic whose zone segment is
        not in the active set (clearing the broker's retained copy), and
        unsubscribe. An empty ``zones`` list purges ALL retained zone topics —
        a zoneless config must leave no ghost zone state behind.

        Non-blocking with respect to the pipeline: collection happens on the
        paho network loop, the purge on a daemon timer thread. If the broker
        is not yet connected, the reconcile is deferred to ``_on_connect``.
        """
        self._active_zone_segments = frozenset(
            self._zone_segment(name, zid) for name, zid in zones
        )
        self._reconcile_pending = True
        self._start_zone_reconcile()

    def close(self) -> None:
        """Disconnect then stop the MQTT loop. Idempotent.

        ``disconnect()`` is issued *before* ``loop_stop()`` so the DISCONNECT
        packet is handed to a still-running network loop and actually goes out
        on the wire (or is cleanly abandoned against a dead broker) before the
        loop thread is torn down. This mirrors
        ``isicomms.mqtt_subscriber.MqttSubscriber.stop``.
        """
        if self._closed:
            return
        self._closed = True
        if self._reconcile_timer is not None:
            self._reconcile_timer.cancel()
        try:
            self._client.disconnect()
        except Exception:
            logger.warning("MqttSink.close: disconnect failed", exc_info=True)
        try:
            self._client.loop_stop()
        except Exception:
            logger.warning("MqttSink.close: loop_stop failed", exc_info=True)

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

    def _publish_retained(self, topic: str, payload: bytes) -> None:
        """Publish with ``retain=True`` unconditionally (for config advertisements)."""
        try:
            self._client.publish(topic, payload, qos=self._qos, retain=True)
        except Exception:
            logger.warning(
                "MqttSink._publish_retained failed on topic %r", topic, exc_info=True
            )

    @property
    def _zone_wildcard(self) -> str:
        """The single subscription filter covering every zone subtopic."""
        return self._zone_topic_head + "#"

    def _start_zone_reconcile(self) -> None:
        """Subscribe to ``{prefix}/zone/#`` and arm the purge timer. One-shot.

        A subscribe issued before the CONNACK returns MQTT_ERR_NO_CONN and is
        not queued by paho, so a failed attempt stays *pending* and is retried
        from ``_on_connect``. Swallow-and-log like every other broker touch.
        """
        if not self._reconcile_pending or self._reconcile_started or self._closed:
            return
        if not self._zone_topic_head.endswith("/"):
            # A custom zone_state_topic where the {zone} token is not a whole
            # topic level can't be wildcard-matched — skip rather than guess.
            logger.warning(
                "MqttSink: zone_state_topic %r does not isolate {zone} as a "
                "topic level — skipping retained zone-topic reconcile",
                self._zone_state_topic,
            )
            self._reconcile_pending = False
            return
        try:
            self._client.on_message = self._on_reconcile_message
            rc = self._client.subscribe(self._zone_wildcard, qos=1)[0]
            if rc != mqtt.MQTT_ERR_SUCCESS:
                return  # not connected yet — _on_connect retries
            self._reconcile_started = True
            self._reconcile_timer = threading.Timer(
                self._zone_reconcile_window_s, self._finish_zone_reconcile
            )
            self._reconcile_timer.daemon = True
            self._reconcile_timer.start()
        except Exception:
            logger.warning(
                "MqttSink: retained zone-topic reconcile could not start", exc_info=True
            )

    def _on_reconcile_message(self, client: mqtt.Client, userdata: object, m: object) -> None:
        """Collect retained topics under our own zone root during the window."""
        try:
            if not getattr(m, "retain", False) or not m.payload:
                return  # live traffic or an already-cleared topic
            if not m.topic.startswith(self._zone_topic_head):
                return  # never touch other prefixes
            with self._reconcile_lock:
                self._reconcile_topics.add(m.topic)
        except Exception:
            logger.warning("MqttSink: reconcile message handling failed", exc_info=True)

    def _finish_zone_reconcile(self) -> None:
        """Purge collected topics whose zone segment is not active. Idempotent."""
        if self._reconcile_finished:
            return
        self._reconcile_finished = True
        self._reconcile_pending = False
        try:
            with self._reconcile_lock:
                collected = sorted(self._reconcile_topics)
            head_len = len(self._zone_topic_head)
            stale_topics: list[str] = []
            stale_zones: set[str] = set()
            for topic in collected:
                segment = topic[head_len:].split("/", 1)[0]
                if segment and segment not in self._active_zone_segments:
                    stale_topics.append(topic)
                    stale_zones.add(segment)
            for topic in stale_topics:
                # Retained-empty payload deletes the broker's retained copy.
                self._client.publish(topic, b"", qos=1, retain=True)
            self._client.unsubscribe(self._zone_wildcard)
            self._client.on_message = None
            if stale_topics:
                logger.info(
                    "MqttSink: purged %d stale retained zone topic(s): %s",
                    len(stale_topics),
                    ", ".join(sorted(stale_zones)),
                )
            else:
                logger.debug("MqttSink: no stale retained zone topics found")
        except Exception:
            logger.warning(
                "MqttSink: retained zone-topic reconcile failed", exc_info=True
            )

    def _on_connect(self, client: mqtt.Client, userdata: object, flags: dict, rc: int) -> None:
        if rc == 0:
            logger.info("MqttSink: connected to %s:%s", self._host, self._port)
            # Re-publish the retained config advert now that the socket is up.
            # The startup publish_config() raced the async CONNACK and may have
            # been dropped; this guarantees a late-joining gateway sees it.
            if self._retained_config is not None:
                topic, payload = self._retained_config
                self._publish_retained(topic, payload)
            # A reconcile requested before the CONNACK could not subscribe —
            # start it now (self-guarded one-shot).
            self._start_zone_reconcile()
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
