"""MQTT consumer + per-node state cache.

Subscribes to ``{base}/#`` on a central broker. Each arriving message:

1. Derives ``node_id`` from the second topic segment.
2. Parses the payload via ``backbone.comms.schemas.parse_envelope``.
3. Routes into a per-node ``NodeState`` cache (thread-safe).

``update_from_message`` is intentionally split out so tests can feed
messages directly without a live broker.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import paho.mqtt.client as mqtt
from backbone.comms.schemas import (
    ConfigMessage,
    DiagnosticsMessage,
    ImageRefMessage,
    PassingEventMessage,
    SchemaVersionError,
    Track2DMessage,
    Track3DMessage,
    ZoneStateMessage,
    parse_envelope,
)

logger = logging.getLogger(__name__)

# Matches a topic version segment like "v1", "v2" — i.e. the optional segment
# between <base> and <node_id> in <base>/<version>/<node_id>/<suffix>.
_VERSION_RE = re.compile(r"^v\d+$")


@dataclass
class NodeState:
    """All cached state for a single Backbone node."""

    last_track2d_by_id: dict[int, Track2DMessage] = field(default_factory=dict)
    last_track3d_by_id: dict[int, Track3DMessage] = field(default_factory=dict)
    passings: deque = field(default_factory=deque)   # maxlen set at init
    zone_state_by_zone: dict[str, ZoneStateMessage] = field(default_factory=dict)
    last_diagnostics: DiagnosticsMessage | None = None
    config: ConfigMessage | None = None
    last_seen: float = 0.0   # time.time() at last update
    topic_version: str = "v1"  # topic-contract version this node publishes on


@dataclass
class _Stats:
    received: int = 0
    dropped_malformed: int = 0
    dropped_version: int = 0


class MqttSubscriber:
    """Paho-based MQTT consumer that aggregates per-node state.

    Design notes:
    - ``connect_async`` + ``loop_start`` so a broker that is down at startup
      triggers background reconnects rather than a hard failure at start().
    - ``reconnect_delay_set`` caps the back-off so a recovered broker is
      re-joined within a few seconds.
    - All state reads/writes are guarded by ``_lock``.
    - ``update_from_message`` has no I/O dependency — tests call it directly.
    """

    def __init__(
        self,
        host: str,
        port: int,
        base: str,
        *,
        tls: bool = False,
        ca_cert: str | None = None,
        tls_insecure: bool = False,
        username: str | None = None,
        password: str | None = None,
        passings_buffer: int = 200,
        recent_buffer: int = 300,
    ) -> None:
        self._host = host
        self._port = port
        self._base = base
        self._tls = tls
        self._ca_cert = ca_cert
        self._tls_insecure = tls_insecure
        self._username = username
        self._password = password
        self._passings_buffer = passings_buffer

        self._nodes: dict[str, NodeState] = {}
        self._lock = threading.Lock()
        self._stats = _Stats()
        # Raw tail for the /ui probe: EVERY arriving message (topic + payload,
        # malformed included — that is the point of a probe), newest last.
        self._recent: deque = deque(maxlen=max(1, int(recent_buffer)))
        # Latest message per topic — the /ui schema tree. Unlike the ring, a
        # topic seen once (retained config adverts) stays visible forever.
        # Safety-capped; in practice the topic set is small and bounded.
        self._latest_by_topic: dict[str, dict] = {}
        self._topics_cap = 1000

        self._client: mqtt.Client | None = None
        self._started = False

    # ---- lifecycle --------------------------------------------------------

    def start(self) -> None:
        """Connect to the broker in the background.  Safe to call multiple times."""
        if self._started:
            return
        self._started = True

        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
        if self._username is not None:
            client.username_pw_set(self._username, self._password)
        if self._tls:
            client.tls_set(ca_certs=self._ca_cert)
            if self._tls_insecure:
                client.tls_insecure_set(True)
        client.reconnect_delay_set(min_delay=1, max_delay=10)
        client.on_connect = self._on_connect
        client.on_message = self._on_message

        try:
            client.connect_async(self._host, self._port)
        except OSError as exc:
            # Broker unreachable at startup — paho will retry in the bg loop.
            logger.warning(
                "mqtt_subscriber: initial connect failed (%s) — will retry", exc
            )
        client.loop_start()
        self._client = client
        logger.info(
            "mqtt_subscriber: connecting to %s:%d base=%r",
            self._host, self._port, self._base,
        )

    def stop(self) -> None:
        """Disconnect and stop the background loop.  Idempotent."""
        if self._client is None:
            return
        try:
            # disconnect() first wakes the network loop out of its reconnect
            # backoff so loop_stop()'s thread-join returns promptly (otherwise it
            # can block for the full reconnect delay when the broker is down).
            self._client.disconnect()
            self._client.loop_stop()
        except Exception:
            pass
        self._client = None
        self._started = False

    # ---- MQTT callbacks ---------------------------------------------------

    def _on_connect(
        self,
        client: mqtt.Client,
        userdata: Any,
        flags: dict,
        rc: int,
    ) -> None:
        if rc == 0:
            topic = f"{self._base}/#"
            client.subscribe(topic)
            logger.info("mqtt_subscriber: subscribed to %s", topic)
        else:
            logger.warning("mqtt_subscriber: connect result code %d — will retry", rc)

    def _parse_topic(self, topic: str) -> tuple[str, str] | None:
        """Derive ``(node_id, topic_version)`` from a topic, or ``None`` if malformed.

        Two layouts are accepted so the gateway works across the version
        transition:

        * versioned ``<base>/v<N>/<node_id>/...`` → ``(node_id, "v<N>")``
        * legacy   ``<base>/<node_id>/...``        → ``(node_id, "v0")``
        """
        parts = topic.split("/")
        if len(parts) < 2 or parts[0] != self._base:
            return None
        if _VERSION_RE.match(parts[1]):
            # versioned: <base>/<version>/<node_id>/...
            if len(parts) < 3:
                return None
            return parts[2], parts[1]
        # legacy unversioned: <base>/<node_id>/...
        return parts[1], "v0"

    def _on_message(
        self,
        client: mqtt.Client,
        userdata: Any,
        m: mqtt.MQTTMessage,
    ) -> None:
        topic: str = m.topic
        with self._lock:
            entry = {
                "ts": time.time(),
                "topic": topic,
                "payload": m.payload.decode("utf-8", errors="replace"),
                "bytes": len(m.payload),
            }
            self._recent.append(entry)
            known = self._latest_by_topic.get(topic)
            if known is None and len(self._latest_by_topic) >= self._topics_cap:
                oldest = min(self._latest_by_topic, key=lambda t: self._latest_by_topic[t]["ts"])
                del self._latest_by_topic[oldest]
            self._latest_by_topic[topic] = {
                **entry, "count": (known["count"] + 1) if known else 1}
        parsed = self._parse_topic(topic)
        if parsed is None:
            with self._lock:
                self._stats.dropped_malformed += 1
            logger.debug("mqtt_subscriber: malformed topic %r", topic)
            return

        node_id, topic_version = parsed

        try:
            payload_text = m.payload.decode("utf-8")
            data = json.loads(payload_text)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            with self._lock:
                self._stats.dropped_malformed += 1
            logger.debug("mqtt_subscriber: bad payload on %r: %s", topic, exc)
            return

        try:
            msg = parse_envelope(data)
        except SchemaVersionError as exc:
            with self._lock:
                self._stats.dropped_version += 1
            logger.warning("mqtt_subscriber: version error on %r: %s", topic, exc)
            return
        except Exception as exc:
            with self._lock:
                self._stats.dropped_malformed += 1
            logger.debug(
                "mqtt_subscriber: parse error on %r (%s): %s",
                topic, type(exc).__name__, exc,
            )
            return

        self.update_from_message(node_id, msg, topic_version=topic_version)

    # ---- state mutation (no broker dependency — testable directly) --------

    def update_from_message(
        self,
        node_id: str,
        msg: (
            Track2DMessage
            | Track3DMessage
            | PassingEventMessage
            | ImageRefMessage
            | ZoneStateMessage
            | DiagnosticsMessage
            | ConfigMessage
        ),
        topic_version: str = "v1",
    ) -> None:
        """Fold one parsed message into the per-node cache.

        Thread-safe.  Called from the paho network thread or directly from tests.
        ``topic_version`` is the topic-contract version derived from the MQTT
        topic ("v1" for the current versioned layout, "v0" for legacy
        unversioned topics).
        """
        with self._lock:
            self._stats.received += 1
            node = self._nodes.get(node_id)
            if node is None:
                node = NodeState(
                    passings=deque(maxlen=self._passings_buffer),
                )
                self._nodes[node_id] = node
            node.last_seen = time.time()
            node.topic_version = topic_version

            if isinstance(msg, Track2DMessage):
                node.last_track2d_by_id[msg.track_id] = msg
            elif isinstance(msg, Track3DMessage):
                node.last_track3d_by_id[msg.track_id] = msg
            elif isinstance(msg, PassingEventMessage):
                node.passings.append(msg)
            elif isinstance(msg, ZoneStateMessage):
                node.zone_state_by_zone[msg.zone] = msg
            elif isinstance(msg, DiagnosticsMessage):
                node.last_diagnostics = msg
            elif isinstance(msg, ConfigMessage):
                node.config = msg
            # ImageRefMessage: just bump last_seen (already done above).

    # ---- state reads (all locked, return copies) -------------------------

    def snapshot_nodes(self) -> dict[str, NodeState]:
        """Return a copy of the node dict whose mutable containers are detached.

        All copies are taken under the lock so callers (route handlers, test
        code) can iterate ``last_track2d_by_id``, ``last_track3d_by_id``, and
        ``passings`` without racing against the MQTT network thread.

        Immutable fields (``last_diagnostics``, ``config``) are frozen pydantic
        models — safe to share without copying.  ``last_seen`` is a plain float.
        ``passings`` is returned as a ``list`` (a deque snapshot); routes that
        iterate it are unaffected because they only use ``for … in``.
        """
        with self._lock:
            return {
                node_id: NodeState(
                    last_track2d_by_id=dict(node.last_track2d_by_id),
                    last_track3d_by_id=dict(node.last_track3d_by_id),
                    passings=deque(node.passings),
                    zone_state_by_zone=dict(node.zone_state_by_zone),
                    last_diagnostics=node.last_diagnostics,
                    config=node.config,
                    last_seen=node.last_seen,
                    topic_version=node.topic_version,
                )
                for node_id, node in self._nodes.items()
            }

    def node_alive(self, node_id: str, now: float, stale_after: float) -> bool:
        """True iff the node has been seen within ``stale_after`` seconds."""
        with self._lock:
            node = self._nodes.get(node_id)
        if node is None:
            return False
        return (now - node.last_seen) <= stale_after

    def topics(self) -> dict[str, dict]:
        """Latest message per topic (+ per-topic count) — the schema tree."""
        with self._lock:
            return {t: dict(v) for t, v in self._latest_by_topic.items()}

    def recent(self, limit: int = 100) -> list[dict]:
        """The last ``limit`` raw messages (topic/ts/payload), newest last."""
        with self._lock:
            msgs = list(self._recent)
        return msgs[-max(1, int(limit)):]

    def stats(self) -> dict[str, int]:
        """Return a copy of the ingestion counters."""
        with self._lock:
            return {
                "received": self._stats.received,
                "dropped_malformed": self._stats.dropped_malformed,
                "dropped_version": self._stats.dropped_version,
            }
