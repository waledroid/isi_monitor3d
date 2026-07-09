"""``UdpSink`` — emit ``Track2D`` / ``Track3D`` as UDP/JSON datagrams.

Each track becomes one UDP packet containing the JSON-serialized
``Track2DMessage`` / ``Track3DMessage`` from ``schemas.py``. UDP is intentional
— industrial bus consumers (Sécurité, Palettes, Dashboard, PLC gateway) all
poll their inbox at their own cadence; we don't want a slow consumer to
back-pressure the Backbone.

Unicast only in v1. If multicast becomes a real deployment need, register a
sibling ``udp_multicast`` plugin rather than overloading this one — the
extra socket options + TTL plumbing don't belong in the common path.

This is the *only* code path that touches a socket on the publish side.
"""

from __future__ import annotations

import json
import logging
import socket
import uuid

from backbone.core.interfaces import MetadataSink, metadata_sink_registry
from backbone.core.types import Track2D, Track3D

from .schemas import (
    SCHEMA_VERSION,
    ConfigMessage,
    DiagnosticsMessage,
    ImageRefMessage,
    ObservationsMessage,
    PassingEventMessage,
    ProximityMessage,
    Track2DMessage,
    Track3DMessage,
    ZoneStateMessage,
)

logger = logging.getLogger(__name__)

# Application-layer fragmentation. Datagrams above the path MTU get
# IP-fragmented, and some network layers silently drop the fragments — WSL2
# ``networkingMode=mirrored`` drops EVERY loopback UDP datagram over ~1.5 KB,
# which is exactly the size of an observations message carrying mask polygons.
# Payloads above _FRAGMENT_ABOVE are sliced into _CHUNK_CHARS-sized pieces,
# each wrapped in a ``FragmentMessage`` (consumers reassemble with
# ``FragmentBuffer``). Both bounds keep every fragment datagram comfortably
# under a 1500-byte MTU after the JSON envelope + string escaping.
_FRAGMENT_ABOVE = 1300
_CHUNK_CHARS = 1100


def send_json_datagram(sock: socket.socket, addr: tuple[str, int],
                       payload: bytes) -> None:
    """Send one JSON payload as UDP, fragmenting above ``_FRAGMENT_ABOVE``.

    The single shared implementation of the application-layer fragmentation
    contract — used by ``UdpSink`` (Backbone → bus) and by perception
    producers (dashboard → Backbone ingest, Direction 1) so both directions
    stay wire-compatible with ``FragmentBuffer``. Raises ``OSError`` on send
    failure (callers decide how loudly to log).
    """
    if len(payload) <= _FRAGMENT_ABOVE:
        sock.sendto(payload, addr)
        return
    text = payload.decode("utf-8")
    fid = uuid.uuid4().hex[:12]
    chunks = [text[i:i + _CHUNK_CHARS]
              for i in range(0, len(text), _CHUNK_CHARS)]
    for i, chunk in enumerate(chunks):
        frag = json.dumps(
            {"schema_version": SCHEMA_VERSION, "type": "fragment",
             "fid": fid, "i": i, "n": len(chunks), "data": chunk},
            separators=(",", ":"))
        sock.sendto(frag.encode("utf-8"), addr)


@metadata_sink_registry.register("udp")
class UdpSink(MetadataSink):
    """UDP/JSON publisher. One process, one sink instance, one socket."""

    def __init__(self, host: str = "127.0.0.1", port: int = 50001) -> None:
        if not (0 < port < 65536):
            raise ValueError(f"port must be in (0, 65535], got {port}")
        self._addr = (host, int(port))
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    @property
    def address(self) -> tuple[str, int]:
        return self._addr

    def publish_track_2d(self, track: Track2D) -> None:
        msg = Track2DMessage.from_track(track)
        self._send(msg.model_dump_json().encode("utf-8"))

    def publish_track_3d(self, track: Track3D) -> None:
        msg = Track3DMessage.from_track(track)
        self._send(msg.model_dump_json().encode("utf-8"))

    def publish_event(self, event: object) -> None:
        msg = PassingEventMessage.from_event(event)
        self._send(msg.model_dump_json().encode("utf-8"))

    def publish_image_ref(
        self,
        track_id: int,
        cls: str,
        zone: str,
        ts: float,
        url: str,
    ) -> None:
        msg = ImageRefMessage(track_id=track_id, cls=cls, zone=zone, ts=ts, url=url)
        self._send(msg.model_dump_json().encode("utf-8"))

    def publish_zone_state(self, msg: object) -> None:
        assert isinstance(msg, ZoneStateMessage)
        self._send(msg.model_dump_json().encode("utf-8"))

    def publish_proximity(self, msg: object) -> None:
        assert isinstance(msg, ProximityMessage)
        self._send(msg.model_dump_json().encode("utf-8"))

    def publish_observations(self, msg: object) -> None:
        assert isinstance(msg, ObservationsMessage)
        self._send(msg.model_dump_json().encode("utf-8"))

    def publish_diagnostics(self, msg: object) -> None:
        assert isinstance(msg, DiagnosticsMessage)
        self._send(msg.model_dump_json().encode("utf-8"))

    def publish_config(self, msg: object) -> None:
        assert isinstance(msg, ConfigMessage)
        self._send(msg.model_dump_json().encode("utf-8"))

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            logger.warning("UdpSink.close: socket already closed", exc_info=True)

    def _send(self, payload: bytes) -> None:
        try:
            send_json_datagram(self._sock, self._addr, payload)
        except OSError:
            logger.warning("UdpSink.sendto failed", exc_info=True)
