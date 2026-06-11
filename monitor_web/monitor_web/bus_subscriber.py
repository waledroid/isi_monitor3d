"""UDP listener — typed-parse Backbone envelopes into an in-process state.

Runs in a daemon thread. Each received packet is decoded via
``backbone.metadata.schemas.parse_envelope`` (typed pydantic models) and
folded into a small thread-safe state:

* ``last_envelope_ts`` — wall clock of the most recent envelope (drives
  the dashboard's freshness indicator).
* ``last_tracks`` — most recent set of ``Track2D`` per ``track_id``, used by
  the floor-map renderer when a freshly-connected WebSocket client subscribes.
* ``ws_broadcast_queue`` — every incoming envelope goes here too so the
  ``/ws/tracks`` route can fan it out to live clients.

Malformed packets, version-mismatched envelopes, and unknown JSON shapes are
logged and dropped — the listener never crashes on bad UDP input.
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass, field

from backbone.metadata.schemas import (
    SchemaVersionError,
    Track2DMessage,
    Track3DMessage,
    parse_envelope,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BusState:
    """Snapshot of the most recent UDP traffic for dashboard rendering."""

    last_envelope_ts: float = 0.0   # time.time() of last packet (zero = never)
    last_track2d_by_id: dict[int, Track2DMessage] = field(default_factory=dict)
    last_track3d_by_id: dict[int, Track3DMessage] = field(default_factory=dict)
    received: int = 0
    dropped_malformed: int = 0
    dropped_version: int = 0
    # Live capture→receive latency over a rolling window (KPI: p95 < 200 ms). None
    # until messages with a capture timestamp arrive. This is the dashboard-side
    # proxy for capture→publish — on loopback the network term is negligible.
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    latency_samples: int = 0


class BusSubscriber:
    """UDP socket listener + broadcast queue for the dashboard."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        broadcast_queue: asyncio.Queue | None = None,
    ) -> None:
        self._host = host
        self._port = int(port)
        self._broadcast_queue = broadcast_queue
        self._loop: asyncio.AbstractEventLoop | None = None

        self._state = BusState()
        self._latencies: deque[float] = deque(maxlen=300)   # capture→receive ms window
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._sock: socket.socket | None = None

    # ---- lifecycle ----

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Bind to the FastAPI event loop so we can schedule broadcasts from threads."""
        self._loop = loop

    def start(self) -> None:
        if self._thread is not None:
            return
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self._host, self._port))
        self._sock.settimeout(0.5)
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="bus-subscriber",
        )
        self._thread.start()
        logger.info("bus_subscriber: listening on %s:%d", self._host, self._port)

    def stop(self) -> None:
        self._stop_event.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # ---- state accessors ----

    @staticmethod
    def _pct(sorted_vals: list[float], p: float) -> float | None:
        if not sorted_vals:
            return None
        k = min(len(sorted_vals) - 1, round((p / 100.0) * (len(sorted_vals) - 1)))
        return round(sorted_vals[k], 1)

    def snapshot(self) -> BusState:
        with self._lock:
            lats = sorted(self._latencies)
            return BusState(
                last_envelope_ts=self._state.last_envelope_ts,
                last_track2d_by_id=dict(self._state.last_track2d_by_id),
                last_track3d_by_id=dict(self._state.last_track3d_by_id),
                received=self._state.received,
                dropped_malformed=self._state.dropped_malformed,
                dropped_version=self._state.dropped_version,
                latency_p50_ms=self._pct(lats, 50),
                latency_p95_ms=self._pct(lats, 95),
                latency_samples=len(lats),
            )

    def is_fresh(self, threshold_s: float) -> bool:
        """True iff at least one envelope arrived within ``threshold_s`` seconds."""
        last = self._state.last_envelope_ts
        if last == 0.0:
            return False
        return (time.time() - last) <= threshold_s

    # ---- internals ----

    def _run(self) -> None:
        assert self._sock is not None
        while not self._stop_event.is_set():
            try:
                payload, _addr = self._sock.recvfrom(8192)
            except TimeoutError:
                continue
            except OSError as exc:
                if self._stop_event.is_set():
                    return
                logger.warning("bus_subscriber: recv error: %s", exc)
                continue
            self._handle_payload(payload)

    def _handle_payload(self, payload: bytes) -> None:
        try:
            data = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            with self._lock:
                self._state.dropped_malformed += 1
            return
        try:
            msg = parse_envelope(data)
        except SchemaVersionError as exc:
            with self._lock:
                self._state.dropped_version += 1
            logger.warning("bus_subscriber: %s", exc)
            return
        except Exception as exc:
            with self._lock:
                self._state.dropped_malformed += 1
            logger.debug("bus_subscriber: bad envelope (%s): %s", type(exc).__name__, exc)
            return

        now = time.time()
        ts = getattr(msg, "ts", None)
        with self._lock:
            self._state.received += 1
            self._state.last_envelope_ts = now
            if ts is not None:
                self._latencies.append(max(0.0, (now - float(ts)) * 1000.0))
            if isinstance(msg, Track2DMessage):
                self._state.last_track2d_by_id[msg.track_id] = msg
            elif isinstance(msg, Track3DMessage):
                self._state.last_track3d_by_id[msg.track_id] = msg

        # Broadcast to live WebSocket clients via the event loop, if attached.
        if self._broadcast_queue is not None and self._loop is not None:
            try:
                self._loop.call_soon_threadsafe(
                    self._broadcast_queue.put_nowait, msg,
                )
            except (RuntimeError, asyncio.QueueFull):
                pass
