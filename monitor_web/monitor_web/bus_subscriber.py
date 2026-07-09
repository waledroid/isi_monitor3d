"""UDP listener — typed-parse Backbone envelopes into an in-process state.

Runs in a daemon thread. Each received packet is decoded via
``backbone.comms.schemas.parse_envelope`` (typed pydantic models) and
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

from backbone.comms.schemas import (
    DiagnosticsMessage,
    FragmentBuffer,
    FragmentMessage,
    ObservationsMessage,
    SchemaVersionError,
    Track2DMessage,
    Track3DMessage,
    ZoneStateMessage,
    parse_envelope,
)

logger = logging.getLogger(__name__)


def _offer_broadcast(q: asyncio.Queue, msg) -> None:
    """Runs ON the event loop: enqueue ``msg``, evicting the oldest entry when
    the queue is full (nobody draining /ws/tracks must never error-spam)."""
    try:
        q.put_nowait(msg)
    except asyncio.QueueFull:
        try:
            q.get_nowait()               # drop the oldest…
        except asyncio.QueueEmpty:
            pass
        try:
            q.put_nowait(msg)            # …keep the newest
        except asyncio.QueueFull:
            pass


@dataclass(slots=True)
class BusState:
    """Snapshot of the most recent UDP traffic for dashboard rendering."""

    last_envelope_ts: float = 0.0   # time.time() of last packet (zero = never)
    last_track2d_by_id: dict[int, Track2DMessage] = field(default_factory=dict)
    last_track3d_by_id: dict[int, Track3DMessage] = field(default_factory=dict)
    # Latest per-zone contents (ZoneStateMessage) — the COMMUNICATION panel's
    # zone cards read this when no gateway is configured (local UDP path).
    zone_state_by_zone: dict[str, ZoneStateMessage] = field(default_factory=dict)
    # Latest per-camera raw detections (ObservationsMessage) — the single-
    # perception feed the zone panels / cards / cam-view boxes render from.
    observations_by_camera: dict[str, ObservationsMessage] = field(default_factory=dict)
    received: int = 0
    dropped_malformed: int = 0
    dropped_version: int = 0
    # Live capture→receive latency over a rolling window (KPI: p95 < 200 ms). None
    # until messages with a capture timestamp arrive. This is the dashboard-side
    # proxy for capture→publish — on loopback the network term is negligible.
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    latency_samples: int = 0
    # From the Backbone's diagnostics heartbeat (5 s interval): per-camera
    # ingest fps + pipeline fps. ``diagnostics_ts`` (receive time) lets readers
    # ignore stale values after a STOP.
    fps_by_camera: dict[str, float] = field(default_factory=dict)
    pipeline_fps: float | None = None
    diagnostics_ts: float = 0.0
    # Engine-side capture→publish latency from the diagnostics heartbeat —
    # the AUTHORITATIVE KPI (measured inside the Backbone against the frame's
    # capture_ts). The bus's own latency_p50/p95 above measure this consumer
    # thread's processing lag instead (UI lag), a different thing.
    engine_latency_ms: dict | None = None


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
        self._frag_buf = FragmentBuffer()   # reassembles UdpSink's large-payload fragments
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

    def clear_live_state(self) -> None:
        """Empty every LIVE cache (tracks, zone states, observations, fps) —
        called on backbone STOP so the map/cards/panels blank immediately
        instead of aging out. Counters and latency stats are kept."""
        with self._lock:
            self._state.last_track2d_by_id.clear()
            self._state.last_track3d_by_id.clear()
            self._state.zone_state_by_zone.clear()
            self._state.observations_by_camera.clear()
            self._state.fps_by_camera = {}
            self._state.pipeline_fps = None
            self._state.diagnostics_ts = 0.0
            self._state.engine_latency_ms = None

    def snapshot(self) -> BusState:
        with self._lock:
            lats = sorted(self._latencies)
            return BusState(
                last_envelope_ts=self._state.last_envelope_ts,
                last_track2d_by_id=dict(self._state.last_track2d_by_id),
                last_track3d_by_id=dict(self._state.last_track3d_by_id),
                zone_state_by_zone=dict(self._state.zone_state_by_zone),
                observations_by_camera=dict(self._state.observations_by_camera),
                received=self._state.received,
                dropped_malformed=self._state.dropped_malformed,
                dropped_version=self._state.dropped_version,
                latency_p50_ms=self._pct(lats, 50),
                latency_p95_ms=self._pct(lats, 95),
                latency_samples=len(lats),
                fps_by_camera=dict(self._state.fps_by_camera),
                engine_latency_ms=(dict(self._state.engine_latency_ms)
                                   if self._state.engine_latency_ms else None),
                pipeline_fps=self._state.pipeline_fps,
                diagnostics_ts=self._state.diagnostics_ts,
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

        # Transport fragments (UdpSink splits payloads that would exceed the
        # path MTU — WSL2 mirrored networking drops big loopback datagrams).
        # Buffer until the group completes, then parse the joined text as if
        # it had arrived whole.
        if isinstance(msg, FragmentMessage):
            text = self._frag_buf.add(msg, time.time())
            if text is None:
                return
            try:
                msg = parse_envelope(json.loads(text))
            except Exception as exc:
                with self._lock:
                    self._state.dropped_malformed += 1
                logger.debug("bus_subscriber: bad reassembled envelope (%s): %s",
                             type(exc).__name__, exc)
                return

        now = time.time()
        ts = getattr(msg, "ts", None)
        with self._lock:
            self._state.received += 1
            self._state.last_envelope_ts = now
            # Latency window counts FRESH-per-pair messages only. Retained/
            # refresh types (zone_state refreshes) legitimately carry an OLD
            # ts (last change time) and would poison the percentile — a
            # static warehouse read as "1.4 s latency" while tracks were
            # arriving 74 ms after capture.
            if ts is not None and isinstance(
                    msg, (Track2DMessage, Track3DMessage, ObservationsMessage)):
                self._latencies.append(max(0.0, (now - float(ts)) * 1000.0))
            if isinstance(msg, Track2DMessage):
                self._state.last_track2d_by_id[msg.track_id] = msg
            elif isinstance(msg, Track3DMessage):
                self._state.last_track3d_by_id[msg.track_id] = msg
            elif isinstance(msg, ZoneStateMessage):
                self._state.zone_state_by_zone[msg.zone] = msg
            elif isinstance(msg, ObservationsMessage):
                self._state.observations_by_camera[msg.camera_id] = msg
            elif isinstance(msg, DiagnosticsMessage):
                self._state.fps_by_camera = dict(msg.fps_by_camera)
                self._state.pipeline_fps = float(msg.fps)
                self._state.diagnostics_ts = now
                try:
                    self._state.engine_latency_ms = msg.latency_ms.model_dump()
                except Exception:
                    self._state.engine_latency_ms = None

        # Broadcast to live WebSocket clients via the event loop, if attached.
        # NOTE: call_soon_threadsafe only SCHEDULES the callback — an exception
        # inside it (QueueFull once no /ws/tracks client is draining) would be
        # raised later ON the event loop, unhandled, and uvloop then logs the
        # whole handle repr including every queued message (the giant console
        # dumps). _offer_broadcast handles the full queue itself: drop the
        # OLDEST and keep the newest — latest-only, silent, bounded.
        # Only the types /ws/tracks actually sends (its _serialize drops the
        # rest) — anything else scheduled here is a wasted event-loop wakeup,
        # and at points-mode rates (~100 envelopes/s incl. mask-heavy
        # observations) that churn measurably starves the loop.
        if (self._broadcast_queue is not None and self._loop is not None
                and isinstance(msg, (Track2DMessage, Track3DMessage))):
            try:
                self._loop.call_soon_threadsafe(
                    _offer_broadcast, self._broadcast_queue, msg,
                )
            except RuntimeError:
                pass          # loop shutting down
