"""Shared live-camera hub: one RTSP/V4L2 session per camera, fanned out to all
viewers.

The dashboard opens an MJPEG ``<img>`` per camera view and re-opens it on every
tab switch, settings save, or calibration change (the stream URL carries a
cache-busting nonce). Naively each open built its own ``RtspFrameSource`` — i.e.
its own RTSP session to the camera. Hikvision cameras cap concurrent sessions,
and a V4L2 device can't be opened twice at all, so stacking sessions caused the
intermittent "camera won't display" failures.

This hub keeps exactly ONE long-lived source per camera. A single *pump* thread
pulls decoded frames and publishes the latest into a shared slot; every viewer
subscribes to that slot via :meth:`CameraStream.read`. The server→camera leg
stays up across UI churn — only the browser→server HTTP leg reconnects. When the
last viewer leaves, the source is stopped after a short idle grace period so an
unused camera isn't held open.

Resilience is owned here too: the pump retries a failing/slow live source with
backoff and publishes a "connecting…" placeholder so viewers always receive
valid image bytes (never the ``<img>`` alt caption).
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator

import cv2
import numpy as np
from backbone.ingestion.replay import ReplayFrameSource
from backbone.ingestion.rtsp import RtspFrameSource
from backbone.ingestion.v4l2 import V4l2FrameSource

logger = logging.getLogger(__name__)

# Keep a camera session open this long after the last viewer disconnects. Tab
# switches / nonce bumps drop+reopen within ~1 s, so a few seconds of grace
# avoids tearing the RTSP session down and renegotiating needlessly.
IDLE_GRACE_S = 15.0
# (Re)connect backoff bounds for a flaky live source.
_RETRY_MIN_S = 1.0
_RETRY_MAX_S = 5.0


def _placeholder_frame(text: str) -> np.ndarray:
    """A dark 640x360 BGR frame with a status caption — published as the latest
    frame while a live source is (re)connecting, so viewers always receive valid
    image bytes and never fall back to the ``<img>`` alt caption."""
    img = np.full((360, 640, 3), 26, dtype=np.uint8)
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(img, text, ((640 - tw) // 2, 188), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (180, 180, 180), 2, cv2.LINE_AA)
    return img


# While on our OWN source, peek at the frame bus this often — when the
# perception producer comes up we hand over and close the duplicate session.
_BUS_RECHECK_S = 5.0


def _cfg_key(src_cfg: dict) -> tuple:
    """A hashable, comparable key for a source config so the hub can tell when a
    camera was reconfigured (URL/device/type changed) and must be rebuilt."""
    return tuple(sorted((k, repr(v)) for k, v in src_cfg.items()))


def _build_source(camera_id: str, plugin: str, src_cfg: dict):
    """Instantiate (and start, for live plugins) the configured frame source."""
    if plugin == "rtsp":
        src = RtspFrameSource(camera_id=camera_id, **src_cfg)
        src.start()
    elif plugin == "v4l2":
        src = V4l2FrameSource(camera_id=camera_id, **src_cfg)
        src.start()
    elif plugin == "replay":
        src = ReplayFrameSource(camera_id=camera_id, **src_cfg)
    else:
        raise ValueError(f"unsupported source plugin {plugin!r}")
    return src


class CameraStream:
    """One shared live source for a camera + latest-frame fan-out.

    A single pump thread (producer) publishes the most recent decoded frame into
    a slot guarded by a condition variable; any number of :meth:`read` iterators
    (consumers) wake on each new frame. A slow consumer simply skips intermediate
    frames — it never back-pressures the camera.
    """

    def __init__(self, camera_id: str, plugin: str, src_cfg: dict) -> None:
        self.camera_id = camera_id
        self.plugin = plugin
        self.src_cfg = dict(src_cfg)
        self.key = (plugin, _cfg_key(src_cfg))

        self._cond = threading.Condition()
        self._latest: np.ndarray | None = None
        self._shm_reader = None               # lazy FrameShmReader (frame bus)
        self._latest_ts: float = 0.0         # capture_ts of the latest REAL frame
        self._latest_is_placeholder = True   # the first published frame is always a placeholder
        self._version = 0
        self._finished = False          # replay reached EOF → readers should stop
        self._stop = threading.Event()
        self._src = None                # the live source while the pump holds it
        self._pump_thread: threading.Thread | None = None

    # ---- lifecycle ----

    def ensure_pump(self) -> None:
        """Start the pump thread if it isn't already running. Idempotent."""
        if self._pump_thread is None or not self._pump_thread.is_alive():
            self._stop.clear()
            with self._cond:
                self._finished = False
            self._pump_thread = threading.Thread(
                target=self._pump, name=f"camhub[{self.camera_id}]", daemon=True)
            self._pump_thread.start()

    def stop(self) -> None:
        """Stop the pump + underlying source and wake all readers."""
        self._stop.set()
        src = self._src
        if src is not None and hasattr(src, "stop"):
            try:
                src.stop()
            except Exception:  # stopping a half-built source must never raise
                logger.debug("camhub %s: source stop raised", self.camera_id, exc_info=True)
        with self._cond:
            self._cond.notify_all()
        t = self._pump_thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=6.0)
        self._pump_thread = None

    # ---- pump (single producer) ----

    def _publish(self, frame: np.ndarray, *, placeholder: bool = False,
                 capture_ts: float = 0.0) -> None:
        with self._cond:
            self._latest = frame
            self._latest_is_placeholder = placeholder
            if not placeholder:
                # The KPI clock: capture_ts travels with the frame so the
                # perception producer can stamp DetectionSetMessage.ts with
                # the true capture time, not the read time.
                self._latest_ts = capture_ts
            self._version += 1
            self._cond.notify_all()

    def _pump(self) -> None:
        backoff = _RETRY_MIN_S
        self._publish(_placeholder_frame("connecting to camera…"), placeholder=True)
        while not self._stop.is_set():
            # Prefer the SHARED FRAME BUS: while the perception producer runs,
            # it publishes every decoded frame to /dev/shm — stream from there
            # (one ingest+decode per camera in the whole system, and the
            # panels show the exact pixels the models saw). Absent/stale bus
            # (backbone stopped, frames mode, pre-START preview) → open our
            # own source exactly as before. _stream_from_bus returns when the
            # bus goes stale, so the loop naturally falls through to RTSP and
            # back again when the bus revives.
            if self.plugin in ("rtsp", "v4l2") and self._stream_from_bus():
                if self._stop.is_set():
                    return
                self._publish(_placeholder_frame("reconnecting…"), placeholder=True)
                continue
            try:
                src = _build_source(self.camera_id, self.plugin, self.src_cfg)
            except Exception as exc:  # startup failure must not kill the pump
                logger.warning("camhub %s: source start failed (%s); retrying",
                               self.camera_id, exc)
                self._publish(_placeholder_frame("connecting to camera…"), placeholder=True)
                if self._stop.wait(backoff):
                    return
                backoff = min(backoff * 2, _RETRY_MAX_S)
                continue
            backoff = _RETRY_MIN_S
            self._src = src
            last_bus_check = 0.0
            try:
                for frame in src.frames():
                    if self._stop.is_set():
                        break
                    self._publish(frame.image, capture_ts=frame.capture_ts)
                    # Periodically peek at the bus: when the producer comes up
                    # (START), hand over and drop our own RTSP session.
                    now_m = time.monotonic()
                    if (self.plugin in ("rtsp", "v4l2")
                            and now_m - last_bus_check > _BUS_RECHECK_S):
                        last_bus_check = now_m
                        if self._bus_reader().fresh():
                            logger.info("camhub %s: frame bus is live — "
                                        "releasing own source", self.camera_id)
                            break
            except Exception as exc:  # a mid-stream source error → reconnect
                logger.warning("camhub %s: source error (%s)", self.camera_id, exc)
            finally:
                self._src = None
                if hasattr(src, "stop"):
                    try:
                        src.stop()
                    except Exception:
                        logger.debug("camhub %s: source stop raised", self.camera_id,
                                     exc_info=True)
            if self.plugin == "replay":           # a finished file is done
                with self._cond:
                    self._finished = True
                    self._cond.notify_all()
                return
            if self._stop.is_set():
                return
            self._publish(_placeholder_frame("reconnecting…"), placeholder=True)
            if self._stop.wait(backoff):
                return

    def _bus_reader(self):
        if self._shm_reader is None:
            from backbone.shared.frame_shm import FrameShmReader
            self._shm_reader = FrameShmReader(self.camera_id)
        return self._shm_reader

    def _stream_from_bus(self) -> bool:
        """Stream from the shared frame bus until it goes stale.

        Returns True if we streamed at least one frame (the caller re-loops:
        stale bus → RTSP fallback), False when the bus was never fresh (go
        straight to RTSP without logging noise)."""
        reader = self._bus_reader()
        got = reader.latest()
        if got is None:
            return False
        logger.info("camhub %s: streaming from the shared frame bus", self.camera_id)
        last_ts = 0.0
        while not self._stop.is_set():
            got = reader.latest()
            if got is None:
                logger.info("camhub %s: frame bus stale — falling back to own source",
                            self.camera_id)
                return True
            image, ts = got
            if ts > last_ts:
                last_ts = ts
                self._publish(image, capture_ts=ts)
            else:
                self._stop.wait(0.005)
        return True

    # ---- reader (multi consumer) ----

    def read(self) -> Iterator[np.ndarray]:
        """Yield the latest frame each time a newer one is published. Returns
        when the stream is stopped, or (replay) when the source is exhausted."""
        last_seen = -1
        while True:
            with self._cond:
                while (self._version == last_seen and not self._stop.is_set()
                       and not self._finished):
                    self._cond.wait(timeout=1.0)
                if self._stop.is_set():
                    return
                if self._version == last_seen and self._finished:
                    return
                last_seen = self._version
                frame = self._latest
            if frame is not None:
                yield frame

    def latest_real_frame_with_ts(self) -> tuple[np.ndarray, float] | None:
        """Latest REAL frame + its capture_ts (0.0 when the source predates
        the ts plumbing). None while only placeholders have been published."""
        with self._cond:
            if self._latest is not None and not self._latest_is_placeholder:
                return self._latest, self._latest_ts
            return None

    def latest_real_frame(self) -> np.ndarray | None:
        """Non-blocking snapshot of the most recent genuine frame (not the
        "connecting…" placeholder), or ``None``. For multi-stream consumers (the
        Mode-2 unified composite) that grab the current frame from each camera
        without blocking the driving loop, and for per-camera liveness checks."""
        with self._cond:
            if self._latest is not None and not self._latest_is_placeholder:
                return self._latest
            return None

    def wait_for_real_frame(self, timeout: float = 4.0) -> np.ndarray | None:
        """Block until a genuine decoded frame (not a "connecting…" placeholder)
        is available, and return it; return ``None`` on timeout/stop.

        One-shot consumers (e.g. the MAP warp-snapshot) need an actual camera
        frame, not the synthetic placeholder the pump publishes first while the
        live source warms up. The continuous CAM stream advances past it on its
        own; a single ``read()`` would capture it, hence this explicit wait."""
        with self._cond:
            deadline = time.monotonic() + timeout
            while not self._stop.is_set():
                if self._latest is not None and not self._latest_is_placeholder:
                    return self._latest
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(timeout=remaining)
        return None


class CameraHub:
    """Registry of one :class:`CameraStream` per camera, with reader-counted
    lifecycle and idle shutdown."""

    def __init__(self) -> None:
        self._streams: dict[str, CameraStream] = {}
        self._readers: dict[str, int] = {}
        self._idle_timers: dict[str, threading.Timer] = {}
        self._lock = threading.Lock()

    def acquire(self, camera_id: str, plugin: str, src_cfg: dict) -> CameraStream:
        """Register one more viewer for ``camera_id`` and return its shared
        stream — creating (or rebuilding, on config change) the source as needed.
        Pair every ``acquire`` with a ``release`` in a ``finally``."""
        key = (plugin, _cfg_key(src_cfg))
        to_stop: CameraStream | None = None
        with self._lock:
            stream = self._streams.get(camera_id)
            if stream is not None and stream.key != key:
                to_stop = stream                 # reconfigured → rebuild (stop outside lock)
                self._streams.pop(camera_id, None)
                self._readers.pop(camera_id, None)
                stream = None
            if stream is None:
                stream = CameraStream(camera_id, plugin, src_cfg)
                self._streams[camera_id] = stream
                self._readers[camera_id] = 0
            timer = self._idle_timers.pop(camera_id, None)
            if timer is not None:
                timer.cancel()                   # a viewer returned before idle shutdown
            self._readers[camera_id] += 1
            stream.ensure_pump()
        if to_stop is not None:
            logger.info("camhub %s: config changed → rebuilding source", camera_id)
            to_stop.stop()
        return stream

    def release(self, stream: CameraStream) -> None:
        """Drop one viewer. When the last leaves, schedule an idle shutdown."""
        camera_id = stream.camera_id
        with self._lock:
            if self._streams.get(camera_id) is not stream:
                return                           # already replaced/retired
            self._readers[camera_id] = max(0, self._readers.get(camera_id, 0) - 1)
            if self._readers[camera_id] == 0 and camera_id not in self._idle_timers:
                timer = threading.Timer(IDLE_GRACE_S, self._retire, args=(camera_id, stream))
                timer.daemon = True
                self._idle_timers[camera_id] = timer
                timer.start()

    def _retire(self, camera_id: str, stream: CameraStream) -> None:
        with self._lock:
            if self._streams.get(camera_id) is not stream:
                return
            if self._readers.get(camera_id, 0) > 0:
                return                           # someone reconnected during the grace window
            self._streams.pop(camera_id, None)
            self._readers.pop(camera_id, None)
            self._idle_timers.pop(camera_id, None)
        logger.info("camhub %s: idle %.0fs → releasing source", camera_id, IDLE_GRACE_S)
        stream.stop()

    def shutdown(self) -> None:
        """Stop every source — called from the app's lifespan teardown."""
        with self._lock:
            streams = list(self._streams.values())
            self._streams.clear()
            self._readers.clear()
            for timer in self._idle_timers.values():
                timer.cancel()
            self._idle_timers.clear()
        for stream in streams:
            stream.stop()


_HUB = CameraHub()


def get_hub() -> CameraHub:
    """The process-wide camera hub singleton."""
    return _HUB
