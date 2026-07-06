"""Shared GStreamer-appsink ``FrameSource`` base for live cameras.

Both the RTSP and the V4L2 camera sources drive a GStreamer pipeline that ends
in an ``appsink name=sink emit-signals=true`` and convert each BGR buffer into a
``Frame``. The only thing that differs between them is the *pipeline string*
(``rtspsrc location=…`` vs ``v4l2src device=…``). Everything else — the daemon
thread + ``GLib.MainLoop``, the ``Queue(maxsize=1)`` drop-old buffering, the
capture-timestamp policy, and the bus error/eos/state handlers — is identical
and lives here exactly once.

Per-camera threading model (unchanged from the original RTSP source):

    Each instance owns one ``Gst.Pipeline`` + one ``GLib.MainLoop`` running in
    a daemon thread. The appsink's ``new-sample`` callback fires inside
    GStreamer's streaming thread, decodes the buffer to a NumPy BGR array, tags
    it with a capture timestamp, and pushes the resulting ``Frame`` into an
    internal ``Queue(maxsize=1)`` with drop-old semantics. The public
    ``frames()`` iterator pops from that queue.

Capture timestamp:

    Captured at the moment the sample arrives at the appsink, via
    ``time.time()`` — the earliest moment the Backbone observes the frame.
    Using it consistently across all cameras keeps cross-camera pairing stable.
    This is the single capture-time clock referenced everywhere downstream.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from abc import abstractmethod
from collections.abc import Iterator

import gi
import numpy as np

from backbone.core.interfaces import FrameSource
from backbone.core.types import Frame

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst  # noqa: E402

logger = logging.getLogger(__name__)

_GST_INIT_LOCK = threading.Lock()
_GST_INITIALIZED = False


def _ensure_gst_initialized() -> None:
    """Initialize GStreamer exactly once per process."""
    global _GST_INITIALIZED
    with _GST_INIT_LOCK:
        if not _GST_INITIALIZED:
            Gst.init(None)
            _GST_INITIALIZED = True


class GstAppsinkFrameSource(FrameSource):
    """Base for live cameras whose pipeline ends in ``appsink name=sink``.

    Subclasses implement :meth:`_build_pipeline_str` and validate their own
    constructor arguments before calling ``super().__init__``.
    """

    def __init__(
        self,
        camera_id: str,
        *,
        startup_timeout_s: float = 10.0,
        capture_fps: float | None = None,
    ) -> None:
        self._camera_id = camera_id
        self._startup_timeout_s = startup_timeout_s

        # Optional frame-rate cap, enforced in `_on_sample` by wall-clock interval
        # (NOT a GStreamer `videorate` element — that stalls on cameras which give
        # no valid buffer timestamps, e.g. some Hikvision-clone H.264 streams that
        # report avg_frame_rate=0/0). Dropping here, before the BGR copy + queue,
        # still caps every downstream consumer's load at `capture_fps`.
        self._min_interval: float | None = (1.0 / capture_fps) if capture_fps else None
        self._next_keep_ts: float = 0.0   # schedule-anchored deadline for the next kept frame

        self._queue: queue.Queue[Frame] = queue.Queue(maxsize=1)
        self._dropped: int = 0
        self._frame_idx: int = 0

        self._stop_event = threading.Event()
        self._started_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._loop: GLib.MainLoop | None = None
        self._pipeline: Gst.Pipeline | None = None
        self._error_message: str | None = None

    # ---- subclass hook ----

    @abstractmethod
    def _build_pipeline_str(self) -> str:
        """Return the GStreamer pipeline string (must contain ``appsink name=sink``)."""

    def _configure_pipeline(self, pipeline) -> None:
        """Optional hook: wire signals on the parsed pipeline BEFORE it goes
        PLAYING (e.g. an rtspsrc ``select-stream`` to drop an unwanted audio
        stream). Default no-op."""

    # ---- public API ----

    @property
    def camera_id(self) -> str:
        return self._camera_id

    @property
    def dropped_count(self) -> int:
        return self._dropped

    def start(self) -> None:
        """Start the GStreamer pipeline in a background thread.

        Blocks until the pipeline reaches PLAYING (or fails). Idempotent.
        """
        if self._thread is not None:
            return
        _ensure_gst_initialized()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name=f"{self._log_prefix()}",
        )
        self._thread.start()
        if not self._started_event.wait(timeout=self._startup_timeout_s):
            self.stop()
            raise RuntimeError(
                f"{self._log_prefix()}: pipeline did not reach PLAYING within "
                f"{self._startup_timeout_s}s"
            )
        if self._error_message:
            raise RuntimeError(f"{self._log_prefix()}: {self._error_message}")

    def frames(self) -> Iterator[Frame]:
        """Yield decoded frames until the pipeline stops."""
        if self._thread is None:
            self.start()
        while not self._stop_event.is_set():
            try:
                yield self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

    def stop(self) -> None:
        self._stop_event.set()
        if self._loop is not None and self._loop.is_running():
            self._loop.quit()
        if self._thread is not None and self._thread.is_alive():
            # A GStreamer NULL-state transition normally completes in < 300 ms;
            # a short join keeps STOP fast (the supervisor's SIGKILL grace is
            # the backstop for a truly hung pipeline — a daemon thread never
            # blocks process exit anyway).
            self._thread.join(timeout=1.5)
            self._thread = None

    # ---- internals ----

    def _log_prefix(self) -> str:
        return f"{type(self).__name__}[{self._camera_id}]"

    def _run(self) -> None:
        pipeline_str = self._build_pipeline_str()
        try:
            self._pipeline = Gst.parse_launch(pipeline_str)
        except GLib.Error as exc:
            self._error_message = f"pipeline parse failed: {exc}"
            self._started_event.set()
            return

        sink = self._pipeline.get_by_name("sink")
        if sink is None:
            self._error_message = "could not find appsink in pipeline"
            self._started_event.set()
            return
        sink.connect("new-sample", self._on_sample)

        # Wire any pre-PLAYING signals (select-stream etc.) — must happen before
        # set_state(PLAYING) so rtspsrc never sets up the rejected streams.
        self._configure_pipeline(self._pipeline)

        bus = self._pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message::error", self._on_bus_error)
        bus.connect("message::eos", self._on_bus_eos)
        bus.connect("message::state-changed", self._on_state_changed)

        self._pipeline.set_state(Gst.State.PLAYING)
        self._loop = GLib.MainLoop()
        try:
            self._loop.run()
        finally:
            if self._pipeline is not None:
                self._pipeline.set_state(Gst.State.NULL)
            self._stop_event.set()

    def _on_sample(self, sink) -> int:
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK

        # Snapshot wall-clock at the earliest observable moment — the appsink
        # callback fires immediately after the decoder emits the buffer. This is
        # what propagates through every downstream stage as the capture clock.
        capture_ts = time.time()

        # Frame-rate cap (timestamp-independent — see __init__). Keep a frame only
        # once we pass the next scheduled deadline, then advance the deadline by
        # exactly one interval. The schedule may lag real time by AT MOST one
        # interval (bounded credit): TCP-delivered RTSP arrives in bursts, and a
        # schedule re-anchored to each kept frame enforces a minimum spacing that
        # silently drops burst frames — measured 18 fps -> 14 fps on the site
        # camera with a 25 fps cap. With one interval of credit, a source slower
        # than the cap passes untouched while a faster one still caps at 1/interval
        # (plus at most one bonus frame after a stall).
        if self._min_interval is not None:
            if capture_ts < self._next_keep_ts:
                return Gst.FlowReturn.OK
            self._next_keep_ts = max(self._next_keep_ts + self._min_interval,
                                     capture_ts - self._min_interval)

        buf = sample.get_buffer()
        caps = sample.get_caps()
        structure = caps.get_structure(0)
        width = structure.get_value("width")
        height = structure.get_value("height")

        success, mapinfo = buf.map(Gst.MapFlags.READ)
        if not success:
            logger.warning("%s: buffer map failed", self._log_prefix())
            return Gst.FlowReturn.OK

        try:
            # mapinfo.data is a memoryview; copy because the buffer is released on unmap.
            arr = np.frombuffer(mapinfo.data, dtype=np.uint8)
            try:
                image = arr.reshape(height, width, 3).copy()
            except ValueError:
                logger.warning(
                    "%s: unexpected buffer size %d for %dx%d BGR",
                    self._log_prefix(),
                    arr.size,
                    width,
                    height,
                )
                return Gst.FlowReturn.OK
        finally:
            buf.unmap(mapinfo)

        frame = Frame(
            camera_id=self._camera_id,
            capture_ts=capture_ts,
            frame_idx=self._frame_idx,
            image=image,
        )
        self._frame_idx += 1

        # Drop-old put: when the consumer is slow, the live pipeline must
        # not back-pressure the camera.
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._dropped += 1
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                pass

        return Gst.FlowReturn.OK

    def _on_bus_error(self, _bus, msg) -> None:
        err, debug_info = msg.parse_error()
        self._error_message = f"{err.message} ({debug_info})"
        logger.error("%s: %s", self._log_prefix(), self._error_message)
        self._started_event.set()
        if self._loop is not None:
            self._loop.quit()

    def _on_bus_eos(self, _bus, _msg) -> None:
        logger.info("%s: EOS", self._log_prefix())
        if self._loop is not None:
            self._loop.quit()

    def _on_state_changed(self, _bus, msg) -> None:
        # Signal "started" only when the *pipeline* (not a child element)
        # reaches PLAYING — that means the source connected and the first
        # frame path is live.
        if self._pipeline is None or msg.src is not self._pipeline:
            return
        _old, new, _pending = msg.parse_state_changed()
        if new == Gst.State.PLAYING:
            self._started_event.set()
