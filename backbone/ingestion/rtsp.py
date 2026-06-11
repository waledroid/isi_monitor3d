"""``RtspFrameSource`` — GStreamer-driven RTSP capture via PyGObject.

We bypass OpenCV's video I/O entirely. The conda-forge OpenCV build ships
without the GStreamer backend (FFmpeg only), and the architecture calls for
direct GStreamer control of the pipeline anyway: ``latency=``, drop-on-latency,
RTCP NTP sync, batched appsink — all features that require the native
GStreamer API surface.

The threading model, appsink decode, capture-timestamp policy, and bus
handling are shared with the V4L2 source via
``backbone.ingestion._gst_source.GstAppsinkFrameSource``; this module only
contributes the RTSP-specific pipeline string and URL validation.

Capture timestamp:

    Captured at the moment the sample arrives at the appsink, via
    ``time.time()`` (in the shared base). With ``rtspsrc latency=100`` this
    lags the camera shutter by ~100 ms + decode time, but it is the earliest
    moment the Backbone can observe the frame, and using it consistently
    across all cameras keeps cross-camera pairing stable. End-to-end latency
    probes measure (publish - capture_ts), so they correctly reflect the time
    the Backbone holds the frame, not the time it sat in the jitter buffer.

    Aligning ``capture_ts`` to true NTP wall-clock (so it reflects the
    camera shutter) would require setting a ``GstNetClientClock`` against
    an NTP server and using ``buf.pts + base_time``. That is a future
    optimization not required for the < 200 ms KPI.
"""

from __future__ import annotations

import logging

from backbone.core.interfaces import frame_source_registry

from ._gst_source import GstAppsinkFrameSource, _ensure_gst_initialized

__all__ = ["PIPELINE_TEMPLATE", "RtspFrameSource", "_ensure_gst_initialized"]

logger = logging.getLogger(__name__)

PIPELINE_TEMPLATE = (
    "rtspsrc name=src "
    "location={url} "
    "latency={latency_ms} "
    "drop-on-latency=true "
    "protocols=tcp "
    "ntp-sync=true "
    "buffer-mode=auto "
    "! rtph264depay "
    "! avdec_h264 "
    "! videoconvert "
    "! video/x-raw,format=BGR "
    "! appsink name=sink emit-signals=true sync=false max-buffers=1 drop=true"
)


@frame_source_registry.register("rtsp")
class RtspFrameSource(GstAppsinkFrameSource):
    """RTSP camera as a Backbone ``FrameSource``."""

    def __init__(
        self,
        camera_id: str,
        url: str,
        *,
        latency_ms: int = 100,
        startup_timeout_s: float = 10.0,
        capture_fps: float | None = None,
    ) -> None:
        if not url.startswith(("rtsp://", "rtsps://")):
            raise ValueError(f"RtspFrameSource: bad URL {url!r}, expected rtsp:// or rtsps://")
        super().__init__(camera_id, startup_timeout_s=startup_timeout_s)
        self._url = url
        self._latency_ms = int(latency_ms)
        # Optional input frame-rate cap. When set, a `videorate drop-only` element
        # throttles the stream to `capture_fps` right after decode — so the BGR
        # convert + appsink copy (and every downstream consumer: the Backbone's own
        # detector AND the dashboard preview) run at `capture_fps` instead of the
        # camera's native rate. avdec_h264 still decodes every frame (H.264 is
        # inter-coded — you can't skip a P/B frame without decoding it), so this
        # cuts per-frame CPU + inference frequency, not decode. The best way to cut
        # decode too is to lower the camera's sub-stream FPS in its own web UI.
        self._capture_fps = float(capture_fps) if capture_fps else None

    def _build_pipeline_str(self) -> str:
        if not self._capture_fps:
            return PIPELINE_TEMPLATE.format(url=self._url, latency_ms=self._latency_ms)
        rate = (
            f"! videorate drop-only=true "
            f"! video/x-raw,framerate={int(round(self._capture_fps))}/1 "
        )
        capped = PIPELINE_TEMPLATE.replace("! videoconvert ", rate + "! videoconvert ", 1)
        return capped.format(url=self._url, latency_ms=self._latency_ms)
