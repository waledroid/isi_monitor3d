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

# The depay+decode pair is codec-specific and chosen at start() from the stream's
# actual codec (see `_probe_rtsp_codec`). We use an *explicit* depay rather than
# `decodebin` on purpose:
#   * Race-free start. A depayloader has a static sink pad, so rtspsrc's delayed
#     link to it completes before the first buffer is pushed. `decodebin`'s
#     autoplug widens that window and rtspsrc intermittently aborts with
#     "Internal data stream error / streaming stopped, reason not-linked (-1)"
#     — observed ~1-in-4 starts on an H.264 camera that also carries audio.
#   * Audio-safe. The codec-specific depay only accepts its own video RTP, so an
#     audio pad the camera advertises (e.g. Dahua's PCM A-law track) is left
#     harmlessly unlinked on rtspsrc — whereas decodebin would build a dead audio
#     branch and trip the same not-linked error.
PIPELINE_TEMPLATE = (
    "rtspsrc name=src "
    "location={url} "
    "latency={latency_ms} "
    "drop-on-latency=true "
    "protocols=tcp "
    "ntp-sync=true "
    "buffer-mode=auto "
    "! {depay} "
    "! {decoder} "
    "! videoconvert "
    "! video/x-raw,format=BGR "
    "! appsink name=sink emit-signals=true sync=false max-buffers=1 drop=true"
)

# codec name (ffprobe `codec_name`) -> (depay element, decoder element).
_CODEC_ELEMENTS = {
    "h264": ("rtph264depay", "avdec_h264"),
    "hevc": ("rtph265depay", "avdec_h265"),
    "h265": ("rtph265depay", "avdec_h265"),
}
_DEFAULT_CODEC = "h264"


def _probe_rtsp_codec(url: str, *, timeout_s: float = 8.0) -> str | None:
    """Return the RTSP stream's video codec (``"h264"``/``"hevc"``) via ffprobe.

    Used to pick the right depay+decode pair. Returns ``None`` if ffprobe is
    absent, times out, or reports an unexpected codec — the caller then falls
    back to the H.264 elements (the pre-existing behaviour). Probing over TCP
    matches the pipeline's ``protocols=tcp`` so a TCP-only camera still answers.
    """
    import shutil
    import subprocess

    if shutil.which("ffprobe") is None:
        return None
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-rtsp_transport", "tcp",
             "-select_streams", "v:0",
             "-show_entries", "stream=codec_name",
             "-of", "default=noprint_wrappers=1:nokey=1", url],
            capture_output=True, text=True, timeout=timeout_s,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    codec = out.stdout.strip().lower().splitlines()
    return codec[0] if codec else None


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
        # camera's native rate. The decoder still decodes every frame (H.264/H.265
        # are inter-coded — you can't skip a P/B frame without decoding it), so
        # this cuts per-frame CPU + inference frequency, not decode. The best way to cut
        # decode too is to lower the camera's sub-stream FPS in its own web UI.
        self._capture_fps = float(capture_fps) if capture_fps else None
        self._codec: str | None = None  # resolved lazily on first _build_pipeline_str

    def _depay_decoder(self) -> tuple[str, str]:
        """Probe the stream codec once and map it to (depay, decoder) elements."""
        if self._codec is None:
            self._codec = _probe_rtsp_codec(self._url) or _DEFAULT_CODEC
        depay, decoder = _CODEC_ELEMENTS.get(self._codec, _CODEC_ELEMENTS[_DEFAULT_CODEC])
        if self._codec not in _CODEC_ELEMENTS:
            logger.warning(
                "%s: unrecognised codec %r, defaulting to %s",
                self._log_prefix(), self._codec, _DEFAULT_CODEC,
            )
        return depay, decoder

    def _build_pipeline_str(self) -> str:
        depay, decoder = self._depay_decoder()
        template = PIPELINE_TEMPLATE
        if self._capture_fps:
            rate = (
                f"! videorate drop-only=true "
                f"! video/x-raw,framerate={round(self._capture_fps)}/1 "
            )
            template = template.replace("! videoconvert ", rate + "! videoconvert ", 1)
        return template.format(
            url=self._url, latency_ms=self._latency_ms, depay=depay, decoder=decoder
        )
