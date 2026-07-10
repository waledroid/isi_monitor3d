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
import time
from collections.abc import Callable

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
    # drop-on-latency=false: over TCP the jitterbuffer's per-packet dropping
    # only loses frames (measured ~1 fps on the site cameras) — the appsink's
    # max-buffers=1 drop=true already keeps end-to-end latency bounded by
    # always serving the newest decoded frame.
    "drop-on-latency=false "
    "protocols=tcp "
    "ntp-sync=true "
    "buffer-mode=auto "
    "! {depay} "
    "! {decode_chain} "
    "! {sink_caps} "
    "! appsink name=sink emit-signals=true sync=false max-buffers=1 drop=true"
)

# Final appsink caps. `output_wh` appends width/height so the upstream
# converter performs the downscale IN the pipeline — on the GPU for the nvdec
# chain (cudaconvertscale), via an extra `videoscale` for the software chain.
_SINK_CAPS = "video/x-raw,format=BGR"

# codec name (ffprobe `codec_name`) -> (depay element, SOFTWARE decode chain).
_CODEC_ELEMENTS = {
    "h264": ("rtph264depay", "avdec_h264 ! videoconvert"),
    "hevc": ("rtph265depay", "avdec_h265 ! videoconvert"),
    "h265": ("rtph265depay", "avdec_h265 ! videoconvert"),
}
# codec -> NVDEC decode chain (Phase 0.5 of the DeepStream plan): hardware
# decode + GPU colorspace conversion, delivered to the appsink as plain
# system-memory BGR so _on_sample is untouched. Proven live on the site
# cameras (docs/deepstream-ingestion-plan.md, Phase 0 results).
_NVDEC_ELEMENTS = {
    "h264": ("rtph264depay",
             "h264parse ! nvh264dec ! cudaconvertscale "
             "! video/x-raw(memory:CUDAMemory),format=BGR ! cudadownload"),
    "hevc": ("rtph265depay",
             "h265parse ! nvh265dec ! cudaconvertscale "
             "! video/x-raw(memory:CUDAMemory),format=BGR ! cudadownload"),
    "h265": ("rtph265depay",
             "h265parse ! nvh265dec ! cudaconvertscale "
             "! video/x-raw(memory:CUDAMemory),format=BGR ! cudadownload"),
}
_DEFAULT_CODEC = "h264"
_DECODERS = ("software", "nvdec")

# Compressed-bitstream tap (video passthrough). When a `nal_tap` callback is
# given, a `tee` splits the stream RIGHT AFTER the depayloader: the main
# branch decodes exactly as before, the tap branch re-frames the ORIGINAL
# H.264/H.265 bitstream into Annex-B access units and hands them to the
# callback — zero re-encode, near-zero CPU. Construction notes:
#   * The tee sits between two static pads (depay src → tee sink), so the
#     race-free-linking property of the explicit-depay design is preserved.
#   * The tap branch has its OWN parser instance (the nvdec decode chain
#     already starts with h264parse/h265parse — that one stays in the main
#     branch, untouched). `config-interval=-1` makes the parser re-inject
#     SPS/PPS(/VPS) before every keyframe, mandatory so a late-joining
#     consumer can start decoding at any keyframe.
#   * `queue leaky=downstream` isolates the tap from the decode branch: a
#     slow tap consumer drops ITS buffers and can never back-pressure decode.
# This slots into PIPELINE_TEMPLATE's {decode_chain} placeholder, so without
# a tap the rendered pipeline string is byte-identical to before.
_NAL_TAP_TEMPLATE = (
    "tee name=nal_t "
    "! queue leaky=downstream max-size-buffers=120 "
    "! {parse} config-interval=-1 "
    "! video/x-{media},stream-format=byte-stream,alignment=au "
    "! appsink name=nal_sink emit-signals=true sync=false max-buffers=8 drop=false "
    "nal_t. ! {decode_chain}"
)
# codec -> (tap parser element, caps media suffix). Unknown codecs fall back
# to H.264, mirroring `_depay_decoder`'s default.
_NAL_PARSE = {
    "h264": ("h264parse", "h264"),
    "hevc": ("h265parse", "h265"),
    "h265": ("h265parse", "h265"),
}

# Codec probe cache, keyed by URL. ffprobe costs up to 8 s per call and a
# stream's codec never changes mid-process — but the dashboard's camera hub
# re-acquires sources every time the last viewer detaches (page reloads!), so
# an uncached probe made every reconnect pay seconds of dead time (the
# "cam 2 loads slowly" symptom; H.265 probes are the slowest).
_CODEC_CACHE: dict[str, str] = {}


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
        decoder: str = "software",
        output_wh: tuple[int, int] | list[int] | None = None,
        nal_tap: Callable[[bytes, float, bool], None] | None = None,
    ) -> None:
        if not url.startswith(("rtsp://", "rtsps://")):
            raise ValueError(f"RtspFrameSource: bad URL {url!r}, expected rtsp:// or rtsps://")
        if decoder not in _DECODERS:
            raise ValueError(
                f"RtspFrameSource: decoder={decoder!r}, expected one of {_DECODERS}")
        if output_wh is not None:
            output_wh = (int(output_wh[0]), int(output_wh[1]))
            if output_wh[0] <= 0 or output_wh[1] <= 0:
                raise ValueError(f"RtspFrameSource: bad output_wh {output_wh!r}")
        # The frame-rate cap is enforced in the appsink callback (the base class),
        # NOT a GStreamer `videorate` element: `videorate` stalls on cameras that
        # report no valid frame rate / broken buffer timestamps (e.g. some
        # Hikvision-clone H.264 streams, avg_frame_rate=0/0 — the pipeline froze
        # after a few frames). Capping in the callback is timestamp-independent and
        # still throttles the BGR copy + every downstream consumer. The decoder
        # still decodes every frame (H.264/H.265 are inter-coded); to cut decode
        # too, lower the camera's sub-stream FPS in its own web UI.
        super().__init__(camera_id, startup_timeout_s=startup_timeout_s,
                         capture_fps=capture_fps)
        self._url = url
        self._latency_ms = int(latency_ms)
        self._decoder = decoder
        # Downscale delivered frames to this (W, H) inside the pipeline. Pick a
        # size matching the source aspect (16:9 cameras → e.g. 1280x720) — the
        # caps force EXACT dimensions, a mismatched aspect distorts. Downstream
        # geometry is scale-guarded (detections are mapped to calibration-frame
        # pixels in the orchestrator), so calibration stays at native res.
        self._output_wh = output_wh
        # Optional compressed-bitstream tap: called with (annex_b_access_unit,
        # capture_ts, is_keyframe) for every AU, from the tap branch's own
        # streaming thread. capture_ts is time.time() at the tap callback —
        # the same clock policy as the decoded-frame appsink.
        self._nal_tap = nal_tap
        self._codec: str | None = None  # resolved lazily on first _build_pipeline_str

    @property
    def nal_codec(self) -> str | None:
        """The tapped bitstream's codec, normalized to ``"h264"``/``"h265"``.

        ``None`` until the codec probe has run (i.e. before ``start()``).
        Unknown codecs report ``"h264"`` — matching the tap branch actually
        built (``_depay_decoder`` defaults unknowns to the H.264 elements).
        """
        if self._codec is None:
            return None
        return "h265" if self._codec in ("hevc", "h265") else "h264"

    def _nvdec_available(self) -> bool:
        """True iff the GStreamer nvcodec elements exist on this machine.

        Initializes GStreamer if needed — ``ElementFactory.find`` before
        ``Gst.init`` reports nothing and would silently fall back to software
        decode even on a CUDA machine."""
        try:
            from backbone.ingestion._gst_source import Gst
            if not Gst.is_initialized():
                Gst.init(None)
            return (Gst.ElementFactory.find("nvh264dec") is not None
                    and Gst.ElementFactory.find("cudaconvertscale") is not None)
        except Exception:
            return False

    def _depay_decoder(self) -> tuple[str, str]:
        """Probe the stream codec once and map it to (depay, decode-chain).

        ``decoder: nvdec`` selects the NVDEC hardware chain; it degrades to
        the software chain (with a warning, never a failure) when the nvcodec
        plugin isn't present — a shared config stays runnable on machines
        without an NVIDIA GPU."""
        if self._codec is None:
            cached = _CODEC_CACHE.get(self._url)
            if cached is None:
                cached = _probe_rtsp_codec(self._url) or _DEFAULT_CODEC
                _CODEC_CACHE[self._url] = cached
            self._codec = cached
        table = _CODEC_ELEMENTS
        if self._decoder == "nvdec":
            if self._nvdec_available():
                table = _NVDEC_ELEMENTS
            else:
                logger.warning(
                    "%s: decoder=nvdec requested but GStreamer nvcodec is not "
                    "available — falling back to software decode",
                    self._log_prefix(),
                )
        depay, chain = table.get(self._codec, table[_DEFAULT_CODEC])
        if self._codec not in table:
            logger.warning(
                "%s: unrecognised codec %r, defaulting to %s",
                self._log_prefix(), self._codec, _DEFAULT_CODEC,
            )
        return depay, chain

    def _build_pipeline_str(self) -> str:
        depay, decode_chain = self._depay_decoder()
        sink_caps = _SINK_CAPS
        if self._output_wh is not None:
            w, h = self._output_wh
            sink_caps = f"{_SINK_CAPS},width={w},height={h}"
            if "cudaconvertscale" not in decode_chain:
                # Software chain: videoconvert can't scale — add videoscale.
                # (The nvdec chain scales on the GPU in cudaconvertscale.)
                decode_chain = f"{decode_chain} ! videoscale"
        if self._nal_tap is not None:
            parse, media = _NAL_PARSE.get(self._codec, _NAL_PARSE[_DEFAULT_CODEC])
            decode_chain = _NAL_TAP_TEMPLATE.format(
                parse=parse, media=media, decode_chain=decode_chain)
        return PIPELINE_TEMPLATE.format(
            url=self._url, latency_ms=self._latency_ms,
            depay=depay, decode_chain=decode_chain, sink_caps=sink_caps,
        )

    def _configure_pipeline(self, pipeline) -> None:
        # Reject the camera's audio stream at the rtspsrc level. With an explicit
        # video depayloader, a camera that carries audio (e.g. cam_a's PCM A-law
        # track) leaves its audio RTP pad unlinked, and rtspsrc intermittently
        # aborts with "streaming stopped, reason not-linked (-1)" (~1-in-3 starts).
        # `select-stream` tells rtspsrc to never set up the audio stream at all.
        src = pipeline.get_by_name("src")
        if src is not None:
            src.connect("select-stream", self._select_stream)
        if self._nal_tap is not None:
            nal_sink = pipeline.get_by_name("nal_sink")
            if nal_sink is not None:
                nal_sink.connect("new-sample", self._on_nal_sample)

    def _on_nal_sample(self, sink) -> int:
        """Tap-branch appsink callback: one Annex-B access unit per buffer.

        Runs on the tap branch's own GStreamer streaming thread — a slow
        `nal_tap` stalls only the leaky tap queue, never the decode branch.
        The callback is exception-guarded: a broken consumer must never kill
        the streaming thread (a half-torn pipeline segfaults later).
        """
        from backbone.ingestion._gst_source import Gst

        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK
        capture_ts = time.time()
        buf = sample.get_buffer()
        keyframe = not buf.has_flags(Gst.BufferFlags.DELTA_UNIT)
        success, mapinfo = buf.map(Gst.MapFlags.READ)
        if not success:
            logger.warning("%s: nal buffer map failed", self._log_prefix())
            return Gst.FlowReturn.OK
        try:
            data = bytes(mapinfo.data)
        finally:
            buf.unmap(mapinfo)
        try:
            self._nal_tap(data, capture_ts, keyframe)
        except Exception:
            logger.warning("%s: nal_tap callback failed", self._log_prefix(),
                           exc_info=True)
        return Gst.FlowReturn.OK

    @staticmethod
    def _select_stream(rtspsrc, num: int, caps) -> bool:
        """rtspsrc ``select-stream``: return False to skip the audio stream."""
        try:
            media = caps.get_structure(0).get_string("media")
        except Exception:
            return True
        return media != "audio"
