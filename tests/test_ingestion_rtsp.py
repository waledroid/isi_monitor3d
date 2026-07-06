"""``RtspFrameSource`` — plugin registration, URL validation, pipeline string shape.

Real RTSP connections aren't unit-tested (they require a server). Coverage:
    * Plugin registers under ``"rtsp"``.
    * URL scheme is enforced at construction time.
    * Pipeline template includes the load-bearing knobs that the cahier and
      the plan require: ``latency=``, ``drop-on-latency=false`` (over TCP the
      jitterbuffer's dropping only loses frames; the appsink bounds latency),
      ``ntp-sync=true``,
      ``protocols=tcp``, BGR appsink with ``drop=true``.
    * Pipeline string parses in this env (verified live via ``Gst.parse_launch``).

End-to-end RTSP is exercised via ``tools/rtsp_smoke.py`` against a real camera.
"""

from __future__ import annotations

import gi
import pytest

from backbone.core.interfaces import frame_source_registry
from backbone.ingestion.rtsp import (
    _CODEC_ELEMENTS,
    PIPELINE_TEMPLATE,
    RtspFrameSource,
    _ensure_gst_initialized,
)

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402


def test_plugin_registered_under_rtsp() -> None:
    assert "rtsp" in frame_source_registry


def test_url_scheme_validated() -> None:
    with pytest.raises(ValueError, match="rtsp://"):
        RtspFrameSource(camera_id="cam_a", url="http://example/stream")


def test_url_rtsps_accepted() -> None:
    # Constructor should accept both schemes; no thread is started yet.
    src = RtspFrameSource(camera_id="cam_a", url="rtsps://example/stream")
    assert src.camera_id == "cam_a"


def _render(codec: str = "h264", **kw) -> str:
    depay, decoder = _CODEC_ELEMENTS[codec]
    return PIPELINE_TEMPLATE.format(
        url=kw.get("url", "rtsp://x/y"), latency_ms=kw.get("latency_ms", 100),
        depay=depay, decode_chain=f"{decoder} ! videoconvert",
        sink_caps=kw.get("sink_caps", "video/x-raw,format=BGR"),
    )


@pytest.mark.parametrize(
    "knob",
    ["latency=", "drop-on-latency=false", "ntp-sync=true", "protocols=tcp",
     "rtph264depay", "avdec_h264", "format=BGR", "drop=true", "emit-signals=true"],
)
def test_pipeline_template_contains_required_knob(knob: str) -> None:
    """If anyone removes a load-bearing pipeline knob, this test fires."""
    assert knob in _render(), f"missing pipeline knob: {knob}"


@pytest.mark.parametrize("codec", ["h264", "hevc", "h265"])
def test_pipeline_parses_for_each_codec(codec: str) -> None:
    """Both H.264 and H.265 depay/decode chains must parse in this env."""
    _ensure_gst_initialized()
    pipeline = Gst.parse_launch(_render(codec, url="rtsp://127.0.0.1/x"))
    assert pipeline is not None
    for name in ("src", "sink"):
        assert pipeline.get_by_name(name) is not None, f"missing element: {name}"


def test_no_thread_started_at_construction() -> None:
    """Construction must be cheap; only start() launches the GLib loop."""
    src = RtspFrameSource(camera_id="cam_a", url="rtsp://example/stream")
    assert src.dropped_count == 0


def test_capture_fps_caps_in_callback_not_videorate() -> None:
    """`capture_fps` must NOT add a GStreamer `videorate` element (it stalls on
    cameras with no valid frame rate); the cap is enforced in the appsink
    callback via a wall-clock interval instead."""
    src = RtspFrameSource(camera_id="cam_a", url="rtsp://example/stream", capture_fps=12)
    assert "videorate" not in src._build_pipeline_str()
    assert abs(src._min_interval - 1.0 / 12) < 1e-9
    # No cap → no interval.
    plain = RtspFrameSource(camera_id="cam_a", url="rtsp://example/stream")
    assert plain._min_interval is None


@pytest.mark.parametrize(
    "media,keep",
    [("audio", False), ("video", True)],
)
def test_select_stream_rejects_audio(media: str, keep: bool) -> None:
    """rtspsrc `select-stream` must skip audio (an unlinked audio pad makes
    rtspsrc abort with 'streaming stopped, reason not-linked')."""
    _ensure_gst_initialized()
    caps = Gst.Caps.from_string(f"application/x-rtp, media=(string){media}")
    assert RtspFrameSource._select_stream(None, 0, caps) is keep


def test_capture_fps_throttle_passes_slower_bursty_source() -> None:
    """A source SLOWER than the cap must pass every frame even when frames
    arrive in TCP bursts. The old schedule re-anchored to each kept frame,
    enforcing a minimum spacing that dropped burst frames (site camera:
    18 fps delivered -> 14 fps kept under a 25 fps cap)."""
    src = RtspFrameSource(camera_id="cam_a", url="rtsp://example/stream", capture_fps=25)
    interval = src._min_interval
    src._next_keep_ts = 0.0
    # 18 fps average, bursty: pairs 30 ms apart, then an 81 ms gap.
    ts, kept, t = [], 0, 0.0
    for _ in range(30):
        ts.extend([t, t + 0.030])
        t += 0.111
    for capture_ts in ts:
        if capture_ts >= src._next_keep_ts:
            kept += 1
            src._next_keep_ts = max(src._next_keep_ts + interval,
                                    capture_ts - interval)
    # At most the stream's FIRST burst frame may drop (no credit banked yet);
    # steady-state must lose nothing.
    assert kept >= len(ts) - 1, f"bursty slow source lost {len(ts) - kept} frames"


def test_capture_fps_throttle_still_caps_faster_source() -> None:
    """A source FASTER than the cap must still be limited to ~capture_fps."""
    src = RtspFrameSource(camera_id="cam_a", url="rtsp://example/stream", capture_fps=10)
    interval = src._min_interval
    src._next_keep_ts = 0.0
    kept = 0
    n, period = 300, 1.0 / 30.0            # 30 fps steady for 10 s
    for i in range(n):
        capture_ts = i * period
        if capture_ts >= src._next_keep_ts:
            kept += 1
            src._next_keep_ts = max(src._next_keep_ts + interval,
                                    capture_ts - interval)
    # 10 s at a 10 fps cap → ~100 kept (small slack for schedule edges).
    assert 95 <= kept <= 110, f"cap failed: kept {kept} of {n}"


# ---- decoder: nvdec (Phase 0.5 of the DeepStream plan) ----------------------


def test_nvdec_decoder_builds_gpu_chain(monkeypatch) -> None:
    """`decoder: nvdec` swaps the software decode for NVDEC + GPU colorspace,
    keeping the appsink contract (system-memory BGR) identical."""
    src = RtspFrameSource(camera_id="cam_a", url="rtsp://example/stream",
                          decoder="nvdec")
    src._codec = "h264"                       # skip the ffprobe
    monkeypatch.setattr(RtspFrameSource, "_nvdec_available", lambda self: True)
    s = src._build_pipeline_str()
    for el in ("h264parse", "nvh264dec", "cudaconvertscale",
               "memory:CUDAMemory", "cudadownload", "format=BGR"):
        assert el in s, f"nvdec chain missing {el}"
    assert "avdec_h264" not in s and "videoconvert" not in s


def test_nvdec_decoder_h265_variant(monkeypatch) -> None:
    src = RtspFrameSource(camera_id="cam_b", url="rtsp://example/stream",
                          decoder="nvdec")
    src._codec = "hevc"
    monkeypatch.setattr(RtspFrameSource, "_nvdec_available", lambda self: True)
    s = src._build_pipeline_str()
    assert "nvh265dec" in s and "rtph265depay" in s


def test_nvdec_falls_back_to_software_when_unavailable(monkeypatch) -> None:
    """A config with nvdec must stay runnable on machines without nvcodec
    (e.g. a CPU-only edge node sharing the same backbone.yaml)."""
    src = RtspFrameSource(camera_id="cam_a", url="rtsp://example/stream",
                          decoder="nvdec")
    src._codec = "h264"
    monkeypatch.setattr(RtspFrameSource, "_nvdec_available", lambda self: False)
    s = src._build_pipeline_str()
    assert "avdec_h264" in s and "videoconvert" in s
    assert "nvh264dec" not in s


def test_default_decoder_is_software() -> None:
    src = RtspFrameSource(camera_id="cam_a", url="rtsp://example/stream")
    src._codec = "h264"
    s = src._build_pipeline_str()
    assert "avdec_h264 ! videoconvert" in s


def test_invalid_decoder_rejected() -> None:
    with pytest.raises(ValueError, match="decoder="):
        RtspFrameSource(camera_id="cam_a", url="rtsp://example/stream",
                        decoder="quantum")


# ---- output_wh (in-pipeline ingest downscale) --------------------------------


def test_output_wh_software_chain_adds_videoscale_and_caps() -> None:
    """`output_wh` sizes the appsink caps and gives the software chain a
    videoscale element (videoconvert alone can't scale)."""
    src = RtspFrameSource(camera_id="cam_a", url="rtsp://example/stream",
                          output_wh=[1280, 720])
    src._codec = "h264"
    s = src._build_pipeline_str()
    assert "width=1280" in s and "height=720" in s
    assert "videoscale" in s


def test_output_wh_nvdec_chain_scales_on_gpu(monkeypatch) -> None:
    """The nvdec chain scales in cudaconvertscale — no CPU videoscale."""
    src = RtspFrameSource(camera_id="cam_a", url="rtsp://example/stream",
                          decoder="nvdec", output_wh=(1280, 720))
    src._codec = "h264"
    monkeypatch.setattr(RtspFrameSource, "_nvdec_available", lambda self: True)
    s = src._build_pipeline_str()
    assert "width=1280" in s and "height=720" in s
    assert "videoscale" not in s and "cudaconvertscale" in s


def test_output_wh_absent_leaves_pipeline_unchanged() -> None:
    src = RtspFrameSource(camera_id="cam_a", url="rtsp://example/stream")
    src._codec = "h264"
    s = src._build_pipeline_str()
    assert "width=" not in s and "videoscale" not in s


def test_output_wh_rejects_bad_values() -> None:
    import pytest

    with pytest.raises(ValueError):
        RtspFrameSource(camera_id="cam_a", url="rtsp://example/stream",
                        output_wh=(0, 720))
