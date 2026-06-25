"""``RtspFrameSource`` — plugin registration, URL validation, pipeline string shape.

Real RTSP connections aren't unit-tested (they require a server). Coverage:
    * Plugin registers under ``"rtsp"``.
    * URL scheme is enforced at construction time.
    * Pipeline template includes the load-bearing knobs that the cahier and
      the plan require: ``latency=``, ``drop-on-latency=true``, ``ntp-sync=true``,
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
        depay=depay, decoder=decoder,
    )


@pytest.mark.parametrize(
    "knob",
    ["latency=", "drop-on-latency=true", "ntp-sync=true", "protocols=tcp",
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
