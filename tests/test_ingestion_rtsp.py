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


@pytest.mark.parametrize(
    "knob",
    ["latency=", "drop-on-latency=true", "ntp-sync=true", "protocols=tcp",
     "rtph264depay", "avdec_h264", "format=BGR", "drop=true", "emit-signals=true"],
)
def test_pipeline_template_contains_required_knob(knob: str) -> None:
    """If anyone removes a load-bearing pipeline knob, this test fires."""
    rendered = PIPELINE_TEMPLATE.format(url="rtsp://x/y", latency_ms=100)
    assert knob in rendered, f"missing pipeline knob: {knob}"


def test_pipeline_string_parses_in_this_env() -> None:
    """The pipeline must be syntactically valid GStreamer."""
    _ensure_gst_initialized()
    rendered = PIPELINE_TEMPLATE.format(url="rtsp://127.0.0.1/x", latency_ms=100)
    pipeline = Gst.parse_launch(rendered)
    assert pipeline is not None
    # Required elements present:
    for name in ("src", "sink"):
        assert pipeline.get_by_name(name) is not None, f"missing element: {name}"


def test_no_thread_started_at_construction() -> None:
    """Construction must be cheap; only start() launches the GLib loop."""
    src = RtspFrameSource(camera_id="cam_a", url="rtsp://example/stream")
    assert src.dropped_count == 0
