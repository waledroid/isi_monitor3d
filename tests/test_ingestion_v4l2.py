"""``V4l2FrameSource`` — registration, device validation, pipeline string shape.

Real V4L2 capture isn't unit-tested (it requires a ``/dev/video*`` device that
WSL2/CI lack). Coverage mirrors ``test_ingestion_rtsp.py``:
    * Plugin registers under ``"v4l2"``.
    * Device argument is validated at construction.
    * Pipeline string contains the load-bearing elements and the configured
      device, with optional geometry caps when width/height/fps are given.
    * The pipeline string parses in this env (``Gst.parse_launch``).
"""

from __future__ import annotations

import gi
import pytest

from backbone.core.interfaces import frame_source_registry
from backbone.ingestion._gst_source import _ensure_gst_initialized
from backbone.ingestion.v4l2 import V4l2FrameSource

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402


def test_plugin_registered_under_v4l2() -> None:
    assert "v4l2" in frame_source_registry


def test_device_validated() -> None:
    with pytest.raises(ValueError, match="device"):
        V4l2FrameSource(camera_id="cam_usb", device="")


def test_default_device() -> None:
    src = V4l2FrameSource(camera_id="cam_usb")
    assert src.camera_id == "cam_usb"
    assert "/dev/video0" in src._build_pipeline_str()


@pytest.mark.parametrize(
    "element",
    ["v4l2src", "device=/dev/video2", "decodebin", "videoconvert",
     "format=BGR", "appsink", "drop=true", "emit-signals=true"],
)
def test_pipeline_contains_required_element(element: str) -> None:
    src = V4l2FrameSource(camera_id="cam_usb", device="/dev/video2")
    rendered = src._build_pipeline_str()
    assert element in rendered, f"missing pipeline element: {element}"


def test_pipeline_includes_caps_when_geometry_given() -> None:
    src = V4l2FrameSource(camera_id="cam_usb", device="/dev/video0",
                          width=1280, height=720, fps=30)
    rendered = src._build_pipeline_str()
    assert "width=1280" in rendered
    assert "height=720" in rendered
    assert "framerate=30/1" in rendered


def test_pipeline_omits_caps_when_no_geometry() -> None:
    src = V4l2FrameSource(camera_id="cam_usb", device="/dev/video0")
    rendered = src._build_pipeline_str()
    assert "width=" not in rendered
    assert "framerate=" not in rendered


def test_pipeline_string_parses_in_this_env() -> None:
    """The pipeline must be syntactically valid GStreamer (elements may not all
    be present on a box without a camera, but parse must succeed)."""
    _ensure_gst_initialized()
    src = V4l2FrameSource(camera_id="cam_usb", device="/dev/video0")
    pipeline = Gst.parse_launch(src._build_pipeline_str())
    assert pipeline is not None
    assert pipeline.get_by_name("sink") is not None


def test_no_thread_started_at_construction() -> None:
    src = V4l2FrameSource(camera_id="cam_usb", device="/dev/video0")
    assert src.dropped_count == 0
