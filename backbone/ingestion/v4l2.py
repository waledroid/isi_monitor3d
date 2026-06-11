"""``V4l2FrameSource`` — USB / UVC camera capture via GStreamer ``v4l2src``.

A locally-attached camera (USB webcam, USB capture card) appears on Linux as a
V4L2 device node ``/dev/videoN``. This source drives it with the same
GStreamer-appsink machinery as the RTSP source (see
``backbone.ingestion._gst_source``); only the pipeline differs.

Pipeline:

    ``v4l2src device=/dev/video0 ! decodebin ! videoconvert
      ! video/x-raw,format=BGR ! appsink name=sink …``

``decodebin`` is deliberately used instead of a fixed decoder: UVC cameras
expose either raw frames (YUYV) or MJPEG, and ``decodebin`` auto-negotiates
both. When ``width`` / ``height`` / ``fps`` are supplied they are pushed onto
``v4l2src`` as a capsfilter so the camera is asked for that mode (subject to
what the device actually supports).

Live verification requires real hardware reachable as ``/dev/video*``; on WSL2
the device must first be attached with ``usbipd-win``. Unit tests cover the
pipeline-string shape and the buffer→Frame conversion with a mocked sample.
"""

from __future__ import annotations

import logging

from backbone.core.interfaces import frame_source_registry

from ._gst_source import GstAppsinkFrameSource

__all__ = ["V4l2FrameSource"]

logger = logging.getLogger(__name__)


@frame_source_registry.register("v4l2")
class V4l2FrameSource(GstAppsinkFrameSource):
    """USB / UVC camera as a Backbone ``FrameSource``."""

    def __init__(
        self,
        camera_id: str,
        device: str = "/dev/video0",
        *,
        width: int | None = None,
        height: int | None = None,
        fps: int | None = None,
        startup_timeout_s: float = 10.0,
    ) -> None:
        if not isinstance(device, str) or not device:
            raise ValueError(
                f"V4l2FrameSource: bad device {device!r}, expected a path like /dev/video0"
            )
        super().__init__(camera_id, startup_timeout_s=startup_timeout_s)
        self._device = device
        self._width = int(width) if width else None
        self._height = int(height) if height else None
        self._fps = int(fps) if fps else None

    def _source_caps(self) -> str:
        """Optional capsfilter applied right after ``v4l2src`` (empty if unset)."""
        if not (self._width or self._height or self._fps):
            return ""
        fields = []
        if self._width:
            fields.append(f"width={self._width}")
        if self._height:
            fields.append(f"height={self._height}")
        if self._fps:
            fields.append(f"framerate={self._fps}/1")
        # decodebin negotiates the codec; we only constrain geometry/rate.
        return "! video/x-raw," + ",".join(fields) + " "

    def _build_pipeline_str(self) -> str:
        return (
            f"v4l2src name=src device={self._device} "
            f"{self._source_caps()}"
            "! decodebin "
            "! videoconvert "
            "! video/x-raw,format=BGR "
            "! appsink name=sink emit-signals=true sync=false max-buffers=1 drop=true"
        )
