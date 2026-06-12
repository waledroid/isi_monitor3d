"""MJPEG re-streamer — `<img src="/stream/video/cam_a">` in the browser.

For each requested camera, opens an ``RtspFrameSource`` from the Backbone
package (re-used as a consumer-side library) and encodes each frame to JPEG
on the fly. The browser handles MJPEG natively via ``multipart/x-mixed-replace``.

Down-scaling: target ~720p for the dashboard. The Hikvision sub-stream is
already ~1080p; we resize before JPEG-encoding to keep the dashboard's
bandwidth bounded (~1.5-3 Mbps per camera at JPEG quality 75).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import cv2
import numpy as np

logger = logging.getLogger(__name__)

JPEG_BOUNDARY = "frame"
JPEG_QUALITY = 75
MAX_HEIGHT_PX = 720


def encode_jpeg(image: np.ndarray) -> bytes:
    """Encode a BGR ``(H, W, 3)`` ndarray as bare JPEG bytes (≤720p downscale).
    Shared by the MJPEG multipart wrapper below and the /ws/video binary frames."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) BGR, got {image.shape}")
    if image.shape[0] > MAX_HEIGHT_PX:
        new_h = MAX_HEIGHT_PX
        new_w = round(image.shape[1] * (new_h / image.shape[0]))
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


def encode_mjpeg_frame(image: np.ndarray) -> bytes:
    """Encode a BGR ``(H, W, 3)`` ndarray as one multipart MJPEG chunk."""
    payload = encode_jpeg(image)
    return (
        b"--" + JPEG_BOUNDARY.encode() + b"\r\n"
        b"Content-Type: image/jpeg\r\n"
        b"Content-Length: " + str(len(payload)).encode() + b"\r\n\r\n"
        + payload + b"\r\n"
    )


def mjpeg_stream(frame_iterator: Iterator[np.ndarray]) -> Iterator[bytes]:
    """Wrap an iterator of BGR frames into a generator of multipart MJPEG chunks.

    Caller (the FastAPI route) plugs this into ``StreamingResponse(...,
    media_type="multipart/x-mixed-replace; boundary=frame")``.
    """
    for frame in frame_iterator:
        try:
            yield encode_mjpeg_frame(frame)
        except (ValueError, RuntimeError):
            logger.warning("mjpeg_stream: dropped a malformed frame", exc_info=True)
            continue
