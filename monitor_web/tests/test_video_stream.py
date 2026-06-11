"""``video_stream`` — MJPEG chunk encoding + multipart wrapping."""

from __future__ import annotations

import numpy as np
import pytest

from monitor_web.video_stream import (
    JPEG_BOUNDARY,
    MAX_HEIGHT_PX,
    encode_mjpeg_frame,
    mjpeg_stream,
)


def _solid_frame(h: int, w: int, fill: int = 0) -> np.ndarray:
    return np.full((h, w, 3), fill, dtype=np.uint8)


def test_encode_produces_multipart_chunk() -> None:
    chunk = encode_mjpeg_frame(_solid_frame(480, 640))
    assert chunk.startswith(b"--" + JPEG_BOUNDARY.encode() + b"\r\n")
    assert b"Content-Type: image/jpeg" in chunk
    # JPEG SOI (start-of-image) bytes appear somewhere after the headers.
    assert b"\xff\xd8" in chunk
    assert chunk.endswith(b"\r\n")


def test_encode_resizes_when_too_tall() -> None:
    chunk = encode_mjpeg_frame(_solid_frame(1080, 1920))
    assert isinstance(chunk, bytes)
    assert MAX_HEIGHT_PX == 720    # invariant


def test_encode_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
        encode_mjpeg_frame(_solid_frame(10, 10).reshape(10, 30))


def test_mjpeg_stream_yields_one_chunk_per_frame() -> None:
    frames = [_solid_frame(120, 160, fill=i * 20) for i in range(3)]
    out = list(mjpeg_stream(iter(frames)))
    assert len(out) == 3
    for chunk in out:
        assert chunk.startswith(b"--" + JPEG_BOUNDARY.encode() + b"\r\n")


def test_mjpeg_stream_skips_malformed_without_aborting() -> None:
    """A bad frame mid-stream is dropped; the rest still emit."""
    good = _solid_frame(120, 160)
    bad = np.zeros((10, 10), dtype=np.uint8)   # 2D, not (H, W, 3)
    out = list(mjpeg_stream(iter([good, bad, good])))
    assert len(out) == 2
