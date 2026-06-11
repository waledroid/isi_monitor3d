"""``ReplayFrameSource`` — read frames from memory or a recorded MP4 + sidecar JSON.

Used by:
    * Unit tests (hermetic, in-memory frames with controlled timestamps).
    * Dev sessions when no live camera is attached — record once on-site
      via ``tools/rtsp_record.py`` and replay locally.

Plugin name: ``replay``.

Two construction modes:

    >>> # In-memory: a list of (image, capture_ts) pairs.
    >>> src = ReplayFrameSource(camera_id="cam_a", frames=[(img1, 0.0), (img2, 0.033)])

    >>> # MP4 + timestamps.json: the JSON is a flat list of floats, one per frame.
    >>> src = ReplayFrameSource(
    ...     camera_id="cam_a",
    ...     mp4_path="recordings/cam_a.mp4",
    ...     timestamps_json="recordings/cam_a.timestamps.json",
    ... )
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from backbone.core.interfaces import FrameSource, frame_source_registry
from backbone.core.types import Frame


@frame_source_registry.register("replay")
class ReplayFrameSource(FrameSource):
    """Replay frames either from in-memory fixtures or from disk."""

    def __init__(
        self,
        camera_id: str,
        *,
        frames: list[tuple[np.ndarray, float]] | None = None,
        mp4_path: str | Path | None = None,
        timestamps_json: str | Path | None = None,
    ) -> None:
        self._camera_id = camera_id
        if frames is not None:
            self._frames: list[tuple[np.ndarray, float]] = list(frames)
        elif mp4_path is not None and timestamps_json is not None:
            self._frames = _load_mp4_with_timestamps(Path(mp4_path), Path(timestamps_json))
        else:
            raise ValueError(
                "ReplayFrameSource needs either frames=[...] or "
                "(mp4_path=..., timestamps_json=...) — got neither"
            )
        self._stop_event = threading.Event()

    @property
    def camera_id(self) -> str:
        return self._camera_id

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    def frames(self) -> Iterator[Frame]:
        """Yield ``Frame`` objects in capture-time order until exhausted or stopped."""
        for idx, (image, capture_ts) in enumerate(self._frames):
            if self._stop_event.is_set():
                break
            yield Frame(
                camera_id=self._camera_id,
                capture_ts=capture_ts,
                frame_idx=idx,
                image=image,
            )

    def stop(self) -> None:
        self._stop_event.set()


def _load_mp4_with_timestamps(
    mp4_path: Path,
    timestamps_json: Path,
) -> list[tuple[np.ndarray, float]]:
    """Read an MP4 + per-frame timestamps file into a list of (image, ts) pairs.

    The sidecar JSON is a flat list of floats — one capture timestamp per
    decoded frame, in the same order the codec emits them. Mismatched lengths
    raise immediately because silently truncating would mis-align timestamps.
    """
    import cv2  # imported lazily so unit tests of the in-memory path don't need it

    if not mp4_path.exists():
        raise FileNotFoundError(f"MP4 not found: {mp4_path}")
    if not timestamps_json.exists():
        raise FileNotFoundError(f"timestamps sidecar not found: {timestamps_json}")

    timestamps_raw = json.loads(timestamps_json.read_text())
    if not isinstance(timestamps_raw, list):
        raise ValueError(f"{timestamps_json}: expected JSON list of floats")
    timestamps = [float(t) for t in timestamps_raw]

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {mp4_path}")

    out: list[tuple[np.ndarray, float]] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            out.append((frame, 0.0))  # placeholder ts, fixed below
    finally:
        cap.release()

    if len(out) != len(timestamps):
        raise ValueError(
            f"{mp4_path}: decoded {len(out)} frames but {timestamps_json} has "
            f"{len(timestamps)} timestamps"
        )
    return [(img, ts) for (img, _), ts in zip(out, timestamps, strict=True)]
