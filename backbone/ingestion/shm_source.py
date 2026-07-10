"""``ShmFrameSource`` — frames from the shared frame bus, not a camera.

Consumes the frame bus (``backbone/shared/frame_shm.py``) that the perception
process publishes: yields a ``Frame`` whenever a NEW ``capture_ts`` appears in
shared memory. Zero network, zero decode — a memory copy per frame. The
``capture_ts`` is the original writer's clock, so the KPI timeline is shared,
not re-stamped.

Ends (StopIteration) after ``stale_after_s`` without a fresh frame — the
writer is gone; the consumer decides what to do next (the dashboard's camera
hub falls back to its own RTSP session).
"""

from __future__ import annotations

import threading
from collections.abc import Iterator

from backbone.core.interfaces import FrameSource, frame_source_registry
from backbone.core.types import Frame
from backbone.shared.frame_shm import FrameShmReader


@frame_source_registry.register("shm")
class ShmFrameSource(FrameSource):
    """Latest-frame reader over the shared frame bus for one camera."""

    def __init__(self, camera_id: str, *, directory: str | None = None,
                 poll_s: float = 0.005, stale_after_s: float = 2.0) -> None:
        self._camera_id = camera_id
        self._reader = FrameShmReader(camera_id, directory,
                                      max_age_s=stale_after_s)
        self._poll_s = float(poll_s)
        self._stale_after_s = float(stale_after_s)
        self._stop = threading.Event()

    @property
    def camera_id(self) -> str:
        return self._camera_id

    def start(self) -> None:   # lifecycle parity with the live sources
        self._stop.clear()

    def frames(self) -> Iterator[Frame]:
        import time
        last_ts = 0.0
        last_fresh = time.monotonic()
        idx = 0
        while not self._stop.is_set():
            got = self._reader.latest()
            if got is not None and got[1] > last_ts:
                image, ts = got
                last_ts = ts
                last_fresh = time.monotonic()
                yield Frame(camera_id=self._camera_id, capture_ts=ts,
                            frame_idx=idx, image=image)
                idx += 1
                continue
            if time.monotonic() - last_fresh > self._stale_after_s:
                return          # writer gone — let the consumer fall back
            self._stop.wait(self._poll_s)

    def stop(self) -> None:
        self._stop.set()
        self._reader.close()
