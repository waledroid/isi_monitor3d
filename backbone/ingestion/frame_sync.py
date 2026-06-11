"""Approximate-time pairing of per-camera frames into ``FramePair``.

Each camera produces ``Frame`` objects independently, with ``capture_ts``
values that should be near-simultaneous but rarely identical. This module
turns N independent streams into one stream of aligned ``FramePair``s.

Algorithm (ROS-style approximate-time policy, single capture-time axis):

    * Per-camera bounded deque of incoming ``Frame``s.
    * **Strict alignment (preferred):** on each new frame, if every camera's
      head is within ``max_skew_ms``, consume the heads → multi-cam pair.
    * **Solo emit (degraded / Mode 1):** if no strict alignment fired AND the
      oldest buffered frame across all cameras has waited more than
      ``degraded_emit_after_ms`` since the latest capture seen, emit it as a
      single-camera ``FramePair``. This covers two cases with one mechanism:
        - Mode 1 deployments where only one camera is configured.
        - Mode 2 deployments where one camera has failed at runtime.
    * **Eviction:** drop frames older than ``max_age_ms`` from the reference
      time so a slow consumer doesn't accumulate unbounded backlog.

The reference time is the latest ``capture_ts`` seen across any camera — NOT
wall clock. If every camera goes silent the pipeline goes quiet rather than
emitting stale frames; systemd handles the catastrophic case.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Iterable

from backbone.core.types import Frame, FramePair


class FrameSynchronizer:
    """Pairs frames from a fixed set of cameras within a skew tolerance."""

    def __init__(
        self,
        camera_ids: Iterable[str],
        max_skew_ms: float = 33.0,
        max_age_ms: float = 1000.0,
        degraded_emit_after_ms: float = 100.0,
        buffer_size: int = 8,
    ) -> None:
        ids = tuple(camera_ids)
        if len(ids) < 1:
            raise ValueError(f"FrameSynchronizer needs >=1 camera, got {ids}")
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate camera_ids in {ids}")
        self._camera_ids = ids
        self._max_skew_s = max_skew_ms / 1000.0
        self._max_age_s = max_age_ms / 1000.0
        self._degraded_after_s = degraded_emit_after_ms / 1000.0
        self._buffers: dict[str, deque[Frame]] = {cid: deque(maxlen=buffer_size) for cid in ids}
        self._lock = threading.Lock()
        self._pair_counter = 0
        self._latest_capture_ts: float = 0.0

    @property
    def camera_ids(self) -> tuple[str, ...]:
        return self._camera_ids

    def submit(self, frame: Frame) -> FramePair | None:
        """Add a frame and try to emit a pair.

        Returns a ``FramePair`` if one is ready (strict-aligned multi-cam OR
        a solo emission for a frame whose partner is overdue), else ``None``.
        """
        if frame.camera_id not in self._buffers:
            return None

        with self._lock:
            self._buffers[frame.camera_id].append(frame)
            if frame.capture_ts > self._latest_capture_ts:
                self._latest_capture_ts = frame.capture_ts
            self._evict_stale(self._latest_capture_ts)
            pair = self._try_emit_aligned()
            if pair is not None:
                return pair
            return self._try_emit_solo()

    def _evict_stale(self, reference_ts: float) -> None:
        """Drop frames older than the configured age threshold."""
        oldest_allowed = reference_ts - self._max_age_s
        for buf in self._buffers.values():
            while buf and buf[0].capture_ts < oldest_allowed:
                buf.popleft()

    def _try_emit_aligned(self) -> FramePair | None:
        """Multi-cam alignment — every camera has a head within ``max_skew_s``.

        Single-camera configurations (``len(camera_ids) == 1``) intentionally
        do **not** match here — they take the solo path, which adds the
        ``degraded_emit_after_ms`` wait so a multi-cam config that happens to
        have only one camera alive doesn't immediately emit single-cam pairs
        on every frame.
        """
        if len(self._camera_ids) < 2:
            return None
        if not all(self._buffers[cid] for cid in self._camera_ids):
            return None

        heads: dict[str, Frame] = {cid: self._buffers[cid][-1] for cid in self._camera_ids}
        timestamps = [f.capture_ts for f in heads.values()]
        skew = max(timestamps) - min(timestamps)
        if skew > self._max_skew_s:
            return None

        self._pair_counter += 1
        pair = FramePair(
            capture_ts=sum(timestamps) / len(timestamps),
            frame_idx=self._pair_counter,
            frames=heads,
        )
        for cid, f in heads.items():
            try:
                self._buffers[cid].remove(f)
            except ValueError:
                pass
        return pair

    def _try_emit_solo(self) -> FramePair | None:
        """Emit the oldest frame as a solo pair if it has waited too long.

        Fires when:
            * The configuration is single-camera (Mode 1) — every buffered
              frame eventually ages past ``degraded_emit_after_s`` relative to
              the latest seen on the same camera.
            * The configuration is multi-camera but one camera has stopped
              feeding (Mode 2 with a failed cam) — the surviving camera's
              oldest frame ages past the threshold and emits solo.

        Subsequent ``submit()`` calls drain the buffer at the input rate.
        """
        for cid in self._camera_ids:
            buf = self._buffers[cid]
            if not buf:
                continue
            age = self._latest_capture_ts - buf[0].capture_ts
            if age >= self._degraded_after_s:
                head = buf.popleft()
                self._pair_counter += 1
                return FramePair(
                    capture_ts=head.capture_ts,
                    frame_idx=self._pair_counter,
                    frames={cid: head},
                )
        return None

    @property
    def buffer_depths(self) -> dict[str, int]:
        with self._lock:
            return {cid: len(buf) for cid, buf in self._buffers.items()}

    @property
    def pairs_emitted(self) -> int:
        return self._pair_counter
