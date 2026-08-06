"""Approximate-time pairing of per-camera frames into ``FramePair``.

Each camera produces ``Frame`` objects independently, with ``capture_ts``
values that should be near-simultaneous but rarely identical. This module
turns N independent streams into one stream of aligned ``FramePair``s.

Algorithm (ROS-style approximate-time policy, single capture-time axis):

    * Per-camera bounded deque of incoming ``Frame``s.
    * **Strict alignment (preferred):** on each new frame, if every camera's
      head is within ``max_skew_ms``, consume the heads → multi-cam pair.
    * **Solo emit (degraded / Mode 1):** LATEST-FRAME-ONLY via a sticky
      per-camera *degraded* flag. A camera ENTERS degraded when its oldest
      buffered frame has waited more than ``degraded_emit_after_ms`` since the
      latest capture seen (the partner is overdue); at that moment the NEWEST
      buffered frame emits as a single-camera ``FramePair`` and everything
      older is discarded — never an oldest-first backlog drain (the old
      behaviour re-served up to ``max_age_ms`` of stale frames when the
      consumer was slow). WHILE degraded, every subsequent frame emits
      immediately (full input fps, zero buffering). The flag clears when a
      strict alignment fires again (partner recovered). Two cases, one
      mechanism:
        - Mode 1 (single camera configured): permanently degraded from
          construction — every frame emits immediately, no 100 ms tax.
        - Mode 2 with a camera failed at runtime: pays the wait once at
          entry, then streams the survivor in real time.
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
        # Sticky degraded flag (see module docstring). A single-camera config
        # is degraded from the start: there is no partner to wait for, so
        # every frame emits immediately.
        self._degraded: dict[str, bool] = {cid: len(ids) == 1 for cid in ids}
        # All-degraded recovery probe: once EVERY camera is sticky-degraded,
        # solo emission clears the buffers on each submit, so the aligned
        # path never sees two heads at once and the flags can never clear.
        # Every probe interval, one camera is un-degraded so it buffers
        # briefly and gives alignment a chance to re-form; a truly dead
        # partner just re-degrades after ``degraded_emit_after_ms``.
        self._probe_interval_s = self._degraded_after_s * 10.0
        self._last_probe_ts = 0.0
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
        # An aligned pair means the partner is back — leave degraded mode.
        for cid in self._camera_ids:
            self._degraded[cid] = False
        return pair

    def _try_emit_solo(self) -> FramePair | None:
        """Latest-only solo emission (see module docstring).

        A camera already in degraded mode emits its NEWEST buffered frame
        immediately (steady state: the frame just submitted — zero latency,
        zero buffering). A camera not yet degraded enters the mode when its
        oldest frame has aged past ``degraded_emit_after_s`` (partner overdue):
        the NEWEST frame emits and every older one is discarded, so a slow
        consumer or a dead partner never causes a stale-frame backlog drain.
        Emitting newest-only keeps ``capture_ts`` monotonic per camera.
        """
        if (len(self._camera_ids) >= 2
                and all(self._degraded.values())
                and self._latest_capture_ts - self._last_probe_ts
                >= self._probe_interval_s):
            self._last_probe_ts = self._latest_capture_ts
            self._degraded[self._camera_ids[0]] = False
        for cid in self._camera_ids:
            buf = self._buffers[cid]
            if not buf:
                continue
            if not self._degraded[cid]:
                age = self._latest_capture_ts - buf[0].capture_ts
                if age < self._degraded_after_s:
                    continue
                self._degraded[cid] = True
            newest = buf[-1]
            buf.clear()
            self._pair_counter += 1
            return FramePair(
                capture_ts=newest.capture_ts,
                frame_idx=self._pair_counter,
                frames={cid: newest},
            )
        return None

    @property
    def buffer_depths(self) -> dict[str, int]:
        with self._lock:
            return {cid: len(buf) for cid, buf in self._buffers.items()}

    @property
    def pairs_emitted(self) -> int:
        return self._pair_counter
