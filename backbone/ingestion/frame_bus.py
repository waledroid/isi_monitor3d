"""In-process bounded queue for ``FramePair`` objects.

Decouples ingestion threads from the downstream pipeline with one bounded
queue and drop-old semantics: if the pipeline lags, the oldest pending
``FramePair`` is evicted to keep latency honest. The publisher (the
synchronizer) never blocks.

v1 has exactly one consumer (the orchestrator's pipeline thread). The bus
intentionally stays single-subscriber — if a second consumer ever materializes
(a dashboard preview thread, a recording sidecar), wrap this bus in a fan-out
adapter rather than re-introducing multi-subscriber plumbing here.
"""

from __future__ import annotations

import logging
import queue
import threading

from backbone.core.types import FramePair

logger = logging.getLogger(__name__)


class FrameBus:
    """Bounded drop-old queue. Single producer, single consumer."""

    def __init__(self, default_maxsize: int = 2) -> None:
        self._queue: queue.Queue[FramePair] = queue.Queue(maxsize=default_maxsize)
        self._lock = threading.Lock()
        self._closed = False
        self._dropped = 0

    @property
    def dropped(self) -> int:
        """Monotonic count of items evicted because the consumer was slow."""
        return self._dropped

    @property
    def depth(self) -> int:
        return self._queue.qsize()

    def publish(self, item: FramePair) -> None:
        """Deliver ``item`` to the consumer, dropping the oldest pending if full."""
        if self._closed:
            return
        try:
            self._queue.put_nowait(item)
            return
        except queue.Full:
            pass
        with self._lock:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._dropped += 1
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                logger.warning("FrameBus full after eviction; item lost")

    def get(self, timeout: float | None = None) -> FramePair:
        return self._queue.get(timeout=timeout)

    def get_nowait(self) -> FramePair | None:
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def close(self) -> None:
        self._closed = True
