"""``Publisher`` — fan ``Track2D`` / ``Track3D`` to one or more ``MetadataSink``s.

Concrete, single-implementation. Sits between the runtime pipeline and the
sink plugins (`udp`, future `mqtt`, etc.). One process publishes to one or
more sinks; the orchestrator builds and owns the Publisher.

Errors in one sink don't suppress publication to the others — a UDP socket
write failing should never break the loop. Errors are logged at WARNING.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

from backbone.core.interfaces import MetadataSink
from backbone.core.types import Track2D, Track3D

logger = logging.getLogger(__name__)


class Publisher:
    """Fan-out of ``Track2D`` / ``Track3D`` to multiple ``MetadataSink``s."""

    def __init__(self, sinks: Iterable[MetadataSink]) -> None:
        self._sinks: tuple[MetadataSink, ...] = tuple(sinks)
        self._closed = False

    @property
    def sinks(self) -> tuple[MetadataSink, ...]:
        return self._sinks

    def publish_track_2d(self, track: Track2D) -> None:
        if self._closed:
            return
        for sink in self._sinks:
            try:
                sink.publish_track_2d(track)
            except Exception:
                logger.warning("sink %s failed on track_2d", type(sink).__name__, exc_info=True)

    def publish_track_3d(self, track: Track3D) -> None:
        if self._closed:
            return
        for sink in self._sinks:
            try:
                sink.publish_track_3d(track)
            except Exception:
                logger.warning("sink %s failed on track_3d", type(sink).__name__, exc_info=True)

    def publish_event(self, event: object) -> None:
        """Fan-out a ``PassingEvent`` to all sinks via ``publish_event``."""
        if self._closed:
            return
        for sink in self._sinks:
            try:
                sink.publish_event(event)
            except Exception:
                logger.warning("sink %s failed on publish_event", type(sink).__name__, exc_info=True)

    def publish_image_ref(
        self,
        track_id: int,
        cls: str,
        zone: str,
        ts: float,
        url: str,
    ) -> None:
        """Fan-out an image-reference URL to all sinks via ``publish_image_ref``."""
        if self._closed:
            return
        for sink in self._sinks:
            try:
                sink.publish_image_ref(track_id, cls, zone, ts, url)
            except Exception:
                logger.warning(
                    "sink %s failed on publish_image_ref", type(sink).__name__, exc_info=True
                )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for sink in self._sinks:
            try:
                sink.close()
            except Exception:
                logger.warning("sink %s failed on close", type(sink).__name__, exc_info=True)
