"""``TemporalStabilizer`` — class voting + flicker suppression on tracker output.

The Tracker (``ByteTrackMeters``) emits one ``Track2D`` per confirmed track per
frame, using the latest observation's class label. That label can flicker if
the detector occasionally mis-classifies a frame (e.g. a person briefly read
as "forklift"). The stabilizer fixes this by emitting a **majority-vote**
class over a sliding window of recent observations.

Optionally suppresses tracks that haven't been confirmed for enough frames —
useful when downstream subscribers care about identity stability more than
recall (the architecture's "industrial defaults" principle).

The stabilizer **does not own state**: it reads from the tracker's
``InternalTrack`` objects directly. Pass the tracker into the stabilizer so
it can introspect each emitted ``Track2D``'s underlying internal track.
"""

from __future__ import annotations

from collections import Counter

from backbone.core.types import Track2D

from .bytetrack import ByteTrackMeters
from .track import InternalTrack


class TemporalStabilizer:
    """Re-label and optionally filter ``Track2D`` based on per-track history."""

    def __init__(
        self,
        tracker: ByteTrackMeters,
        *,
        min_frames_confirmed: int = 1,
    ) -> None:
        """Args:
            tracker: the ByteTrack instance whose internal tracks back the
                emitted Track2D list. The stabilizer reads ``class_history``
                from those internal tracks.
            min_frames_confirmed: extra suppression threshold — only emit
                Track2D for tracks whose class_history is at least this long.
                Default 1 (publish every confirmed track).
        """
        self._tracker = tracker
        self._min_frames_confirmed = int(min_frames_confirmed)

    def stabilize(self, tracks: list[Track2D]) -> list[Track2D]:
        """Apply class voting + flicker suppression."""
        if not tracks:
            return []

        internal_by_id = {t.track_id: t for t in self._tracker.active_tracks}
        out: list[Track2D] = []
        for tr in tracks:
            internal = internal_by_id.get(tr.track_id)
            if internal is None:
                # Shouldn't happen if the tracker just emitted it, but be defensive.
                out.append(tr)
                continue
            if len(internal.class_history) < self._min_frames_confirmed:
                continue
            voted_cls = _majority_class(internal)
            if voted_cls == tr.cls:
                out.append(tr)
            else:
                out.append(
                    Track2D(
                        track_id=tr.track_id,
                        cls=voted_cls,
                        capture_ts=tr.capture_ts,
                        xy_m=tr.xy_m,
                        vxy_m=tr.vxy_m,
                        confidence=tr.confidence,
                        cameras_seeing=tr.cameras_seeing,
                    )
                )
        return out


def _majority_class(track: InternalTrack) -> str:
    """Most common class in the track's history; ties broken by recency."""
    counter = Counter(track.class_history)
    most_common = counter.most_common()
    top_count = most_common[0][1]
    tied = [cls for cls, n in most_common if n == top_count]
    if len(tied) == 1:
        return tied[0]
    # Tie-break by most recent occurrence in the history.
    for label in reversed(track.class_history):
        if label in tied:
            return label
    return tied[0]
