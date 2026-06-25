"""``SnapshotWriter`` — write a JPEG snapshot on a zone-passing event.

Best-effort: write failures are logged and return ``None``; they never
propagate into the pipeline.  Raw bytes are written to disk only; the
returned URL (not the bytes) is what goes on the bus.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Characters unsafe for filenames: keep only word chars, dash, and dot.
_UNSAFE_RE = re.compile(r"[^\w\-.]")


def _sanitize_zone(zone: str) -> str:
    """Replace filesystem-unsafe chars (including ``/``) with ``_``."""
    return _UNSAFE_RE.sub("_", zone)


class SnapshotWriter:
    """Write JPEG snapshots to ``out_dir`` and return a URL.

    Args:
        out_dir:      Directory where JPEG files are written.  Created on
                      construction (``parents=True``, ``exist_ok=True``).
        url_base:     Prefix for the returned URL; trailing ``/`` is stripped.
                      Default ``"file://"`` produces local file URLs.
        jpeg_quality: JPEG quality 0-100.  Passed to ``cv2.imwrite``.
    """

    def __init__(
        self,
        out_dir: str,
        url_base: str = "file://",
        jpeg_quality: int = 85,
    ) -> None:
        self._out_dir = Path(out_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._url_base = url_base.rstrip("/")
        self._encode_params = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write(
        self,
        image: np.ndarray,
        track_id: int,
        zone: str,
        ts: float,
    ) -> str | None:
        """Write ``image`` as a JPEG and return its URL.

        The filename is deterministic: ``{ts_ms}_{zone}_{track_id}.jpg``.

        Args:
            image:    BGR ``np.ndarray`` (the current camera frame).
            track_id: Numeric track identifier.
            zone:     Zone name (sanitised before use in the filename).
            ts:       Capture timestamp in Unix seconds.

        Returns:
            The full URL string on success, or ``None`` on any failure
            (imwrite returns ``False``, or an exception is raised).
        """
        try:
            safe_zone = _sanitize_zone(zone)
            filename = f"{int(ts * 1000)}_{safe_zone}_{track_id}.jpg"
            dest = self._out_dir / filename
            ok = cv2.imwrite(str(dest), image, self._encode_params)
            if not ok:
                logger.warning(
                    "SnapshotWriter: cv2.imwrite returned False for %s", dest
                )
                return None
            return f"{self._url_base}/{filename}"
        except Exception:
            logger.warning(
                "SnapshotWriter: failed to write snapshot "
                "(track_id=%s, zone=%r, ts=%.3f)",
                track_id,
                zone,
                ts,
                exc_info=True,
            )
            return None
