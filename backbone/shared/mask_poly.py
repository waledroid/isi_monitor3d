"""Binary instance mask → simplified polygon (for the observations wire).

Full-frame mask bitmaps are far too heavy to publish; a simplified outline of
the largest connected component (``cv2.findContours`` + ``approxPolyDP``) is
a few hundred bytes and renders as a filled path on any canvas. Coordinates
come back in the mask's own space shifted by ``offset_xy`` (the crop origin
for crop-relative masks), i.e. FRAME pixels.
"""

from __future__ import annotations

import cv2
import numpy as np


def mask_to_polygon(mask: np.ndarray, offset_xy: tuple[int, int] | None = None,
                    *, epsilon_px: float = 2.0,
                    max_points: int = 60) -> list[list[float]] | None:
    """Largest-contour simplified polygon of a binary mask, or ``None``.

    ``None`` for empty/degenerate masks (< 3 vertices or ~zero area) — the
    consumer then falls back to the bbox. ``max_points`` bounds the vertex
    count (epsilon escalates until it fits): this is a WIRE-SIZE budget, not
    a quality knob — a display outline never needs hundreds of vertices, and
    every extra point is bytes in a UDP datagram.
    """
    if mask is None or mask.size == 0:
        return None
    m = np.ascontiguousarray(mask.astype(np.uint8))
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 4.0:
        return None
    eps = float(epsilon_px)
    approx = cv2.approxPolyDP(contour, eps, True).reshape(-1, 2)
    while len(approx) > max_points and eps < 64.0:
        eps *= 1.7
        approx = cv2.approxPolyDP(contour, eps, True).reshape(-1, 2)
    if len(approx) < 3:
        return None
    ox, oy = offset_xy if offset_xy is not None else (0, 0)
    return [[float(x + ox), float(y + oy)] for x, y in approx]
