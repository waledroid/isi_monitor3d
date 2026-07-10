"""Image enhancement for detection crops (the Settings "ENH" switch).

Warehouse crops are often under-lit and low-contrast (dark aisles, backlit
doors). CLAHE on the luminance channel lifts local contrast without blowing
out highlights the way a global histogram equalisation does, and an optional
gamma pass brightens shadows. Both run on the small ZONE CROP (or tile), not
the full frame, so the cost is negligible next to inference.

Applied by ``ZoneScopedDetector`` immediately before letterboxing, so the
model sees exactly what the enhancement produced — and so the published
observations' pixel coordinates are unaffected (enhancement never resizes).
"""

from __future__ import annotations

import cv2
import numpy as np


def enhance_bgr(image: np.ndarray, *, clip_limit: float = 2.0,
                tile_grid: int = 8, gamma: float = 1.0) -> np.ndarray:
    """CLAHE on L (LAB) + optional gamma. Returns a NEW BGR array."""
    if image.size == 0:
        return image
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit),
                            tileGridSize=(int(tile_grid), int(tile_grid)))
    lab = cv2.merge((clahe.apply(l_chan), a_chan, b_chan))
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    if abs(gamma - 1.0) > 1e-3:
        inv = 1.0 / max(0.05, float(gamma))
        lut = np.clip(((np.arange(256) / 255.0) ** inv) * 255.0, 0, 255).astype(np.uint8)
        out = cv2.LUT(out, lut)
    return out
