"""Canny edge control map — pure cv2, no model, hermetically testable."""

from __future__ import annotations

import cv2
import numpy as np

from .base import CONTROL_MAP_EXTRACTORS, ControlMapExtractor


@CONTROL_MAP_EXTRACTORS.register("canny")
class CannyExtractor(ControlMapExtractor):
    """Grayscale → optional Gaussian blur → cv2.Canny → uint8 edge map."""

    map_name = "canny"

    def __init__(self, low: int = 100, high: int = 200, blur_ksize: int = 3, **cfg) -> None:
        super().__init__(low=low, high=high, blur_ksize=blur_ksize, **cfg)
        self.low = int(low)
        self.high = int(high)
        self.blur_ksize = int(blur_ksize)

    def extract(self, image_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        if self.blur_ksize and self.blur_ksize >= 3:
            k = self.blur_ksize | 1            # kernel must be odd
            gray = cv2.GaussianBlur(gray, (k, k), 0)
        return cv2.Canny(gray, self.low, self.high)
