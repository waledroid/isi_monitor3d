import cv2
import numpy as np
import logging
from src.shared.registry import PREPROCESSORS

logger = logging.getLogger(__name__)

@PREPROCESSORS.register('specular-guard')
class SpecularGuard:
    """
    Enhances image lightness in LAB color space to preserve actual parcel colors
    while lifting deep shadows and stabilizing polybag glare.
    """
    def __init__(self, clip_limit: float = 2.5, tile_grid: list = [8, 8]):
        # Convert list to tuple for OpenCV
        grid_tuple = tuple(tile_grid)
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_tuple)
        logger.info(f"🛡️ SpecularGuard initialized (Clip: {clip_limit}, Grid: {grid_tuple})")

    def process(self, image: np.ndarray) -> np.ndarray:
        """Applies the color-safe enhancement to a single BGR image."""
        if image is None:
            raise ValueError("Received empty image array.")
            
        # 1. Convert to LAB for color-safe processing
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # 2. Apply CLAHE to the L (Lightness) channel only
        l_enhanced = self.clahe.apply(l)

        # 3. Reconstruct and return the BGR image
        enhanced_lab = cv2.merge((l_enhanced, a, b))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
