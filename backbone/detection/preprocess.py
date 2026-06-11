"""YOLO11 detector preprocessing: letterbox → normalize → CHW tensor.

Pure NumPy. No PyTorch, no OpenCV image processing — we do everything by
hand so the postprocess step can invert the exact transform we applied. The
inverse mapping is stored alongside the tensor in :class:`LetterboxResult`.

Conventions:
    * Input image: ``(H, W, 3)`` ``uint8`` BGR (OpenCV / ``Frame.image`` convention).
    * Output tensor: ``(3, target_h, target_w)`` ``float32`` RGB in ``[0, 1]``.
    * Letterbox pads with neutral grey (``114, 114, 114``) on the short side,
      preserving aspect ratio — same default Ultralytics' export uses, so a
      pretrained ``yolo11n.onnx`` trained at 640 sees the right pixel distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

LETTERBOX_PAD_VALUE: int = 114
"""Neutral grey used by Ultralytics' default training. Don't change without retraining."""


@dataclass(slots=True)
class LetterboxResult:
    """A letterboxed tensor plus the inverse transform needed by postprocess.

    ``scale`` is the single scalar applied to both axes (aspect-preserving).
    ``pad_xy`` is ``(pad_x, pad_y)`` in target-frame pixels — the number of pad
    pixels on the *left* and *top* edges. Postprocess maps a bbox detected in
    target-frame coordinates back to source-frame coordinates as:

        x_source = (x_target - pad_x) / scale
        y_source = (y_target - pad_y) / scale
    """

    tensor: np.ndarray              # (3, H, W) float32, RGB, [0, 1]
    scale: float                    # uniform resize factor source → target
    pad_xy: tuple[int, int]         # (pad_x, pad_y) in target pixels (left + top)
    source_shape_hw: tuple[int, int]  # original (H, W) for sanity checks


def letterbox(
    image: np.ndarray,
    target: tuple[int, int] = (640, 640),
) -> LetterboxResult:
    """Aspect-preserving resize-and-pad into a square target.

    Args:
        image: ``(H, W, 3)`` uint8 BGR.
        target: ``(target_h, target_w)``. Must be equal in the typical YOLO case.

    Returns:
        :class:`LetterboxResult` with the model-ready CHW float32 tensor and
        the inverse-transform metadata.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"letterbox expects (H, W, 3) BGR image, got shape {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"letterbox expects uint8 image, got dtype {image.dtype}")

    src_h, src_w = image.shape[:2]
    target_h, target_w = target

    # Single scale factor — preserves aspect ratio.
    scale = min(target_w / src_w, target_h / src_h)
    new_w = round(src_w * scale)
    new_h = round(src_h * scale)

    # INTER_AREA is the correct filter for shrinking (avoids aliasing); INTER_LINEAR
    # for enlarging. scale < 1 means we're downscaling the feed to the model size
    # (the common case: a 1080p/720p feed into a 320-1024 model input).
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)

    # Pad to (target_h, target_w), centred.
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    pad_x_right = target_w - new_w - pad_x
    pad_y_bottom = target_h - new_h - pad_y

    padded = cv2.copyMakeBorder(
        resized,
        pad_y,
        pad_y_bottom,
        pad_x,
        pad_x_right,
        cv2.BORDER_CONSTANT,
        value=(LETTERBOX_PAD_VALUE, LETTERBOX_PAD_VALUE, LETTERBOX_PAD_VALUE),
    )

    # BGR -> RGB, HWC -> CHW, uint8 -> float32 / 255.0
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32, copy=False).transpose(2, 0, 1) / 255.0

    return LetterboxResult(
        tensor=tensor,
        scale=scale,
        pad_xy=(pad_x, pad_y),
        source_shape_hw=(src_h, src_w),
    )


def batch_letterbox(
    images: list[np.ndarray],
    target: tuple[int, int] = (640, 640),
) -> tuple[np.ndarray, list[LetterboxResult]]:
    """Letterbox every image and stack into one batched tensor.

    Returns:
        ``(batch_tensor (N, 3, H, W), per_image_results)``. The per-image
        results carry the inverse-transform metadata postprocess needs.
    """
    if not images:
        raise ValueError("batch_letterbox needs at least one image")
    results = [letterbox(img, target) for img in images]
    batch = np.stack([r.tensor for r in results], axis=0)
    return batch, results


def invert_letterbox_xyxy(
    boxes_target_xyxy: np.ndarray,
    lb: LetterboxResult,
) -> np.ndarray:
    """Map ``(N, 4)`` boxes from target-frame back to source-frame coordinates.

    Boxes are clipped to the source image bounds — the model occasionally
    returns boxes that extend slightly into the letterbox padding, and we
    don't want those clipped pixels to be a problem downstream.
    """
    if boxes_target_xyxy.size == 0:
        return boxes_target_xyxy.copy()
    pad_x, pad_y = lb.pad_xy
    src_h, src_w = lb.source_shape_hw
    out = boxes_target_xyxy.astype(np.float32, copy=True)
    out[:, [0, 2]] = (out[:, [0, 2]] - pad_x) / lb.scale
    out[:, [1, 3]] = (out[:, [1, 3]] - pad_y) / lb.scale
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0.0, src_w - 1)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0.0, src_h - 1)
    return out
