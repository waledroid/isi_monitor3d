"""Capture inference-identical zone crops as YOLO background images.

Saves gray polygon-filled zone crops — exactly the pixels ZoneScopedDetector
feeds the model — so an empty scene (e.g. the flat wooden pallet support) can
be added to a dataset as hard negatives.

Merge procedure (after HUMAN review of every crop):
  1. Delete any crop containing ANY instance of ANY class — an unlabeled
     object in a background image teaches the model to miss that class.
  2. Copy ~90% into <dataset>/images/train/ and ~10% into images/val/ with
     NO label files (YOLO's background convention; labels/ stays untouched).
  3. Retrain. Start ~250 backgrounds (5% of train), ceiling ~500 (10%).

Usage:
  conda activate monitor3d
  python tools/capture_zone_bg.py --config config/backbone.yaml \
      [--out trainer/isidet/data/bg_captures] [--prefix bg] [--interval 2.0] \
      [--count 300] [--min-diff 4.0] [--cams cam_a,cam_b]

Frames come from the /dev/shm frame bus when isistream is running (zero extra
RTSP session, zero GPU); otherwise the tool opens its own RTSP session with
SOFTWARE decode. Run `--prefix pos` sessions with palettes present to also
collect in-domain positives (label those via the LabelMe flow).
"""
from __future__ import annotations

import argparse
import logging
import math
import re
import signal
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from backbone.detection.zone_scope import _FILL_GRAY  # noqa: F401  (re-exported for tests)

logger = logging.getLogger("capture_zone_bg")


def scale_box(box, calib_wh, frame_wh):
    """Calibration-frame box → actual-frame box: ``(fx0, fy0, fx1, fy1, sx, sy)``.

    Same arithmetic as ``ZoneScopedDetector.detect`` (zone_scope.py) so the
    saved crop covers exactly the pixels the detector sees on a possibly
    ingest-downscaled frame.
    """
    x0, y0, x1, y1 = box
    calib_w, calib_h = calib_wh
    fw, fh = frame_wh
    sx, sy = fw / calib_w, fh / calib_h
    fx0, fy0 = max(0, int(x0 * sx)), max(0, int(y0 * sy))
    fx1, fy1 = min(fw, math.ceil(x1 * sx)), min(fh, math.ceil(y1 * sy))
    return fx0, fy0, fx1, fy1, sx, sy


def fill_crop(crop_img, fill, sx, sy, fx0, fy0):
    """Gray-out crop pixels outside the (dilated) zone polygon — a COPY.

    Mirrors ``ZoneScopedDetector._fill_outside`` (minus its cache): polygon is
    calibration-frame px, scaled by (sx, sy) and shifted by the crop origin.
    """
    poly_px, dilate_px = fill
    ch, cw = crop_img.shape[:2]
    pts = np.round(poly_px * (sx, sy) - (fx0, fy0)).astype(np.int32)
    inside = np.zeros((ch, cw), np.uint8)
    cv2.fillPoly(inside, [pts], 255)
    r = max(1, round(dilate_px * (sx + sy) / 2.0))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    inside = cv2.dilate(inside, kernel)
    out = crop_img.copy()
    out[inside == 0] = _FILL_GRAY
    return out


class CropDeduper:
    """Save a crop only when it visibly differs from the LAST SAVED one.

    Signature = 64×64 grayscale; difference = mean absolute pixel delta.
    A static empty zone then costs one file, not one per interval.
    """

    def __init__(self, min_diff: float = 4.0) -> None:
        self._min = float(min_diff)
        self._last: dict[tuple[str, str], np.ndarray] = {}

    def should_save(self, cam_id: str, zone_name: str, crop: np.ndarray) -> bool:
        sig = cv2.resize(
            cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (64, 64)).astype(np.float32)
        last = self._last.get((cam_id, zone_name))
        if last is not None and float(np.abs(sig - last).mean()) < self._min:
            return False
        self._last[(cam_id, zone_name)] = sig
        return True
