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


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-") or "zone"


def capture_loop(providers, boxes, fill_polys, calib_wh, out_dir, *,
                 prefix: str = "bg", interval_s: float = 2.0, count: int = 300,
                 min_diff: float = 4.0, max_idle_polls: int = 60,
                 stop: threading.Event | None = None,
                 sleep=time.sleep) -> dict[str, int]:
    """Poll every provider each tick; save deduped filled crops until ``count``
    images exist, ``stop`` is set, or ``max_idle_polls`` consecutive ticks
    yield no frame from any camera. Returns a ``{"cam/zone": n}`` tally."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dedup = CropDeduper(min_diff)
    counters: dict[tuple[str, str], int] = {}
    tally: dict[str, int] = {}
    saved_total = 0
    idle = 0
    while saved_total < count and (stop is None or not stop.is_set()):
        got_any = False
        for cam_id, provider in providers.items():
            if saved_total >= count:
                break
            frame = provider()
            if frame is None:
                continue
            got_any = True
            fh, fw = frame.shape[:2]
            for zone_name, box in boxes.get(cam_id) or []:
                if saved_total >= count:
                    break
                fx0, fy0, fx1, fy1, sx, sy = scale_box(
                    box, calib_wh.get(cam_id, (fw, fh)), (fw, fh))
                if fx1 - fx0 < 8 or fy1 - fy0 < 8:
                    continue
                crop = frame[fy0:fy1, fx0:fx1]
                fill = (fill_polys.get(cam_id) or {}).get(zone_name)
                if fill is not None:
                    crop = fill_crop(crop, fill, sx, sy, fx0, fy0)
                if not dedup.should_save(cam_id, zone_name, crop):
                    continue
                key = (cam_id, zone_name)
                n = counters.get(key, 0)
                counters[key] = n + 1
                path = out_dir / f"{prefix}_{cam_id}_{_slug(zone_name)}_{n:04d}.jpg"
                cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                tally_key = f"{cam_id}/{zone_name}"
                tally[tally_key] = tally.get(tally_key, 0) + 1
                saved_total += 1
                logger.info("saved %s (%d/%d)", path.name, saved_total, count)
        idle = 0 if got_any else idle + 1
        if idle >= max_idle_polls:
            logger.warning("no camera delivered frames for %d polls — stopping", idle)
            break
        if interval_s:
            sleep(interval_s)
    return tally


class BusProvider:
    """Latest frame from the /dev/shm bus (isistream running) — zero RTSP."""

    def __init__(self, camera_id: str, directory: str | None = None) -> None:
        from backbone.shared.frame_shm import FrameShmReader
        self.camera_id = camera_id
        self._reader = FrameShmReader(camera_id, directory=directory)

    def __call__(self):
        got = self._reader.latest()
        return None if got is None else got[0]

    def stop(self) -> None:
        pass


class RtspProvider:
    """Own RTSP session (SOFTWARE decode — never touch the GPU) pumping the
    newest frame into a slot; used only when the frame bus is absent/stale."""

    def __init__(self, camera_id: str, source_cfg: dict) -> None:
        import backbone.ingestion  # noqa: F401  auto-registration fires @register
        from backbone.core.registry import frame_source_registry
        kwargs = {k: source_cfg[k]
                  for k in ("latency_ms", "capture_fps", "output_wh")
                  if source_cfg.get(k) is not None}
        self.camera_id = camera_id
        self._src = frame_source_registry.create(
            "rtsp", camera_id=camera_id, url=source_cfg["url"],
            decoder="software", **kwargs)
        self._latest = None
        self._lock = threading.Lock()
        threading.Thread(target=self._pump, daemon=True,
                         name=f"rtsp-pump-{camera_id}").start()

    def _pump(self) -> None:
        try:
            for frame in self._src.frames():
                with self._lock:
                    self._latest = frame.image
        except Exception:
            logger.exception("%s: RTSP pump died", self.camera_id)

    def __call__(self):
        with self._lock:
            return self._latest

    def stop(self) -> None:
        self._src.stop()


def make_provider(cam_id: str, source_cfg: dict, *, bus_wait_s: float = 5.0,
                  frame_wait_s: float = 15.0, directory: str | None = None,
                  poll_s: float = 0.25):
    """Bus if it delivers within ``bus_wait_s``; else RTSP fallback (when the
    camera's config source is rtsp) if IT delivers within ``frame_wait_s``;
    else ``None`` (caller skips the camera)."""
    bus = BusProvider(cam_id, directory=directory)
    deadline = time.monotonic() + bus_wait_s
    while time.monotonic() < deadline:
        if bus() is not None:
            logger.info("%s: using /dev/shm frame bus", cam_id)
            return bus
        time.sleep(poll_s)
    source_cfg = source_cfg or {}
    if source_cfg.get("name") == "rtsp" and source_cfg.get("url"):
        logger.info("%s: bus absent — opening RTSP (software decode)", cam_id)
        try:
            rtsp = RtspProvider(cam_id, source_cfg)
        except Exception:
            logger.exception("%s: RTSP fallback failed to build", cam_id)
            return None
        deadline = time.monotonic() + frame_wait_s
        while time.monotonic() < deadline:
            if rtsp() is not None:
                return rtsp
            time.sleep(poll_s)
        rtsp.stop()
        logger.warning("%s: RTSP delivered no frame in %.0fs", cam_id, frame_wait_s)
    return None
