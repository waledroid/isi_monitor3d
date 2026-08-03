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
import yaml

from backbone.detection.enhance import enhance_bgr
from backbone.detection.zone_scope import (
    _FILL_GRAY,
    zone_crop_boxes,
    zone_fill_polygons,
)
from backbone.shared.camera_rig import CameraRig
from backbone.shared.zones import ZoneRegistry

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

    Signature = 64x64 grayscale; difference = mean absolute pixel delta.
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


def _seed_counter(out_dir: Path, prefix: str, cam_id: str, zone_name: str) -> int:
    """Continue numbering from existing ``{prefix}_{cam}_{slug}_NNNN.jpg``
    files in ``out_dir`` (max index + 1) so a second session never clobbers a
    first one's files — counters otherwise start at 0000 every run."""
    pattern = f"{prefix}_{cam_id}_{_slug(zone_name)}_*.jpg"
    best = -1
    for p in out_dir.glob(pattern):
        m = re.search(r"_(\d+)\.jpg$", p.name)
        if m:
            best = max(best, int(m.group(1)))
    return best + 1


def capture_loop(providers, boxes, fill_polys, calib_wh, out_dir, *,
                 prefix: str = "bg", interval_s: float = 2.0, count: int = 300,
                 min_diff: float = 4.0, max_idle_polls: int = 60,
                 stop: threading.Event | None = None,
                 sleep=time.sleep, enhance=None) -> dict[str, int]:
    """Poll every provider each tick; save deduped filled crops until ``count``
    images exist, ``stop`` is set, or ``max_idle_polls`` consecutive ticks
    yield no frame from any camera. Returns a ``{"cam/zone": n}`` tally.

    ``enhance``, when given, is a ``callable(crop) -> crop`` applied to EVERY
    crop (filled or not) right after the polygon fill — mirroring
    ``ZoneScopedDetector.detect`` (zone_scope.py: fill, then
    ``enhance_bgr``), so a saved background matches the inference domain when
    ``detection.enhance.enabled`` is on.
    """
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
                if enhance is not None:
                    crop = enhance(crop)
                if not dedup.should_save(cam_id, zone_name, crop):
                    continue
                key = (cam_id, zone_name)
                if key not in counters:
                    counters[key] = _seed_counter(out_dir, prefix, cam_id, zone_name)
                n = counters[key]
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Save inference-identical (gray-filled) zone crops as "
                    "YOLO background images. See module docstring for the "
                    "dataset merge procedure.")
    ap.add_argument("--config", required=True, help="backbone.yaml path")
    ap.add_argument("--out", default="trainer/isidet/data/bg_captures")
    ap.add_argument("--prefix", default="bg",
                    help="filename prefix; use 'pos' for occupied-zone sessions")
    ap.add_argument("--interval", type=float, default=2.0, help="seconds between polls")
    ap.add_argument("--count", type=int, default=300, help="stop after N saved images")
    ap.add_argument("--min-diff", type=float, default=4.0,
                    help="mean abs gray delta vs last saved crop to count as new")
    ap.add_argument("--cams", default=None, help="comma list; default: all in rig")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    cfg = yaml.safe_load(Path(args.config).read_text()) or {}
    rig = CameraRig.from_file(cfg["calibration_path"])
    zones_path = cfg.get("zones_path")
    zones = ZoneRegistry.load(zones_path) if zones_path else ZoneRegistry.empty()
    if len(zones) == 0:
        logger.error("no zones configured (%s) — nothing to crop; draw zones first",
                     zones_path or "no zones_path in config")
        return 2

    det = cfg.get("detection", {}) or {}
    crop_h = float(det.get("zone_crop_height_m", 0.0) or 0.0)
    boxes = zone_crop_boxes(rig, zones, crop_height_m=crop_h)
    fills = zone_fill_polygons(rig, zones, crop_height_m=crop_h)
    calib_wh = {cid: rig[cid].image_size_wh for cid in rig.camera_ids}

    # Mirror the live detector's crop enhancement (isistream/core.py,
    # zone_scope.py:467-468) so captured backgrounds match the domain the
    # model was actually shown, not a CLAHE/gamma-mismatched raw crop.
    enhance_cfg = det.get("enhance")
    enhance_fn = None
    if isinstance(enhance_cfg, dict) and enhance_cfg.get("enabled"):
        enh_kwargs = {
            "clip_limit": float(enhance_cfg.get("clip_limit", 2.0)),
            "tile_grid": int(enhance_cfg.get("tile_grid", 8)),
            "gamma": float(enhance_cfg.get("gamma", 1.0)),
        }
        enhance_fn = lambda img: enhance_bgr(img, **enh_kwargs)  # noqa: E731
        logger.info("crop enhancement ON (mirrors live detector: CLAHE clip=%.1f, "
                    "gamma=%.2f) — captures will match the inference domain",
                    enh_kwargs["clip_limit"], enh_kwargs["gamma"])

    cams = ([c.strip() for c in args.cams.split(",") if c.strip()]
            if args.cams else list(rig.camera_ids))

    # Substream domain mismatch: isistream defaults detect_substream to True,
    # so a per-camera detect_source (the camera's SUBSTREAM) silently becomes
    # the pixels the live detector sees, while this tool always captures the
    # camera's `source` (main stream). Only an explicit `false` guarantees
    # they're the same stream.
    isis_cfg = cfg.get("isistream", cfg.get("perception", {})) or {}
    if isis_cfg.get("detect_substream", True) is not False:
        cams_cfg = cfg.get("cameras", {}) or {}
        if any((cams_cfg.get(c) or {}).get("detect_source") for c in cams):
            logger.warning(
                "isistream.detect_substream is not explicitly false and at "
                "least one selected camera defines detect_source — captured "
                "crops come from the MAIN stream but live detection may run "
                "on the SUBSTREAM. Captures may not match the detection pixel "
                "domain; set isistream.detect_substream: false or verify manually.")

    providers = {}
    for cam_id in cams:
        source_cfg = ((cfg.get("cameras", {}).get(cam_id) or {}).get("source")) or {}
        provider = make_provider(cam_id, source_cfg)
        if provider is not None:
            providers[cam_id] = provider
        else:
            logger.warning("skipping %s: no frames from bus or RTSP", cam_id)
    if not providers:
        logger.error("no camera delivered frames — is the system (or a camera) up?")
        return 1

    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())

    try:
        tally = capture_loop(
            providers, boxes, fills, calib_wh, Path(args.out),
            prefix=args.prefix, interval_s=args.interval, count=args.count,
            min_diff=args.min_diff, stop=stop, enhance=enhance_fn)
    finally:
        for provider in providers.values():
            provider.stop()

    total = sum(tally.values())
    logger.info("done: %d image(s) in %s", total, args.out)
    for key in sorted(tally):
        logger.info("  %-30s %d", key, tally[key])
    logger.info("next: review every crop (delete any containing an object), then "
                "copy ~90%% to images/train and ~10%% to images/val with NO label files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
