"""Background zone-detection worker — the SINGLE detection driver for zone patches.

Why this exists (replaces the HTTP-driven `_zone_patch_iter` + `_ZONE_DET_CACHE`):
zone detection used to run inside each `/stream/zone/{id}` MJPEG connection, every
panel writing a module-global cache with its OWN timestamp. The cam view merged
entries up to 1.5 s apart, so a moving object appeared twice (one fresh, one stale,
at offset positions — the "duplicate shaky mask" bug); zones without a panel never
detected at all; concurrent connections to one zone raced on the cache.

Now ONE daemon thread per camera ("sandboxed by ownership", not by process — a
separate process per zone would need its own CUDA context on a 12 GB card and is
unnecessary because the detector session is stateless):

  - runs ALL of that camera's zones sequentially on the SAME frame,
  - resolves cross-zone overlaps at publish time (an object lands in exactly ONE
    zone — deepest polygon-centre wins via ``cv2.pointPolygonTest(measureDist)``),
  - publishes ONE atomic snapshot ``{"frame_ts": ts, "zones": {zone_id: [dets]}}``
    (fresh dict, single assignment → readers never see a half-written state),
  - idles (no inference, empty snapshot, hub stream released) while the Backbone
    is not running — preserving the raw-feed-before-START behavior.

The `/stream/zone/{id}` panels and the cam view become pure RENDERERS of this
snapshot. Detector sessions stay shared per (model, input_size) via
``detection_overlay.get_zone_detector``.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import yaml

from .api.routes_zone_patches import load_patches, patch_pixel_box, patch_rect
from .camera_hub import get_hub
from .detection_overlay import (
    ZoneModelUnavailable,
    display_fps,
    get_async_pose,
    get_zone_detector,
    read_backbone,
    resolve_model,
)

# Default loop cadence when the UI "Zones FPS" preference is absent. The worker
# paces its loop at the editable ``display_fps(cfg)`` value (Zones-FPS field in
# the Settings ▸ Zones tab); this is just the fallback ``display_fps`` returns
# when the key is unset. Every zone runs every pass (zones are sequential; a
# per-zone cadence cap cannot reduce total load, it only defers one zone's work
# to the next iteration while still occupying the same loop time).
DEFAULT_DETECTION_FPS: float = 10.0

logger = logging.getLogger(__name__)

# How long a published snapshot stays valid (floor). Older ⇒ consumers draw
# nothing (worker dead) instead of stale ghost boxes. A snapshot can carry its
# own longer `valid_s` when the detect pass itself is slow (heavy models / GPU
# contention), so the boxes never blink just because inference takes a while.
SNAPSHOT_MAX_AGE_S = 1.0


def _snapshot_fresh(snap: dict) -> bool:
    valid = float(snap.get("valid_s", SNAPSHOT_MAX_AGE_S))
    return time.time() - snap.get("frame_ts", 0.0) <= max(SNAPSHOT_MAX_AGE_S, valid)

# Circuit breaker: a zone whose detector failed (build OR inference) is skipped
# for this long, then retried — so a refused/broken model self-heals when VRAM
# frees up, while the other zones keep running undisturbed in between.
ZONE_RETRY_COOLDOWN_S = 30.0

# Motion gate: a zone whose crop hasn't visibly changed since its last
# inference SKIPS the model and republishes its cached detections — in a
# mostly-static warehouse this removes the bulk of zone GPU work (the nano
# model is launch-overhead-bound, so fewer calls is the real lever). "Changed"
# = > MOTION_FRAC of the 32x32 gray signature moved by > MOTION_PIX_DELTA
# levels (robust to sensor/compression noise). A forced re-inference every
# MOTION_REFRESH_S self-heals anything the gate misses (gradual light drift).
MOTION_GATE_ENABLED = True
MOTION_REFRESH_S = 2.0
MOTION_SIG_PX = 32
MOTION_PIX_DELTA = 15
MOTION_FRAC = 0.02

# SAHI frame-skip: the heavy tiled pass runs only every Nth genuine frame. Between
# passes the worker re-publishes the cached snapshot with a bumped frame_ts (the
# objects are static between passes — no tracker in v1), and the published
# `valid_s` is widened to cover the skip window so the carried boxes don't expire.
SAHI_PERIOD = 3

# Where zone detections come from. "backbone" (default): the worker renders
# the Backbone's per-camera ObservationsMessage from the UDP bus — ONE
# perception, zero dashboard inference. "local": today's in-dashboard
# detection path (per-zone models/SAHI/ENH), kept as the fallback/dev mode.
ZONE_SOURCE_KEY = "zone_detection_source"
ZONE_SOURCE_DEFAULT = "backbone"
# An observation older than this is stale (backbone stopped/hiccuped) — the
# worker publishes empty rather than ghost boxes.
OBSERVATIONS_MAX_AGE_S = 2.0


def _zone_source(cfg) -> str:
    """Read the zone-detection source preference from the UI-settings YAML."""
    try:
        data = yaml.safe_load(Path(cfg.ui_settings_path).read_text()) or {}
        val = str(data.get(ZONE_SOURCE_KEY, ZONE_SOURCE_DEFAULT)).lower()
        return val if val in ("backbone", "local") else ZONE_SOURCE_DEFAULT
    except Exception:
        return ZONE_SOURCE_DEFAULT


# Humans are rendered by POSE only on the cam views — any person-class box the
# zone object-model emits is dropped so a person isn't boxed AND skeletoned.
_PERSON_CLASSES = frozenset({"person", "people", "human", "pedestrian"})


def _iou(a, b) -> float:
    """IoU of two xyxy boxes (0 when disjoint)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _drop_persons(dets) -> list:
    """Strip every person-class detection. Humans are NEVER shown in a zone — they
    appear only on the cam views, via pose."""
    return [d for d in dets if str(getattr(d, "cls", "")).lower() not in _PERSON_CLASSES]


def _same_object(a, b, iou_thresh: float) -> bool:
    """Two boxes describe the SAME physical object — robust to partial overlap.

    Plain IoU misses the case that bites when zones overlap: each zone crops the
    object at a DIFFERENT boundary, so the two boxes are offset/clipped and their
    IoU is low even though it's one object. So ALSO treat them as one when:
      - EITHER box's centre falls inside the other (offset twins), or
      - their intersection covers most of the smaller box (one clips the other)."""
    if _iou(a, b) > iou_thresh:
        return True
    acx, acy = (a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0
    bcx, bcy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
    a_in_b = b[0] <= acx <= b[2] and b[1] <= acy <= b[3]
    b_in_a = a[0] <= bcx <= a[2] and a[1] <= bcy <= a[3]
    if a_in_b or b_in_a:
        return True
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    smaller = min(area_a, area_b)
    return smaller > 0.0 and inter / smaller > 0.6


def _zone_objects(dets, *, iou_thresh: float = 0.5) -> list:
    """Drop persons + dedupe same-class boxes describing one object (highest
    confidence wins). Kept as a utility (and for tests); the worker now applies
    the equivalent logic cross-zone at publish time."""
    objs = _drop_persons(dets)
    kept: list = []
    for d in sorted(objs, key=lambda x: float(getattr(x, "confidence", 0.0)), reverse=True):
        dc = str(getattr(d, "cls", "")).lower()
        if any(dc == str(getattr(k, "cls", "")).lower()
               and _same_object(d.bbox_xyxy, k.bbox_xyxy, iou_thresh) for k in kept):
            continue
        kept.append(d)
    return kept


def _enhance_crop(img, infer_size: int, prev_ema):
    """Far/dim-zone slice enhancement (the per-zone ENH toggle). Returns
    ``(enhanced_uint8, new_ema_uint8)``. Deliberately MINIMAL — measured on a
    320^2 crop, CLAHE-style contrast cost 15 ms and unsharp 2.4 ms per tile
    (a full CPU core at zone rates) for marginal CNN gain, so both were cut.
    What remains is ~0.5 ms:

    1. EMA temporal denoise (alpha=0.5, uint8 ``cv2.addWeighted`` — the float
       version cost 4.3 ms in conversions alone): fixed camera + mostly-static
       pallets => averaging across ticks removes IR/night sensor noise WITHOUT
       blurring object detail. Movers ghost slightly; people are pose's job.
    2. Cubic upscale to ``infer_size`` when the crop is smaller (otherwise the
       detector letterbox does a plain linear upscale of a tiny crop).
    """
    if prev_ema is not None and prev_ema.shape == img.shape:
        out = cv2.addWeighted(img, 0.5, prev_ema, 0.5, 0)
    else:
        out = img
    ema = out

    h, w = out.shape[:2]
    longest = max(h, w)
    if 0 < longest < infer_size:
        sc = infer_size / float(longest)
        out = cv2.resize(out, (max(1, round(w * sc)), max(1, round(h * sc))),
                         interpolation=cv2.INTER_CUBIC)
    return out, ema


def _merge_tile_dets(dets, *, ios_thresh: float = 0.15) -> list:
    """UNION-merge tile detections describing one physical object (SAHI).

    NMS-style suppression is wrong for tiled inference: a large object split
    across a 2x2 grid yields four PARTIAL boxes whose pairwise overlap is only
    the thin tile-overlap band — suppression either keeps all four (the
    observed un-joined quadrants) or keeps ONE quarter. Here same-class boxes
    that are connected (IoU, centre containment, or intersection-over-smaller
    >= ``ios_thresh``) merge into their UNION: bbox hull, max confidence,
    OR-composed masks, foot at the hull's bottom-centre. Greedy over clusters,
    repeated until stable so chains (TL-TR-BL-BR) collapse to one object."""
    from backbone.core.types import Detection
    objs = _drop_persons(dets)

    def connected(a, b) -> bool:
        if _iou(a, b) > 0.5:
            return True
        acx, acy = (a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0
        bcx, bcy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
        if (b[0] <= acx <= b[2] and b[1] <= acy <= b[3]) or \
           (a[0] <= bcx <= a[2] and a[1] <= bcy <= a[3]):
            return True
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        smaller = min(max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]),
                      max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]))
        return smaller > 0.0 and inter / smaller >= ios_thresh

    clusters: list[dict] = []
    for d in sorted(objs, key=lambda x: float(getattr(x, "confidence", 0.0)),
                    reverse=True):
        clusters.append({"cls": str(getattr(d, "cls", "")).lower(),
                         "bbox": list(d.bbox_xyxy), "members": [d]})
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(clusters):
            j = i + 1
            while j < len(clusters):
                a, b = clusters[i], clusters[j]
                if a["cls"] == b["cls"] and connected(a["bbox"], b["bbox"]):
                    a["bbox"] = [min(a["bbox"][0], b["bbox"][0]),
                                 min(a["bbox"][1], b["bbox"][1]),
                                 max(a["bbox"][2], b["bbox"][2]),
                                 max(a["bbox"][3], b["bbox"][3])]
                    a["members"].extend(b["members"])
                    del clusters[j]
                    changed = True
                else:
                    j += 1
            i += 1

    merged: list = []
    for c in clusters:
        best = max(c["members"], key=lambda x: float(getattr(x, "confidence", 0.0)))
        if len(c["members"]) == 1:
            merged.append(best)
            continue
        x1, y1, x2, y2 = c["bbox"]
        mask = None
        member_masks = [m.mask for m in c["members"] if m.mask is not None]
        if member_masks:
            mask = member_masks[0].copy()
            for mm in member_masks[1:]:
                mask |= mm
        merged.append(Detection(
            camera_id=best.camera_id, capture_ts=best.capture_ts, cls=best.cls,
            confidence=float(best.confidence),
            bbox_xyxy=(float(x1), float(y1), float(x2), float(y2)),
            foot_uv=((x1 + x2) / 2.0, float(y2)),
            keypoints_uv=None, mask=mask,
        ))
    merged.sort(key=lambda d: float(getattr(d, "confidence", 0.0)), reverse=True)
    return merged


def _remap_det(d, x0, y0, rx, ry, iw, ih, ch, cw):
    """Map one Detection from a zone crop's fed-image frame back to full-frame source
    pixels: scale by the fed→crop ratio (rx, ry) then offset by the crop origin
    (x0, y0). Boxes/foot are affine; the mask is resized to the crop (INTER_LINEAR on
    float + 0.5 threshold — smooth edges, no nearest-neighbour shake) and pasted in."""
    from backbone.core.types import Detection
    bx = d.bbox_xyxy
    bbox = (x0 + bx[0] * rx, y0 + bx[1] * ry, x0 + bx[2] * rx, y0 + bx[3] * ry)
    foot = None if d.foot_uv is None else (x0 + d.foot_uv[0] * rx, y0 + d.foot_uv[1] * ry)
    mask = None
    if d.mask is not None:
        mr = cv2.resize(d.mask.astype(np.uint8), (cw, ch),
                        interpolation=cv2.INTER_NEAREST).astype(bool)
        full = np.zeros((ih, iw), dtype=bool)
        full[y0:y0 + ch, x0:x0 + cw] = mr
        mask = full
    return Detection(camera_id=d.camera_id, capture_ts=d.capture_ts, cls=d.cls,
                     confidence=d.confidence, bbox_xyxy=bbox, foot_uv=foot,
                     keypoints_uv=d.keypoints_uv, mask=mask)


def _tile_det_to_crop(d, tmeta: dict, cw: int, ch: int):
    """Map one SAHI tile Detection from its (resized) fed-image frame into the ZONE
    CROP's pixel frame: scale by the tile fed→crop ratio ``(rx, ry)`` then offset by
    the tile origin ``(tx0, ty0)`` within the crop. The mask is resized to the
    tile's crop-pixel size (INTER_LINEAR + 0.5 threshold) and pasted into a
    crop-sized (``ch x cw``) canvas at the tile offset — so overlapping tiles compose
    in crop coords before the merge + the unchanged ``_remap_det`` (crop→source)."""
    from backbone.core.types import Detection
    tx0, ty0 = tmeta["tx0"], tmeta["ty0"]
    rx, ry, tw, th = tmeta["rx"], tmeta["ry"], tmeta["tw"], tmeta["th"]
    bx = d.bbox_xyxy
    bbox = (tx0 + bx[0] * rx, ty0 + bx[1] * ry, tx0 + bx[2] * rx, ty0 + bx[3] * ry)
    foot = None if d.foot_uv is None else (tx0 + d.foot_uv[0] * rx, ty0 + d.foot_uv[1] * ry)
    mask = None
    if d.mask is not None:
        mr = cv2.resize(d.mask.astype(np.uint8), (tw, th),
                        interpolation=cv2.INTER_NEAREST).astype(bool)
        full = np.zeros((ch, cw), dtype=bool)
        full[ty0:ty0 + th, tx0:tx0 + tw] = mr
        mask = full
    return Detection(camera_id=d.camera_id, capture_ts=d.capture_ts, cls=d.cls,
                     confidence=d.confidence, bbox_xyxy=bbox, foot_uv=foot,
                     keypoints_uv=d.keypoints_uv, mask=mask)


def _scaled_polygon(patch: dict, frame_wh) -> np.ndarray | None:
    """The patch's drawn polygon in CURRENT-frame source pixels (the same
    stored_wh→frame_wh guard as :func:`patch_pixel_box`), or ``None``."""
    poly = patch.get("polygon")
    if not isinstance(poly, list) or len(poly) < 3:
        return None
    fw, fh = int(frame_wh[0]), int(frame_wh[1])
    stored = patch.get("frame_wh")
    sx = sy = 1.0
    if stored and (int(stored[0]), int(stored[1])) != (fw, fh):
        sw, sh = float(stored[0]) or fw, float(stored[1]) or fh
        sx, sy = fw / sw, fh / sh
    return np.array([[float(u) * sx, float(v) * sy] for u, v in poly], dtype=np.float32)


class ZoneDetectionWorker:
    """One background thread per camera: detects every zone on the same frame and
    publishes one coherent snapshot. Single writer; readers get atomic dicts."""

    def __init__(self, camera_id: str, src_cfg: dict, cfg, is_running,
                 detector_factory=get_zone_detector, hub_factory=get_hub,
                 bus_getter=None):
        self.camera_id = camera_id
        self._src_cfg = dict(src_cfg)
        self._cfg = cfg
        self._is_running = is_running
        self._detector_factory = detector_factory
        self._hub_factory = hub_factory
        self._bus_getter = bus_getter
        self._patches: list[dict] = []
        self._snapshot: dict = {"frame_ts": 0.0, "zones": {}}
        self._stop = threading.Event()
        self._reload = threading.Event()
        self._thread: threading.Thread | None = None
        # Per-zone isolation state (worker-thread only — no locking needed):
        # circuit breaker {zone_id: (blocked_until_monotonic, reason)}.
        self._ema: dict[str, np.ndarray] = {}   # per-zone/tile EMA denoise state
        # Motion gate state: zone_id -> {"sig", "dets", "last_infer"} (see the
        # MOTION_* constants). Cleared with the EMA state on any zone change.
        self._motion: dict[str, dict] = {}
        self._zone_breaker: dict[str, tuple[float, str]] = {}
        # SAHI frame-skip counter (worker-thread only): counts genuine frames so
        # the heavy tiled pass runs every SAHI_PERIOD-th frame; between passes the
        # snapshot is carried forward (see _run).
        self._sahi_tick = 0

    # ---- lifecycle ----

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name=f"zonedet[{self.camera_id}]",
        )
        self._thread.start()

    def stop(self, join_timeout: float = 6.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=join_timeout)
            self._thread = None

    def _apply_enhance(self, patch: dict, fed, key: str):
        """Run the ENH chain on a fed crop/tile when the patch opts in; EMA
        state is keyed per zone (and per tile) and self-resets on shape change."""
        if not patch.get("enhance"):
            return fed
        infer_size = int(patch.get("infer_size") or 320)
        fed, ema = _enhance_crop(fed, infer_size, self._ema.get(key))
        self._ema[key] = ema
        return fed

    def set_patches(self, patches: list[dict], src_cfg: dict | None = None) -> None:
        """Swap in a fresh zone list (and optionally a new camera source); the loop
        picks it up at the next iteration via the reload event. Clears the per-zone
        breaker/cadence state so a config save gives every zone a fresh chance."""
        self._patches = list(patches)
        if src_cfg is not None:
            self._src_cfg = dict(src_cfg)
        self._zone_breaker.clear()
        self._ema.clear()          # zone set/geometry changed — fresh denoise state
        self._motion.clear()       # geometry changed — cached dets/signatures stale
        self._reload.set()

    # ---- read API (any thread) ----

    def snapshot(self) -> dict:
        """The latest published snapshot (treat as immutable)."""
        return self._snapshot

    def zone_dets(self, zone_id: str) -> list:
        snap = self._snapshot
        if not _snapshot_fresh(snap):
            return []
        return list(snap.get("zones", {}).get(str(zone_id), []))

    def zone_status(self, zone_id: str) -> str:
        """The zone's last published health: "ok", "no_vram", "error" — or "" when
        this worker hasn't (recently) covered the zone."""
        snap = self._snapshot
        if not _snapshot_fresh(snap):
            return ""
        return str(snap.get("status", {}).get(str(zone_id), ""))

    def all_dets(self) -> list:
        snap = self._snapshot
        if not _snapshot_fresh(snap):
            return []
        out: list = []
        for dets in snap.get("zones", {}).values():
            out.extend(dets)
        return out

    # ---- worker loop ----

    def _run(self) -> None:
        hub = self._hub_factory()
        stream = None
        acquired_cfg = None
        patches = list(self._patches)
        last_frame_id = None
        try:
            while not self._stop.is_set():
                if self._reload.is_set():
                    self._reload.clear()
                    patches = list(self._patches)
                    if stream is not None and self._src_cfg != acquired_cfg:
                        hub.release(stream)      # camera source changed → re-acquire
                        stream = None
                if not patches or not self._is_running():
                    # Idle: no inference, release the camera, publish empty so
                    # consumers render the raw feed (the pre-START state).
                    if stream is not None:
                        hub.release(stream)
                        stream = None
                    self._snapshot = {"frame_ts": time.time(), "zones": {}}
                    self._stop.wait(0.5)
                    continue
                if stream is None:
                    acquired_cfg = dict(self._src_cfg)
                    scfg = dict(self._src_cfg)
                    plugin = scfg.pop("name", "rtsp")
                    stream = hub.acquire(self.camera_id, plugin, scfg)
                frame = stream.latest_real_frame()
                if frame is None or frame is last_frame_id:
                    # No (new) genuine frame yet — placeholder or camera slower
                    # than our tick. Don't burn GPU re-detecting the same image,
                    # but KEEP the snapshot alive: the panels are still showing
                    # this same held frame, so its detections remain correct. A
                    # camera stalling >1 s used to expire the snapshot and make
                    # the overlay boxes blink off/on with RTSP jitter.
                    snap = self._snapshot
                    if snap.get("zones"):
                        self._snapshot = {**snap, "frame_ts": time.time()}
                    self._stop.wait(0.1)
                    continue
                last_frame_id = frame
                # SAHI frame-skip: when any zone slices, run the heavy tiled pass
                # only every SAHI_PERIOD-th genuine frame; between passes carry the
                # cached snapshot forward (bumped frame_ts) so the static boxes
                # don't blink. Zero overhead when no zone enables SAHI.
                self._sahi_tick += 1
                if (any(p.get("sahi") for p in patches)
                        and self._sahi_tick % SAHI_PERIOD != 0
                        and self._snapshot.get("zones")):
                    self._snapshot = {**self._snapshot, "frame_ts": time.time()}
                    self._stop.wait(1.0 / max(1.0, float(display_fps(self._cfg))))
                    continue
                try:
                    if _zone_source(self._cfg) == "backbone":
                        self._snapshot_from_bus(frame, patches)
                    else:
                        self._detect_all_zones(frame, patches)
                except Exception:
                    logger.warning("zone worker[%s]: detect pass failed", self.camera_id,
                                   exc_info=True)
                self._stop.wait(1.0 / max(1.0, float(display_fps(self._cfg))))
        finally:
            if stream is not None:
                hub.release(stream)

    def _detect_all_zones(self, frame, patches: list[dict]) -> None:
        """Run every zone on THIS frame, resolve cross-zone overlaps, publish once.
        Also runs PERSON POSE on the full frame (the map's digital twin needs people
        positions even while no cam stream is open, e.g. on the MAP view).

        Per-zone isolation: a zone whose detector fails (VRAM admission refused,
        build error, inference error) is disabled for ZONE_RETRY_COOLDOWN_S and its
        status published — the OTHER zones keep detecting."""
        ih, iw = frame.shape[:2]
        ts = time.time()
        now = time.monotonic()
        det_cfg = read_backbone(self._cfg).get("detection") or {}
        global_conf = float(det_cfg.get("confidence_threshold", 0.3))
        per_zone: dict[str, list] = {}
        statuses: dict[str, str] = {}
        polys: dict[str, np.ndarray | None] = {}
        # Phase 1 — record polygons + the breaker decision for every zone, and
        # GROUP the breaker-allowed zones by (resolved model, infer_size): zones
        # sharing that key share one detector session (the same key
        # `get_zone_detector` caches on), so they can be fed in ONE detect() call.
        groups: dict[tuple, list[dict]] = {}
        sahi_patches: list[dict] = []
        pending_sig: dict[str, np.ndarray] = {}
        for patch in patches:
            zone_id = str(patch.get("id"))
            polys[zone_id] = _scaled_polygon(patch, (iw, ih))
            blocked_until, reason = self._zone_breaker.get(zone_id, (0.0, ""))
            if now < blocked_until:
                # Breaker open → excluded from the batch entirely (it isn't run).
                per_zone[zone_id], statuses[zone_id] = [], reason
                continue
            # Motion gate (applies to plain AND SAHI zones — SAHI passes are the
            # most expensive, so skipping static ones saves the most).
            if MOTION_GATE_ENABLED:
                sig = self._motion_signature(frame, patch, iw, ih)
                cached = self._motion.get(zone_id)
                if (sig is not None and cached is not None
                        and now - cached["last_infer"] < MOTION_REFRESH_S
                        and float((np.abs(sig - cached["sig"])
                                   > MOTION_PIX_DELTA).mean()) <= MOTION_FRAC):
                    per_zone[zone_id] = list(cached["dets"])
                    statuses[zone_id] = "ok"
                    continue
                if sig is not None:
                    pending_sig[zone_id] = sig
            if patch.get("sahi"):
                # SAHI zones never share a batch — each owns a tiled detect pass.
                sahi_patches.append(patch)
                continue
            model = patch.get("model")
            resolved_model = resolve_model(model, self._cfg) if model else None
            infer_size = int(patch.get("infer_size") or 320)
            # Group by the RESOLVED model when it resolves (same key
            # get_zone_detector caches its session on), else by the raw model
            # string — so two distinct-but-unresolvable models don't collapse
            # into one group (which would let one bad model fail its neighbours).
            model_key = (str(resolved_model) if resolved_model
                         else (str(model) if model else "__global__"))
            key = (model_key, infer_size)
            groups.setdefault(key, []).append(patch)

        # Phase 2 — resolve the detector once per group, then batch (dynamic-batch
        # model + >1 zone) or run per-zone. A batched failure degrades to the
        # per-zone path so the per-zone circuit breaker still pinpoints the culprit.
        for key, group in groups.items():
            # Single-zone groups can't be batched → straight to the per-zone path
            # (which resolves the detector itself and owns the breaker). This also
            # keeps the per-zone path's exact resolve-once behaviour for that zone.
            if len(group) > 1:
                try:
                    detector = self._detector_factory(group[0].get("model"),
                                                      self._cfg, key[1])
                except ZoneModelUnavailable as exc:
                    # VRAM admission refused / build failure for a SHARED model+size
                    # — trip the breaker for every zone in the group (they share the
                    # one resolved model, so they'd all hit the same wall).
                    for patch in group:
                        zid = str(patch.get("id"))
                        self._zone_breaker[zid] = (now + ZONE_RETRY_COOLDOWN_S,
                                                   exc.reason)
                        per_zone[zid], statuses[zid] = [], exc.reason
                    logger.warning("zone worker[%s]: group %s disabled for %.0fs "
                                   "(%s): %s", self.camera_id, key,
                                   ZONE_RETRY_COOLDOWN_S, exc.reason, exc)
                    continue
                except Exception:
                    # Any other resolution failure → degrade to per-zone, which
                    # re-resolves each zone and isolates the culprit via the
                    # breaker without touching its neighbours.
                    self._detect_group_per_zone(frame, group, iw, ih, global_conf,
                                                now, per_zone, statuses)
                    continue
                if getattr(detector, "supports_batch", False):
                    ok = self._detect_group_batched(frame, group, detector, iw, ih,
                                                    global_conf, now, per_zone,
                                                    statuses)
                    if ok:
                        continue
                    # batched detect() raised → fall through to per-zone so the
                    # breaker can pinpoint the offending zone.
            self._detect_group_per_zone(frame, group, iw, ih, global_conf,
                                        now, per_zone, statuses)
        # SAHI zones: each runs its own tiled detect pass, isolated by the same
        # per-zone circuit breaker (VRAM admission / build / inference failures).
        for patch in sahi_patches:
            zid = str(patch.get("id"))
            try:
                dets = self._detect_zone_sahi(frame, patch, iw, ih, global_conf)
            except ZoneModelUnavailable as exc:
                self._zone_breaker[zid] = (now + ZONE_RETRY_COOLDOWN_S, exc.reason)
                per_zone[zid], statuses[zid] = [], exc.reason
                logger.warning("zone worker[%s]: sahi zone %s disabled for %.0fs (%s): %s",
                               self.camera_id, zid, ZONE_RETRY_COOLDOWN_S,
                               exc.reason, exc)
                continue
            except Exception:
                self._zone_breaker[zid] = (now + ZONE_RETRY_COOLDOWN_S, "error")
                per_zone[zid], statuses[zid] = [], "error"
                logger.warning("zone worker[%s]: sahi zone %s failed — disabled for %.0fs",
                               self.camera_id, zid, ZONE_RETRY_COOLDOWN_S,
                               exc_info=True)
                continue
            self._zone_breaker.pop(zid, None)
            per_zone[zid], statuses[zid] = dets, "ok"
        # Commit motion state for every zone that RAN this tick and succeeded —
        # its detections become the cached result served while the crop is still.
        for zid, sig in pending_sig.items():
            if statuses.get(zid) == "ok":
                self._motion[zid] = {"sig": sig,
                                     "dets": list(per_zone.get(zid) or []),
                                     "last_infer": now}
        resolved = self._resolve_overlaps(per_zone, polys)
        # People (full-frame pose, foot points in source px) — best-effort.
        # SHARED async runner (same one the cam view uses): submit the frame,
        # read the LATEST completed skeletons. The zone tick never blocks on
        # pose, and pose runs ONCE per camera on the GPU instead of twice
        # (worker + cam view each ran their own full-frame pass before).
        people: list = []
        try:
            pose = get_async_pose(self._cfg, self.camera_id)
            if pose is not None:
                for p in pose.predict(frame):
                    foot = getattr(p, "foot_uv", None)
                    if foot is not None:
                        people.append({"foot_uv": (float(foot[0]), float(foot[1])),
                                       "confidence": float(getattr(p, "confidence", 0.0))})
        except Exception:
            logger.warning("zone worker[%s]: pose failed", self.camera_id, exc_info=True)
        # Publish: ONE fresh dict, one assignment — atomic under the GIL, so every
        # consumer sees a coherent single-frame result with a single timestamp.
        # `valid_s` scales the consumers' staleness window with the ACTUAL pass
        # duration, so a slow pass (heavy models re-inferring + GPU contention
        # with the live Backbone) can't expire the snapshot between publishes —
        # that gap is exactly what made the overlay boxes blink rhythmically.
        pass_dt = time.time() - ts
        valid_s = max(SNAPSHOT_MAX_AGE_S, 2.5 * pass_dt)
        # When any zone slices, the next genuine detect pass is SAHI_PERIOD loop
        # intervals away; widen the validity window to cover that skip so the
        # carried boxes don't expire between heavy passes.
        if any(p.get("sahi") for p in patches):
            loop_dt = 1.0 / max(1.0, float(display_fps(self._cfg)))
            valid_s = max(valid_s, (SAHI_PERIOD + 1) * loop_dt + pass_dt)
        self._snapshot = {"frame_ts": time.time(), "frame_wh": (iw, ih),
                          "zones": resolved, "status": statuses, "people": people,
                          "valid_s": valid_s}

    def _snapshot_from_bus(self, frame, patches: list[dict]) -> None:
        """Backbone-sourced pass: render the wire's per-camera observations.

        ONE perception — no dashboard inference. The Backbone's
        ``ObservationsMessage`` (calibration-frame coords) is rescaled to THIS
        hub frame, grouped into zone patches by polygon containment, and
        published as the exact snapshot shape the panels / cards / cam views
        already consume. Pose people still ride along (the MAP needs them).
        """
        ih, iw = frame.shape[:2]
        per_zone: dict[str, list] = {str(p.get("id")): [] for p in patches}
        statuses = {zid: "ok" for zid in per_zone}
        polys = {str(p.get("id")): _scaled_polygon(p, (iw, ih)) for p in patches}

        bus = self._bus_getter() if self._bus_getter is not None else None
        msg = None
        if bus is not None:
            try:
                msg = bus.snapshot().observations_by_camera.get(self.camera_id)
            except Exception:
                msg = None
        if msg is not None and time.time() - float(msg.ts) <= OBSERVATIONS_MAX_AGE_S:
            fw, fh = msg.frame_wh
            sx, sy = iw / float(fw), ih / float(fh)
            for od in msg.dets:
                if str(od.cls).lower() in _PERSON_CLASSES:
                    continue                     # humans render via pose only
                x0, y0, x1, y1 = od.bbox_xyxy
                det = SimpleNamespace(
                    camera_id=self.camera_id, capture_ts=float(msg.ts),
                    keypoints_uv=None, mask_offset_xy=None,
                    cls=od.cls, confidence=float(od.confidence),
                    bbox_xyxy=(x0 * sx, y0 * sy, x1 * sx, y1 * sy),
                    foot_uv=(od.foot_uv[0] * sx, od.foot_uv[1] * sy),
                    mask=None,
                    mask_poly=[[px * sx, py * sy] for px, py in od.mask_poly]
                              if od.mask_poly else None,
                    occupancy_state=od.occupancy_state,
                    occupancy_content=od.occupancy_content,
                    occupancy_confidence=float(od.occupancy_confidence),
                )
                cx = (det.bbox_xyxy[0] + det.bbox_xyxy[2]) / 2.0
                cy = (det.bbox_xyxy[1] + det.bbox_xyxy[3]) / 2.0
                for zid, poly in polys.items():
                    if poly is None or len(poly) < 3:
                        continue
                    if cv2.pointPolygonTest(
                            poly.astype(np.float32), (float(cx), float(cy)),
                            False) >= 0:
                        per_zone[zid].append(det)
                # dets outside every patch simply aren't shown — same as today.
        resolved = self._resolve_overlaps(per_zone, polys)

        people: list = []
        try:
            pose = get_async_pose(self._cfg, self.camera_id)
            if pose is not None:
                for pp in pose.predict(frame):
                    foot = getattr(pp, "foot_uv", None)
                    if foot is not None:
                        people.append({"foot_uv": (float(foot[0]), float(foot[1])),
                                       "confidence": float(getattr(pp, "confidence", 0.0))})
        except Exception:
            logger.warning("zone worker[%s]: pose failed", self.camera_id, exc_info=True)

        self._snapshot = {"frame_ts": time.time(), "frame_wh": (iw, ih),
                          "zones": resolved, "status": statuses, "people": people,
                          "valid_s": SNAPSHOT_MAX_AGE_S}

    def _motion_signature(self, frame, patch: dict, iw: int, ih: int):
        """Tiny gray thumbnail of the zone's crop region (``MOTION_SIG_PX``²,
        int16 for signed diffs) — ~0.1 ms. ``None`` for degenerate crops."""
        try:
            rect = patch_rect(patch)
            if rect is None:
                return None
            box = patch_pixel_box(rect, patch.get("frame_wh"), (iw, ih))
            if box is None:
                return None
            x0, y0, x1, y1 = box
            if x1 - x0 < 4 or y1 - y0 < 4:
                return None
            crop = frame[y0:y1, x0:x1]
            small = cv2.resize(crop, (MOTION_SIG_PX, MOTION_SIG_PX),
                               interpolation=cv2.INTER_AREA)
            return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.int16)
        except Exception:
            return None

    def _build_zone_crop(self, frame, patch: dict, iw: int, ih: int):
        """ROI-crop + pre-resize one zone to its ``infer_size``. Returns
        ``(fed, meta)`` where ``fed`` is the BGR image to feed the detector and
        ``meta`` carries everything ``_postprocess_zone`` needs to remap detections
        back to full-frame pixels. Returns ``None`` on a degenerate/zero-size crop
        (the zone is then published as ``[]`` — excluded from any batch)."""
        rect = patch_rect(patch)
        if rect is None:
            return None
        box = patch_pixel_box(rect, patch.get("frame_wh"), (iw, ih))
        if box is None:
            return None
        x0, y0, x1, y1 = box
        crop = frame[y0:y1, x0:x1]
        ch, cw = crop.shape[:2]
        if ch <= 0 or cw <= 0:
            return None
        infer_size = int(patch.get("infer_size") or 320)
        fed = crop
        longest = max(ch, cw)
        if longest > infer_size and longest > 0:
            s = infer_size / float(longest)
            fed = cv2.resize(crop, (max(1, round(cw * s)), max(1, round(ch * s))),
                             interpolation=cv2.INTER_AREA)
        # ENH toggle (before meta: the chain may upscale small crops, and
        # fw/fh below must describe the FED image for the det remap).
        fed = self._apply_enhance(patch, fed, str(patch.get("id")))
        fh, fw = fed.shape[:2]
        meta = {"x0": x0, "y0": y0, "rect": rect, "cw": cw, "ch": ch,
                "fw": fw, "fh": fh}
        return fed, meta

    def _postprocess_zone(self, raw_dets, patch: dict, meta: dict,
                          iw: int, ih: int, global_conf: float) -> list:
        """Filter + clip + remap one zone's raw detections (fed-image coords) to
        full-frame pixels: per-zone confidence floor → drop persons → polygon clip
        (in fed coords) → ``_remap_det``. Per-zone by construction, so a differing
        conf/polygon within a batched group is still applied correctly."""
        rect = meta["rect"]
        x0, y0 = meta["x0"], meta["y0"]
        cw, ch, fw, fh = meta["cw"], meta["ch"], meta["fw"], meta["fh"]
        dets = list(raw_dets)
        # Per-zone confidence as a POST-FILTER (session runs at a low floor).
        conf = patch.get("confidence")
        eff_conf = float(conf) if conf is not None else global_conf
        dets = [d for d in dets if float(getattr(d, "confidence", 0.0)) >= eff_conf]
        dets = _drop_persons(dets)
        # Clip to the drawn POLYGON (in fed-crop coords), not just the bounding rect.
        poly = patch.get("polygon")
        if poly and len(poly) >= 3 and dets:
            rw = float(rect[2] - rect[0]) or 1.0
            rh = float(rect[3] - rect[1]) or 1.0
            fed_poly = np.array(
                [[(u - rect[0]) / rw * fw, (v - rect[1]) / rh * fh] for u, v in poly],
                dtype=np.float32,
            )
            dets = [
                d for d in dets
                if cv2.pointPolygonTest(
                    fed_poly,
                    (float((d.bbox_xyxy[0] + d.bbox_xyxy[2]) / 2.0),
                     float((d.bbox_xyxy[1] + d.bbox_xyxy[3]) / 2.0)),
                    False) >= 0
            ]
        rx, ry = cw / float(fw), ch / float(fh)
        return [_remap_det(d, x0, y0, rx, ry, iw, ih, ch, cw) for d in dets]

    def _detect_zone(self, frame, patch: dict, iw: int, ih: int,
                     global_conf: float) -> list:
        """One zone on one frame → full-frame-coordinate detections (person-free,
        confidence-filtered, polygon-clipped). Thin wrapper: build crop → resolve
        detector → single-frame FramePair (keyed by camera_id) → detect →
        postprocess. Detector build/inference failures PROPAGATE — the caller owns
        the per-zone circuit breaker, so one zone's failure never silently looks
        like 'no objects'. Byte-for-byte equivalent to the pre-batching path."""
        from backbone.core.types import Frame, FramePair
        built = self._build_zone_crop(frame, patch, iw, ih)
        if built is None:
            return []
        fed, meta = built
        infer_size = int(patch.get("infer_size") or 320)
        detector = self._detector_factory(patch.get("model"), self._cfg, infer_size)
        ts = time.time()
        pair = FramePair(capture_ts=ts, frame_idx=0,
                         frames={self.camera_id: Frame(camera_id=self.camera_id,
                                                       capture_ts=ts, frame_idx=0,
                                                       image=fed)})
        raw = detector.detect(pair).get(self.camera_id, [])
        return self._postprocess_zone(raw, patch, meta, iw, ih, global_conf)

    def _build_zone_tiles(self, frame, patch: dict, iw: int, ih: int):
        """Slice ONE zone's crop into a ``rows x cols`` grid of OVERLAPPING tiles for
        SAHI. Returns ``(tiles, zmeta)`` where:
          - ``tiles`` is a list of ``(fed, tmeta)``: ``fed`` is the BGR tile resized
            to ``infer_size`` (same longest-side rule as ``_build_zone_crop``);
            ``tmeta`` carries the tile's origin ``(tx0, ty0)`` and scale
            ``(rx, ry)`` WITHIN THE ZONE CROP (tile-fed→crop pixels), plus its
            crop-pixel size ``(tw, th)``;
          - ``zmeta`` is the zone-crop meta consumed by ``_postprocess_zone`` with
            ``fed == the full crop`` (``fw=cw, fh=ch`` ⇒ unit fed→crop scale), so
            the merged crop-coord detections postprocess + remap unchanged.
        Returns ``None`` on a degenerate/zero-size crop (zone published as ``[]``)."""
        rect = patch_rect(patch)
        if rect is None:
            return None
        box = patch_pixel_box(rect, patch.get("frame_wh"), (iw, ih))
        if box is None:
            return None
        x0, y0, x1, y1 = box
        cw, ch = x1 - x0, y1 - y0
        if cw <= 0 or ch <= 0:
            return None
        crop = frame[y0:y1, x0:x1]
        overlap = max(0.0, min(0.5, float(patch.get("sahi_overlap") or 0.2)))
        infer_size = int(patch.get("infer_size") or 320)
        rows = max(1, min(4, int(patch.get("sahi_rows") or 2)))
        cols = max(1, min(4, int(patch.get("sahi_cols") or 2)))
        # Tile step = crop / count; tile size = step grown by the overlap fraction,
        # clamped to the crop. Origins step by the base size so the grid spans the
        # whole crop with neighbour tiles sharing an `overlap`-wide band.
        base_w, base_h = cw / cols, ch / rows
        tile_w = min(cw, base_w * (1.0 + overlap))
        tile_h = min(ch, base_h * (1.0 + overlap))
        tiles: list = []
        for r in range(rows):
            for c in range(cols):
                tx0 = round(c * base_w)
                ty0 = round(r * base_h)
                tx1 = min(cw, round(tx0 + tile_w))
                ty1 = min(ch, round(ty0 + tile_h))
                tx0 = max(0, min(tx0, tx1 - 1))
                ty0 = max(0, min(ty0, ty1 - 1))
                tw, th = tx1 - tx0, ty1 - ty0
                if tw <= 0 or th <= 0:
                    continue
                sub = crop[ty0:ty1, tx0:tx1]
                fed = sub
                longest = max(tw, th)
                if longest > infer_size and longest > 0:
                    s = infer_size / float(longest)
                    fed = cv2.resize(sub, (max(1, round(tw * s)), max(1, round(th * s))),
                                     interpolation=cv2.INTER_AREA)
                # ENH toggle — per-tile EMA key; rx/ry below use the FED size.
                fed = self._apply_enhance(patch, fed,
                                          f"{patch.get('id')}#{r}x{c}")
                fh, fw = fed.shape[:2]
                # fed→crop scale: a tile-fed pixel maps to (tw/fw, th/fh) crop px.
                tmeta = {"tx0": tx0, "ty0": ty0, "tw": tw, "th": th,
                         "rx": tw / float(fw), "ry": th / float(fh)}
                tiles.append((fed, tmeta))
        if not tiles:
            return None
        zmeta = {"x0": x0, "y0": y0, "rect": rect, "cw": cw, "ch": ch,
                 "fw": cw, "fh": ch}
        return tiles, zmeta

    def _detect_zone_sahi(self, frame, patch: dict, iw: int, ih: int,
                          global_conf: float) -> list:
        """SAHI path for ONE zone: slice the crop into overlapping tiles, detect
        each at ``infer_size``, map every tile's detections into ZONE-CROP coords,
        NMS-merge the overlapping-tile twins (``_zone_objects``), then run the
        EXISTING ``_postprocess_zone`` (conf/person/polygon clip + ``_remap_det``)
        unchanged. Detector build/inference failures PROPAGATE — the caller owns
        the per-zone circuit breaker. Batched in one ``detect()`` for YOLO (dynamic
        batch dim); sequential on the shared session for RF-DETR/OpenVINO."""
        from backbone.core.types import Frame, FramePair
        built = self._build_zone_tiles(frame, patch, iw, ih)
        if built is None:
            return []
        tiles, zmeta = built
        infer_size = int(patch.get("infer_size") or 320)
        detector = self._detector_factory(patch.get("model"), self._cfg, infer_size)
        zid = str(patch.get("id"))
        ts = time.time()
        frames: dict[str, object] = {}
        keys: list[str] = []
        for i, (fed, _tm) in enumerate(tiles):
            k = f"{zid}#{i}"
            keys.append(k)
            frames[k] = Frame(camera_id=k, capture_ts=ts, frame_idx=0, image=fed)
        if getattr(detector, "supports_batch", False):
            out = detector.detect(FramePair(capture_ts=ts, frame_idx=0, frames=frames))
        else:
            # Non-batchable (RF-DETR/OpenVINO): one detect() per tile on the shared
            # session — correct, just N runs.
            out = {}
            for k in keys:
                one = FramePair(capture_ts=ts, frame_idx=0, frames={k: frames[k]})
                out.update(detector.detect(one))
        # Map every tile's detections into zone-crop coords, then concat.
        crop_dets: list = []
        for i, (_fed, tm) in enumerate(tiles):
            crop_dets.extend(
                _tile_det_to_crop(d, tm, zmeta["cw"], zmeta["ch"])
                for d in out.get(keys[i], [])
            )
        merged = _merge_tile_dets(crop_dets)
        return self._postprocess_zone(merged, patch, zmeta, iw, ih, global_conf)

    def _detect_group_batched(self, frame, group: list[dict], detector,
                              iw: int, ih: int, global_conf: float, now: float,
                              per_zone: dict, statuses: dict) -> bool:
        """Run a whole group in ONE ``detect()`` call, the FramePair keyed by
        zone_id. Empty/degenerate crops are excluded from the batch (published
        ``[]``, "ok"). On a detect() exception returns ``False`` so the caller
        falls back to the per-zone path (which isolates the culprit via the
        breaker); NEVER raises. On success: postprocess per zone, clear the
        breaker, status "ok"."""
        from backbone.core.types import Frame, FramePair
        ts = time.time()
        frames: dict[str, object] = {}
        metas: dict[str, dict] = {}
        for patch in group:
            zid = str(patch.get("id"))
            built = self._build_zone_crop(frame, patch, iw, ih)
            if built is None:
                # Degenerate crop: nothing to detect — exclude from the batch.
                self._zone_breaker.pop(zid, None)
                per_zone[zid], statuses[zid] = [], "ok"
                continue
            fed, meta = built
            metas[zid] = meta
            frames[zid] = Frame(camera_id=zid, capture_ts=ts, frame_idx=0, image=fed)
        if not frames:
            return True   # every crop was empty — handled, nothing to batch
        pair = FramePair(capture_ts=ts, frame_idx=0, frames=frames)
        try:
            out = detector.detect(pair)
        except Exception:
            logger.warning("zone worker[%s]: batched detect failed for %d zone(s) — "
                           "falling back per-zone", self.camera_id, len(frames),
                           exc_info=True)
            return False
        for zid, meta in metas.items():
            patch = next(p for p in group if str(p.get("id")) == zid)
            raw = out.get(zid, [])   # missing key ⇒ no detections for that zone
            per_zone[zid] = self._postprocess_zone(raw, patch, meta, iw, ih,
                                                   global_conf)
            statuses[zid] = "ok"
            self._zone_breaker.pop(zid, None)
        return True

    def _detect_group_per_zone(self, frame, group: list[dict], iw: int, ih: int,
                               global_conf: float, now: float,
                               per_zone: dict, statuses: dict) -> None:
        """Today's per-zone loop body — shared by the non-batchable path AND the
        batched-failure fallback, so per-zone error isolation is preserved: a zone
        whose build/inference fails trips its OWN breaker (no_vram/error) while the
        others stay 'ok'."""
        for patch in group:
            zid = str(patch.get("id"))
            try:
                dets = self._detect_zone(frame, patch, iw, ih, global_conf)
            except ZoneModelUnavailable as exc:
                self._zone_breaker[zid] = (now + ZONE_RETRY_COOLDOWN_S, exc.reason)
                per_zone[zid], statuses[zid] = [], exc.reason
                logger.warning("zone worker[%s]: zone %s disabled for %.0fs (%s): %s",
                               self.camera_id, zid, ZONE_RETRY_COOLDOWN_S,
                               exc.reason, exc)
                continue
            except Exception:
                self._zone_breaker[zid] = (now + ZONE_RETRY_COOLDOWN_S, "error")
                per_zone[zid], statuses[zid] = [], "error"
                logger.warning("zone worker[%s]: zone %s failed — disabled for %.0fs",
                               self.camera_id, zid, ZONE_RETRY_COOLDOWN_S,
                               exc_info=True)
                continue
            self._zone_breaker.pop(zid, None)
            per_zone[zid], statuses[zid] = dets, "ok"

    @staticmethod
    def _resolve_overlaps(per_zone: dict[str, list],
                          polys: dict[str, np.ndarray | None]) -> dict[str, list]:
        """Cross-zone dedupe on ONE frame: same-class boxes describing one physical
        object (``_same_object``) collapse to a single det, owned by the zone whose
        polygon contains the box centre the DEEPEST (pointPolygonTest measureDist);
        tie → higher confidence. Each object ends up in exactly one zone's list."""
        flat = [(zid, d) for zid, dets in per_zone.items() for d in dets]
        if len(flat) <= 1:
            return {zid: list(dets) for zid, dets in per_zone.items()}

        def depth(zid, d):
            poly = polys.get(zid)
            if poly is None:
                return 0.0
            cx = float((d.bbox_xyxy[0] + d.bbox_xyxy[2]) / 2.0)
            cy = float((d.bbox_xyxy[1] + d.bbox_xyxy[3]) / 2.0)
            return float(cv2.pointPolygonTest(poly, (cx, cy), True))

        # Deeper-in-polygon first (then confidence) so the greedy keep below
        # naturally assigns each clustered object to its best-owning zone.
        flat.sort(key=lambda t: (depth(t[0], t[1]),
                                 float(getattr(t[1], "confidence", 0.0))), reverse=True)
        out: dict[str, list] = {zid: [] for zid in per_zone}
        kept: list = []   # (zone_id, det) winners across all zones
        for zid, d in flat:
            dc = str(getattr(d, "cls", "")).lower()
            if any(dc == str(getattr(k, "cls", "")).lower()
                   and _same_object(d.bbox_xyxy, k.bbox_xyxy, 0.5) for _, k in kept):
                continue
            kept.append((zid, d))
            out[zid].append(d)
        return out


class ZoneWorkerManager:
    """Owns one :class:`ZoneDetectionWorker` per camera that has zones. Reload is
    explicit (called from the zone-patches / camera save endpoints) — it handles
    thread topology (new camera ⇒ new worker, last zone gone ⇒ worker stopped)."""

    def __init__(self, cfg, is_running, *,
                 detector_factory=get_zone_detector, hub_factory=get_hub,
                 bus_getter=None):
        self._cfg = cfg
        self._is_running = is_running
        self._detector_factory = detector_factory
        self._hub_factory = hub_factory
        self._bus_getter = bus_getter
        self._workers: dict[str, ZoneDetectionWorker] = {}
        self._lock = threading.Lock()

    def start(self) -> None:
        self.reload()

    def stop(self) -> None:
        with self._lock:
            workers, self._workers = list(self._workers.values()), {}
        for w in workers:
            w.stop()

    def reload(self) -> None:
        """Sync workers to the CURRENT zone + camera config. Safe no-op on empty or
        missing config (tests / fresh installs spawn zero threads)."""
        try:
            patches = load_patches(self._cfg)
        except Exception:
            patches = []
        cameras = self._load_cameras()
        by_cam: dict[str, list[dict]] = {}
        for p in patches:
            cam = str(p.get("camera") or "cam_a")
            if cam in cameras:
                by_cam.setdefault(cam, []).append(p)
        with self._lock:
            # Stop workers for cameras that no longer have zones.
            for cam in [c for c in self._workers if c not in by_cam]:
                worker = self._workers.pop(cam)
                worker.stop()
                logger.info("zone worker[%s]: stopped (no zones)", cam)
            # Start/refresh the rest.
            for cam, cam_patches in by_cam.items():
                src_cfg = (cameras.get(cam) or {}).get("source", {})
                worker = self._workers.get(cam)
                if worker is None:
                    worker = ZoneDetectionWorker(
                        cam, src_cfg, self._cfg, self._is_running,
                        detector_factory=self._detector_factory,
                        hub_factory=self._hub_factory,
                        bus_getter=self._bus_getter,
                    )
                    worker.set_patches(cam_patches)
                    worker.start()
                    self._workers[cam] = worker
                    logger.info("zone worker[%s]: started (%d zone(s))",
                                cam, len(cam_patches))
                else:
                    worker.set_patches(cam_patches, src_cfg=src_cfg)

    def _load_cameras(self) -> dict[str, dict]:
        import yaml
        path = self._cfg.backbone_config_path
        try:
            if not path.exists():
                return {}
            data = yaml.safe_load(path.read_text()) or {}
        except (OSError, yaml.YAMLError):
            return {}
        cams = data.get("cameras", {})
        return cams if isinstance(cams, dict) else {}

    # ---- read API ----

    def zone_dets(self, zone_id: str) -> list:
        with self._lock:
            workers = list(self._workers.values())
        for w in workers:
            dets = w.zone_dets(zone_id)
            if dets:
                return dets
        return []

    def zone_status(self, zone_id: str) -> str:
        """Health of one zone across workers ("" when no worker covers it)."""
        with self._lock:
            workers = list(self._workers.values())
        for w in workers:
            status = w.zone_status(zone_id)
            if status:
                return status
        return ""

    def camera_dets(self, camera_id: str) -> list:
        with self._lock:
            worker = self._workers.get(camera_id)
        return worker.all_dets() if worker is not None else []

    def camera_ids(self) -> list[str]:
        with self._lock:
            return list(self._workers)

    def fresh_snapshot(self, camera_id: str) -> dict | None:
        """The camera's latest snapshot, or None when absent/stale — for the map's
        detection twin (objects + people + frame size, all from one frame)."""
        with self._lock:
            worker = self._workers.get(camera_id)
        if worker is None:
            return None
        snap = worker.snapshot()
        if not _snapshot_fresh(snap):
            return None
        return snap
