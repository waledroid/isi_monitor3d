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

import cv2
import numpy as np

from .api.routes_zone_patches import load_patches, patch_pixel_box, patch_rect
from .camera_hub import get_hub
from .detection_overlay import (
    ZoneModelUnavailable,
    display_fps,
    get_pose_detector,
    get_zone_detector,
    read_backbone,
)

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
        mr = cv2.resize(d.mask.astype(np.float32), (cw, ch),
                        interpolation=cv2.INTER_LINEAR) >= 0.5
        full = np.zeros((ih, iw), dtype=bool)
        full[y0:y0 + ch, x0:x0 + cw] = mr
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
                 detector_factory=get_zone_detector, hub_factory=get_hub):
        self.camera_id = camera_id
        self._src_cfg = dict(src_cfg)
        self._cfg = cfg
        self._is_running = is_running
        self._detector_factory = detector_factory
        self._hub_factory = hub_factory
        self._patches: list[dict] = []
        self._snapshot: dict = {"frame_ts": 0.0, "zones": {}}
        self._stop = threading.Event()
        self._reload = threading.Event()
        self._thread: threading.Thread | None = None
        # Per-zone isolation state (worker-thread only — no locking needed):
        # circuit breaker {zone_id: (blocked_until_monotonic, reason)}, and the
        # per-zone cadence budget {zone_id: next_due_monotonic} + the last good
        # detections carried forward while a budgeted zone is not yet due.
        self._zone_breaker: dict[str, tuple[float, str]] = {}
        self._zone_next_due: dict[str, float] = {}
        self._zone_last_dets: dict[str, list] = {}

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

    def set_patches(self, patches: list[dict], src_cfg: dict | None = None) -> None:
        """Swap in a fresh zone list (and optionally a new camera source); the loop
        picks it up at the next iteration via the reload event. Clears the per-zone
        breaker/cadence state so a config save gives every zone a fresh chance."""
        self._patches = list(patches)
        if src_cfg is not None:
            self._src_cfg = dict(src_cfg)
        self._zone_breaker.clear()
        self._zone_next_due.clear()
        self._zone_last_dets.clear()
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
                try:
                    self._detect_all_zones(frame, patches)
                except Exception:
                    logger.warning("zone worker[%s]: detect pass failed", self.camera_id,
                                   exc_info=True)
                # Throttle to the configured display fps.
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
        status published — the OTHER zones keep detecting. A zone with a ``max_fps``
        budget only re-infers when due; in between, its last detections are carried
        forward so the overlay stays stable without burning GPU."""
        ih, iw = frame.shape[:2]
        ts = time.time()
        now = time.monotonic()
        det_cfg = read_backbone(self._cfg).get("detection") or {}
        global_conf = float(det_cfg.get("confidence_threshold", 0.3))
        per_zone: dict[str, list] = {}
        statuses: dict[str, str] = {}
        polys: dict[str, np.ndarray | None] = {}
        for patch in patches:
            zone_id = str(patch.get("id"))
            polys[zone_id] = _scaled_polygon(patch, (iw, ih))
            blocked_until, reason = self._zone_breaker.get(zone_id, (0.0, ""))
            if now < blocked_until:
                per_zone[zone_id], statuses[zone_id] = [], reason
                continue
            max_fps = patch.get("max_fps")
            if max_fps and now < self._zone_next_due.get(zone_id, 0.0):
                # Not due yet — carry the last good result forward (no inference).
                per_zone[zone_id] = list(self._zone_last_dets.get(zone_id, []))
                statuses[zone_id] = "ok"
                continue
            try:
                dets = self._detect_zone(frame, patch, iw, ih, global_conf)
            except ZoneModelUnavailable as exc:
                self._zone_breaker[zone_id] = (now + ZONE_RETRY_COOLDOWN_S, exc.reason)
                per_zone[zone_id], statuses[zone_id] = [], exc.reason
                logger.warning("zone worker[%s]: zone %s disabled for %.0fs (%s): %s",
                               self.camera_id, zone_id, ZONE_RETRY_COOLDOWN_S,
                               exc.reason, exc)
                continue
            except Exception:
                self._zone_breaker[zone_id] = (now + ZONE_RETRY_COOLDOWN_S, "error")
                per_zone[zone_id], statuses[zone_id] = [], "error"
                logger.warning("zone worker[%s]: zone %s failed — disabled for %.0fs",
                               self.camera_id, zone_id, ZONE_RETRY_COOLDOWN_S,
                               exc_info=True)
                continue
            self._zone_breaker.pop(zone_id, None)
            if max_fps:
                self._zone_next_due[zone_id] = now + 1.0 / max(0.1, float(max_fps))
            self._zone_last_dets[zone_id] = dets
            per_zone[zone_id], statuses[zone_id] = dets, "ok"
        resolved = self._resolve_overlaps(per_zone, polys)
        # People (full-frame pose, foot points in source px) — best-effort.
        people: list = []
        try:
            pose = get_pose_detector(self._cfg)
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
        self._snapshot = {"frame_ts": time.time(), "frame_wh": (iw, ih),
                          "zones": resolved, "status": statuses, "people": people,
                          "valid_s": valid_s}

    def _detect_zone(self, frame, patch: dict, iw: int, ih: int,
                     global_conf: float) -> list:
        """One zone on one frame → full-frame-coordinate detections (person-free,
        confidence-filtered, polygon-clipped). Detector build/inference failures
        PROPAGATE — the caller (_detect_all_zones) owns the per-zone circuit
        breaker, so one zone's failure never silently looks like 'no objects'."""
        from backbone.core.types import Frame, FramePair
        rect = patch_rect(patch)
        if rect is None:
            return []
        box = patch_pixel_box(rect, patch.get("frame_wh"), (iw, ih))
        if box is None:
            return []
        x0, y0, x1, y1 = box
        crop = frame[y0:y1, x0:x1]
        ch, cw = crop.shape[:2]
        infer_size = int(patch.get("infer_size") or 320)
        fed = crop
        longest = max(ch, cw)
        if longest > infer_size and longest > 0:
            s = infer_size / float(longest)
            fed = cv2.resize(crop, (max(1, round(cw * s)), max(1, round(ch * s))),
                             interpolation=cv2.INTER_AREA)
        detector = self._detector_factory(patch.get("model"), self._cfg, infer_size)
        fh, fw = fed.shape[:2]
        ts = time.time()
        pair = FramePair(capture_ts=ts, frame_idx=0,
                         frames={self.camera_id: Frame(camera_id=self.camera_id,
                                                       capture_ts=ts, frame_idx=0,
                                                       image=fed)})
        dets = list(detector.detect(pair).get(self.camera_id, []))
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
                 detector_factory=get_zone_detector, hub_factory=get_hub):
        self._cfg = cfg
        self._is_running = is_running
        self._detector_factory = detector_factory
        self._hub_factory = hub_factory
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
