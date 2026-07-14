"""Background zone renderer — turns the Backbone's per-camera observations into
per-zone snapshots for the dashboard's ZONE panels + COMMUNICATION cards.

Doctrine (CLAUDE.md): **isistream is the single source of ingestion + perception.
The dashboard renders; it never infers.** There is exactly ONE perception in the
system, and it is not here — this worker holds no detector, opens no CUDA session,
and runs no pose. It reads the Backbone's ``ObservationsMessage`` off the UDP bus,
rescales it to the hub frame, groups the detections into the operator's zone
polygons, and publishes one atomic snapshot the panels/cards/cam-views consume.

One daemon thread per camera:

  - reads that camera's latest observations from the bus,
  - groups the object detections into zone polygons by containment (a per-zone
    zone — deepest polygon-centre wins via ``cv2.pointPolygonTest(measureDist)``),
  - publishes ONE atomic snapshot ``{"frame_ts": ts, "zones": {zone_id: [dets]}}``
    (fresh dict, single assignment → readers never see a half-written state),
  - idles (empty snapshot, hub stream released) while the Backbone is not running.

Persons ride the same wire (with keypoints) — the map's people come from the SAME
perception as everything else, and are never boxed into a zone (humans are shown
by pose on the cam views only). The producer amortizes pose across ticks, so a
person-less tick is bridged with the last-seen people for ``_PEOPLE_BRIDGE_S``.
"""

from __future__ import annotations

import logging
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
import yaml

from .api.routes_zone_patches import load_patches
from .camera_hub import get_hub
from .yaml_cache import load_yaml_cached

logger = logging.getLogger(__name__)

# How long a published snapshot stays valid (floor). Older ⇒ consumers draw
# nothing (worker dead) instead of stale ghost boxes.
SNAPSHOT_MAX_AGE_S = 1.0

# Minimum sleep between render passes. NOT a display-rate cap (there is none):
# the worker is a pure renderer, paced by the camera and the producer's tick.
_RENDER_FLOOR_S = 0.02


def _snapshot_fresh(snap: dict) -> bool:
    valid = float(snap.get("valid_s", SNAPSHOT_MAX_AGE_S))
    return time.time() - snap.get("frame_ts", 0.0) <= max(SNAPSHOT_MAX_AGE_S, valid)


# An observation older than this is stale (backbone stopped/hiccuped) — the
# worker publishes empty rather than ghost boxes.
OBSERVATIONS_MAX_AGE_S = 2.0
# How long last-seen wire people persist on the snapshot when the producer's
# pose is amortized (pose_every_n ticks carry persons, the rest don't).
_PEOPLE_BRIDGE_S = 1.0

# Humans are rendered by POSE only on the cam views — any person-class detection
# is never grouped into a zone card so a person isn't boxed AND skeletoned.
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
    """One background thread per camera: renders the Backbone's per-camera
    observations into per-zone snapshots. Single writer; readers get atomic
    dicts. No detector, no CUDA, no pose — one perception, and it's not here."""

    def __init__(self, camera_id: str, src_cfg: dict, cfg, is_running,
                 hub_factory=get_hub, bus_getter=None):
        self.camera_id = camera_id
        self._src_cfg = dict(src_cfg)
        self._cfg = cfg
        self._is_running = is_running
        self._hub_factory = hub_factory
        self._bus_getter = bus_getter
        self._people_cache: tuple[float, list] = (0.0, [])
        self._patches: list[dict] = []
        self._snapshot: dict = {"frame_ts": 0.0, "zones": {}}
        self._stop = threading.Event()
        self._reload = threading.Event()
        self._thread: threading.Thread | None = None

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
        picks it up at the next iteration via the reload event."""
        self._patches = list(patches)
        if src_cfg is not None:
            self._src_cfg = dict(src_cfg)
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
        """The zone's last published health ("ok" when covered, else "")."""
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
        try:
            while not self._stop.is_set():
                if self._reload.is_set():
                    self._reload.clear()
                    patches = list(self._patches)
                    if stream is not None and self._src_cfg != acquired_cfg:
                        hub.release(stream)      # camera source changed → re-acquire
                        stream = None
                if not patches or not self._is_running():
                    # Idle: release the camera, publish empty so consumers render
                    # the raw feed (the pre-START state).
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
                if frame is None:
                    # No genuine frame yet (placeholder). Keep the snapshot alive:
                    # the panels still show the held frame, so its detections stay
                    # correct — don't blink the boxes off on RTSP jitter.
                    snap = self._snapshot
                    if snap.get("zones"):
                        self._snapshot = {**snap, "frame_ts": time.time()}
                    self._stop.wait(0.1)
                    continue
                try:
                    self._snapshot_from_bus(frame, patches)
                except Exception:
                    logger.warning("zone worker[%s]: render pass failed", self.camera_id,
                                   exc_info=True)
                # No operator-facing rate limit: the worker only RENDERS the
                # wire's observations onto the newest hub frame, so it is
                # bounded by the camera + the producer's tick. The small floor
                # keeps a frame-less moment from becoming a spin.
                self._stop.wait(_RENDER_FLOOR_S)
        finally:
            if stream is not None:
                hub.release(stream)

    def _snapshot_from_bus(self, frame, patches: list[dict]) -> None:
        """Render the wire's per-camera observations into per-zone snapshots.

        ONE perception — no dashboard inference. The Backbone's
        ``ObservationsMessage`` (calibration-frame coords) is rescaled to THIS
        hub frame, grouped into zone patches by polygon containment, and
        published as the exact snapshot shape the panels / cards / cam views
        consume. Persons ride the wire (with keypoints) — the map's people come
        from the same perception, and are never grouped into a zone card.
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
        wire_people: list = []
        if msg is not None and time.time() - float(msg.ts) <= OBSERVATIONS_MAX_AGE_S:
            fw, fh = msg.frame_wh
            sx, sy = iw / float(fw), ih / float(fh)
            for od in msg.dets:
                if str(od.cls).lower() in _PERSON_CLASSES:
                    # Persons ride the wire (with keypoints) — the map's people
                    # come from the SAME perception as everything else.
                    wire_people.append(
                        {"foot_uv": (od.foot_uv[0] * sx, od.foot_uv[1] * sy),
                         "confidence": float(od.confidence)})
                    continue                     # never grouped into zone cards
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
                # Zone membership by FOOT point (ground contact), matching the
                # metric definition of a floor zone and the cam view's
                # clip_to_zones. Twin polygons are FLOOR projections — flat on
                # the ground — so a tall object's bbox CENTRE floats above
                # them and the old centre test silently missed it.
                fx, fy = det.foot_uv
                for zid, poly in polys.items():
                    if poly is None or len(poly) < 3:
                        continue
                    if cv2.pointPolygonTest(
                            poly.astype(np.float32), (float(fx), float(fy)),
                            False) >= 0:
                        per_zone[zid].append(det)
                # dets outside every patch simply aren't shown — same as today.
        resolved = self._resolve_overlaps(per_zone, polys)

        # People: the producer's pose rides the observations echo — the wire is
        # the ONLY people source and this worker never builds a pose session. The
        # producer amortizes pose across ticks, so bridge person-less ticks with
        # the last-seen people for a short window.
        people: list = wire_people
        now_s = time.time()
        if wire_people:
            self._people_cache = (now_s, wire_people)
        else:
            cached_ts, cached = self._people_cache
            if now_s - cached_ts <= _PEOPLE_BRIDGE_S:
                people = cached

        self._snapshot = {"frame_ts": time.time(), "frame_wh": (iw, ih),
                          "zones": resolved, "status": statuses, "people": people,
                          "valid_s": SNAPSHOT_MAX_AGE_S}

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

    def __init__(self, cfg, is_running, *, hub_factory=get_hub, bus_getter=None):
        self._cfg = cfg
        self._is_running = is_running
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
            # Self-heal: a calibration switched outside the alignment endpoint
            # (isical export, mode change) leaves the stored twins projected
            # through the OLD geometry — regenerate before distributing them.
            from .api.routes_zone_patches import ensure_twins_current
            ensure_twins_current(self._cfg)
        except Exception:
            logger.debug("zone manager: twin freshness check failed", exc_info=True)
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
        path = self._cfg.backbone_config_path
        try:
            if not path.exists():
                return {}
            data = load_yaml_cached(path)
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
