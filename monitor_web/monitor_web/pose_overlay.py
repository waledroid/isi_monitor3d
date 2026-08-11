"""Person-pose display sources for the dashboard — ZERO dashboard inference.

CPU deployment branch: the perception producer (isistream) runs the OpenVINO
pose model and its skeletons ride the observations echo; ``WirePoseSource``
renders them and ``WireObjectSource`` renders the wire's object boxes. The
in-dashboard ORT pose engines of the GPU line (``PoseEngine`` /
``AsyncPoseRunner``) were removed with the onnxruntime dependency — the
smoothing/foot helpers remain because the wire sources use them.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from types import SimpleNamespace

import cv2
import numpy as np

logger = logging.getLogger(__name__)

LEFT_ANKLE, RIGHT_ANKLE = 15, 16
_COCO_SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4), (3, 5), (4, 6),
]


@dataclass(slots=True)
class Pose:
    box_xyxy: np.ndarray   # (4,) x1,y1,x2,y2 in original-image px
    score: float
    keypoints: np.ndarray  # (K,3) x,y,conf in original-image px
    foot_uv: tuple[float, float]


def _centroid(p: Pose) -> np.ndarray:
    b = p.box_xyxy
    return np.array([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0])


def _lerp_pose(prev: Pose, target: Pose, alpha: float) -> Pose:
    """Move ``prev`` toward ``target`` by ``alpha`` (display smoothing only —
    confidences always come from the target, positions are blended)."""
    kp = target.keypoints.copy()
    kp[:, :2] = prev.keypoints[:, :2] + alpha * (target.keypoints[:, :2]
                                                 - prev.keypoints[:, :2])
    box = prev.box_xyxy + alpha * (target.box_xyxy - prev.box_xyxy)
    fu = (prev.foot_uv[0] + alpha * (target.foot_uv[0] - prev.foot_uv[0]),
          prev.foot_uv[1] + alpha * (target.foot_uv[1] - prev.foot_uv[1]))
    return Pose(box_xyxy=box, score=target.score, keypoints=kp, foot_uv=fu)


def _extrapolate(latest: list[Pose], prev: list[Pose], dt_pair_s: float,
                 age_s: float, *, max_age_s: float, snap_px: float) -> list[Pose]:
    """Motion-compensate the newest pose result for its AGE.

    The rendered frame is *now*; the newest result describes where each person
    was ``age_s`` ago. Estimate per-person velocity from the last two results
    (nearest-centroid matching) and project the keypoints/box forward by the
    (clamped) age — so the skeleton lands on the person, not behind them.
    Persons without a trustworthy match (new, teleported, or a degenerate pair
    interval) pass through unextrapolated.
    """
    if not latest:
        return []
    age = max(0.0, min(age_s, max_age_s))
    if not prev or age <= 0.0 or not (0.02 <= dt_pair_s <= 1.0):
        return list(latest)
    prev_c = [_centroid(p) for p in prev]
    used: set[int] = set()
    out: list[Pose] = []
    for t in latest:
        tc = _centroid(t)
        best, best_d = None, float("inf")
        for i, pc in enumerate(prev_c):
            if i in used:
                continue
            d = float(np.hypot(*(tc - pc)))
            if d < best_d:
                best, best_d = i, d
        if best is None or best_d > snap_px:
            out.append(t)                        # new person / fast mover: as-is
            continue
        used.add(best)
        p = prev[best]
        scale = age / dt_pair_s
        kp = t.keypoints.copy()
        kp[:, :2] += (t.keypoints[:, :2] - p.keypoints[:, :2]) * scale
        box = t.box_xyxy + (t.box_xyxy - p.box_xyxy) * scale
        fu = (t.foot_uv[0] + (t.foot_uv[0] - p.foot_uv[0]) * scale,
              t.foot_uv[1] + (t.foot_uv[1] - p.foot_uv[1]) * scale)
        out.append(Pose(box_xyxy=box, score=t.score, keypoints=kp, foot_uv=fu))
    return out


def _advance_smoothing(prev: list[Pose], target: list[Pose], dt_s: float,
                       *, tau_s: float, snap_px: float) -> list[Pose]:
    """One smoothing step: match previous smoothed poses to the newest result
    by nearest box centroid and blend; unmatched or far-jumped targets snap."""
    if not target:
        return []
    if not prev:
        return list(target)
    alpha = 1.0 - float(np.exp(-max(0.0, min(dt_s, 0.25)) / tau_s))
    prev_c = [_centroid(p) for p in prev]
    used: set[int] = set()
    out: list[Pose] = []
    for t in target:
        tc = _centroid(t)
        best, best_d = None, float("inf")
        for i, pc in enumerate(prev_c):
            if i in used:
                continue
            d = float(np.hypot(*(tc - pc)))
            if d < best_d:
                best, best_d = i, d
        if best is None or best_d > snap_px:
            out.append(t)                       # new person / fast mover: snap
        else:
            used.add(best)
            out.append(_lerp_pose(prev[best], t, alpha))
    return out


def _foot(keypoints: np.ndarray, box: np.ndarray, kpt_conf: float) -> tuple[float, float]:
    vis = [keypoints[i] for i in (LEFT_ANKLE, RIGHT_ANKLE) if keypoints[i, 2] >= kpt_conf]
    if vis:
        return float(np.mean([k[0] for k in vis])), float(np.mean([k[1] for k in vis]))
    return float((box[0] + box[2]) / 2.0), float(box[3])


class WirePoseSource:
    """Skeletons from the bus observations — ZERO dashboard inference.

    Direction 1: the perception producer's pose rides the observations echo
    (person dets with ``keypoints_uv``, calibration-frame px). This source
    duck-types ``AsyncPoseRunner.predict`` for ``annotate_frame``: scale the
    wire keypoints to the display frame and return them as :class:`Pose`
    objects.

    FLUIDITY: the wire updates at the perception tick rate (~15 Hz) while
    panels render at the camera rate (18-24 fps), so raw wire poses look
    steppy. This source therefore reuses the AsyncPoseRunner's display
    machinery on the wire results: motion-compensate the newest result for
    its age (per-person velocity from the last two results, projected to
    *now*) and exponentially smooth toward that moving target — identical
    constants, identical feel, still zero inference.
    """

    _MAX_AGE_S = 2.0      # observations older than this ⇒ no skeletons
    _BRIDGE_S = 1.0       # hold last people across person-less ticks
    _SMOOTH_TAU_S = 0.08
    _SNAP_PX = 120.0
    _EXTRAP_MAX_S = 0.35

    def __init__(self, bus_getter, camera_id: str) -> None:
        self._bus_getter = bus_getter
        self._camera_id = camera_id
        self._held: tuple[float, float, list[Pose]] = (0.0, 0.0, [])  # (msg_ts, seen_at, poses)
        self._prev: tuple[float, list[Pose]] = (0.0, [])   # previous wire result
        self._cur: tuple[float, list[Pose]] = (0.0, [])    # newest wire result
        self._smoothed: list[Pose] = []
        self._smooth_ts = 0.0

    def stop(self) -> None:      # lifecycle parity with AsyncPoseRunner
        self._held = (0.0, 0.0, [])
        self._prev = (0.0, [])
        self._cur = (0.0, [])
        self._smoothed = []

    def draw(self, image: np.ndarray, poses: list[Pose],
             kpt_conf: float = 0.35) -> None:
        """Same skeleton rendering as ``PoseEngine.draw`` — wire keypoints
        carry the producer's per-joint confidences, so low-conf joints hide
        exactly as they do for locally-inferred poses."""
        for p in poses:
            for a, b in _COCO_SKELETON:
                if p.keypoints[a, 2] >= kpt_conf and p.keypoints[b, 2] >= kpt_conf:
                    cv2.line(image, tuple(p.keypoints[a, :2].astype(int)),
                             tuple(p.keypoints[b, :2].astype(int)), (255, 180, 0), 2)
            for j in range(p.keypoints.shape[0]):
                if p.keypoints[j, 2] >= kpt_conf:
                    cv2.circle(image, tuple(p.keypoints[j, :2].astype(int)), 3, (0, 0, 255), -1)
            cv2.circle(image, (int(p.foot_uv[0]), int(p.foot_uv[1])), 6, (0, 255, 255), -1)

    def predict(self, frame_bgr: np.ndarray) -> list[Pose]:
        bus = self._bus_getter() if self._bus_getter is not None else None
        if bus is None:
            return []
        try:
            msg = bus.snapshot().observations_by_camera.get(self._camera_id)
        except Exception:
            return []
        now = time.time()
        if msg is None or now - float(msg.ts) > self._MAX_AGE_S:
            return []
        fh, fw = frame_bgr.shape[:2]
        mw, mh = msg.frame_wh
        sx, sy = fw / float(mw), fh / float(mh)
        poses: list[Pose] = []
        for od in msg.dets:
            kps = getattr(od, "keypoints_uv", None)
            if kps is None or str(od.cls).lower() != "person":
                continue
            k = np.asarray(kps, dtype=np.float32).reshape(-1, 3)
            k[:, 0] *= sx
            k[:, 1] *= sy
            x0, y0, x1, y1 = od.bbox_xyxy
            poses.append(Pose(
                box_xyxy=np.array([x0 * sx, y0 * sy, x1 * sx, y1 * sy],
                                  dtype=np.float32),
                score=float(od.confidence),
                keypoints=k,
                foot_uv=(float(od.foot_uv[0]) * sx, float(od.foot_uv[1]) * sy)))
        _held_ts, held_seen, _held_poses = self._held
        if poses:
            self._held = (float(msg.ts), now, poses)
            if float(msg.ts) > self._cur[0]:
                # A genuinely new wire result: shift the pair used for the
                # per-person velocity estimate.
                self._prev = self._cur
                self._cur = (float(msg.ts), poses)
        elif now - held_seen > self._BRIDGE_S:
            # Persistently person-less: the scene really is empty.
            self._prev = (0.0, [])
            self._cur = (0.0, [])
            self._smoothed = []
            return []

        cur_ts, cur_poses = self._cur
        if not cur_poses:
            return []
        # Motion-compensate the newest wire result for its age, then smooth
        # toward that moving target — the exact AsyncPoseRunner recipe, on
        # wire data: skeletons glide at panel rate while pose ticks at ~15 Hz.
        target = _extrapolate(
            cur_poses, self._prev[1],
            cur_ts - self._prev[0],
            now - cur_ts,
            max_age_s=self._EXTRAP_MAX_S, snap_px=self._SNAP_PX)
        self._smoothed = _advance_smoothing(
            self._smoothed, target, now - self._smooth_ts,
            tau_s=self._SMOOTH_TAU_S, snap_px=self._SNAP_PX)
        self._smooth_ts = now
        return list(self._smoothed)


# Person classes ride WirePoseSource (skeletons); the object source drops them.
_WIRE_PERSON_CLASSES = frozenset({"person", "people", "human", "pedestrian"})


class WireObjectSource:
    """Object detections straight from the bus observations — ZERO inference.

    Points mode (Direction 1): isistream detects only inside the floor zones and
    echoes per-camera ``ObservationsMessage``s (calibration/producer-frame px).
    The cam view draws those object dets DIRECTLY — with NO dependency on the
    pixel-space ``zone_patches`` that gate the ZONE PANELS. So boxes appear
    wherever isistream detects, even when ``zone_patches.yaml`` is empty (without
    this, the patch-scoped ``ZoneDetectionWorker`` renders nothing when no patch
    is drawn, so a perfectly good detection never reaches the cam view).

    Returns detections duck-typed for :func:`annotate_frame` — the same
    ``SimpleNamespace`` shape ``ZoneDetectionWorker._snapshot_from_bus`` produces,
    minus the per-zone grouping. Persons are dropped (they ride
    :class:`WirePoseSource`).
    """

    _MAX_AGE_S = 2.0     # observations older than this ⇒ no boxes (no stale ghosts)

    def __init__(self, bus_getter, camera_id: str) -> None:
        self._bus_getter = bus_getter
        self._camera_id = camera_id

    def objects(self, frame_bgr: np.ndarray) -> list:
        bus = self._bus_getter() if self._bus_getter is not None else None
        if bus is None:
            return []
        try:
            msg = bus.snapshot().observations_by_camera.get(self._camera_id)
        except Exception:
            return []
        if msg is None or time.time() - float(msg.ts) > self._MAX_AGE_S:
            return []
        fh, fw = frame_bgr.shape[:2]
        mw, mh = msg.frame_wh
        sx, sy = fw / float(mw), fh / float(mh)
        out: list = []
        for od in msg.dets:
            if str(od.cls).lower() in _WIRE_PERSON_CLASSES:
                continue
            x0, y0, x1, y1 = od.bbox_xyxy
            out.append(SimpleNamespace(
                camera_id=self._camera_id, capture_ts=float(msg.ts),
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
            ))
        return out
