"""Person-pose overlay for the dashboard preview — runs a YOLO-pose ONNX and
draws the skeleton + the foot node (ankle midpoint) alongside the detection
boxes drawn by ``detection_overlay``.

Independent of the trainer package (onnxruntime + OpenCV); the ORT session is
built via the shared ``backbone.shared.ort_session`` helper so it gets the same
memory-safe arena options as every other session. Same raw-head decode the
trainer's PoseOnnxInferencer uses: head ``(1, 4 + nc + K*3, A)`` → person boxes +
``[K, 3]`` keypoints.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np
from backbone.shared.ort_session import build_onnx_session

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
except ImportError as exc:  # pragma: no cover
    raise ImportError("onnxruntime required for pose overlay") from exc

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


class PoseEngine:
    """Lazy YOLO-pose ONNX runner (CUDA → CPU)."""

    def __init__(self, model_path: str, conf: float = 0.3, kpt_conf: float = 0.3,
                 device: str | None = None, imgsz: int | None = None) -> None:
        self.conf, self.kpt_conf = float(conf), float(kpt_conf)
        providers = (["CPUExecutionProvider"] if device == "cpu"
                     or "CUDAExecutionProvider" not in ort.get_available_providers()
                     else ["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.session = build_onnx_session(model_path, providers=providers)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        _, _, h, w = inp.shape
        # Dynamic exports carry SYMBOLIC spatial dims ('height'/'width') —
        # letterbox needs concrete numbers; `imgsz` (detection.pose_imgsz)
        # picks the runtime size there, else YOLO-pose's canonical 640. A
        # STATIC export keeps its baked size — imgsz can't apply.
        fallback = int(imgsz) if imgsz else 640
        self.h = h if isinstance(h, int) else fallback
        self.w = w if isinstance(w, int) else fallback
        out_c = self.session.get_outputs()[0].shape[1]
        self.k = (out_c - 4 - 1) // 3 if isinstance(out_c, int) else 17   # single person class
        self._lb = (1.0, 0, 0)
        logger.info("pose overlay: loaded %s (in=%sx%s, K=%s, %s)", model_path,
                    self.w, self.h, self.k, self.session.get_providers()[0])

    def _letterbox(self, frame: np.ndarray):
        oh, ow = frame.shape[:2]
        r = min(self.h / oh, self.w / ow)
        nw, nh = round(ow * r), round(oh * r)
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        pw, ph = self.w - nw, self.h - nh
        left, top = pw // 2, ph // 2
        padded = cv2.copyMakeBorder(resized, top, ph - top, left, pw - left,
                                    cv2.BORDER_CONSTANT, value=(114, 114, 114))
        self._lb = (r, left, top)
        return padded

    def predict(self, frame_bgr: np.ndarray) -> list[Pose]:
        # BGR→RGB via channel flip fused into the float conversion — this
        # OpenCV build's cvtColor has a ~8 ms fixed dispatch cost per call
        # (see backbone.detection.preprocess).
        padded = self._letterbox(frame_bgr)
        tensor = padded[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32, order="C")
        tensor *= 1.0 / 255.0
        out = self.session.run(None, {self.input_name: tensor})[0]
        return self._decode(out, frame_bgr.shape[1], frame_bgr.shape[0])

    def _decode(self, raw: np.ndarray, ow: int, oh: int) -> list[Pose]:
        preds = raw[0].T                      # (A, 4 + 1 + K*3)
        scores = preds[:, 4].astype(np.float32)
        keep = scores >= self.conf
        if not np.any(keep):
            return []
        preds, scores = preds[keep], scores[keep]
        cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        boxes = np.column_stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]).astype(np.float32)
        kpts = preds[:, 5:5 + self.k * 3].reshape(-1, self.k, 3).astype(np.float32)
        xywh = np.column_stack([boxes[:, 0], boxes[:, 1],
                                boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]]).tolist()
        kept = cv2.dnn.NMSBoxes(xywh, scores.tolist(), self.conf, 0.45)
        if len(kept) == 0:
            return []
        kept = np.array(kept).flatten()
        boxes, scores, kpts = boxes[kept], scores[kept], kpts[kept]
        r, px, py = self._lb
        boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - px) / r, 0, ow)
        boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - py) / r, 0, oh)
        kpts[:, :, 0] = (kpts[:, :, 0] - px) / r
        kpts[:, :, 1] = (kpts[:, :, 1] - py) / r
        return [Pose(b, float(s), kp, _foot(kp, b, self.kpt_conf))
                for b, s, kp in zip(boxes, scores, kpts, strict=True)]

    def draw(self, image: np.ndarray, poses: list[Pose]) -> None:
        """Draw skeleton + keypoints + foot node in place (no person bounding box)."""
        for p in poses:
            for a, b in _COCO_SKELETON:
                if p.keypoints[a, 2] >= self.kpt_conf and p.keypoints[b, 2] >= self.kpt_conf:
                    cv2.line(image, tuple(p.keypoints[a, :2].astype(int)),
                             tuple(p.keypoints[b, :2].astype(int)), (255, 180, 0), 2)
            for j in range(self.k):
                if p.keypoints[j, 2] >= self.kpt_conf:
                    cv2.circle(image, tuple(p.keypoints[j, :2].astype(int)), 3, (0, 0, 255), -1)
            cv2.circle(image, (int(p.foot_uv[0]), int(p.foot_uv[1])), 6, (0, 255, 255), -1)


class AsyncPoseRunner:
    """``predict``/``draw``-compatible wrapper that DECOUPLES pose inference
    from the video loop.

    Running the pose model synchronously per rendered frame chains the video
    rate to the model's latency — under GPU contention with the live Backbone
    the whole cam view dropped to ~4 fps. Here a daemon worker runs the engine
    on the NEWEST submitted frame at whatever rate the GPU allows, and
    ``predict`` returns the latest completed result instantly: the video stays
    at camera rate, skeletons refresh at the pose-achievable rate. Results
    older than ``_STALE_S`` clear (no frozen skeletons); the worker parks after
    ``_IDLE_STOP_S`` without frames (viewer closed) and restarts on demand.
    """

    _STALE_S = 2.0
    _IDLE_STOP_S = 30.0
    # Display smoothing: inference completes at ~10-15 Hz while the video
    # renders at 17-24 fps, so a raw skeleton visibly "steps" every couple of
    # frames. Each predict() (= one rendered frame) advances a smoothed copy
    # toward the newest result with a time-based exponential (tau below); a
    # person whose match moved further than _SNAP_PX snapped instead (fast
    # motion / new person must not rubber-band across the frame).
    _SMOOTH_TAU_S = 0.08
    _SNAP_PX = 120.0
    # Motion compensation: the newest result describes where a person WAS
    # (inference latency + queue wait ≈ 80-150 ms under GPU contention) — a
    # walking body moves 20-40 px in that window, so the raw skeleton draws
    # visibly BESIDE the person. Velocity from the last two results projects
    # it forward by the result's age, capped so a stalled worker never
    # slingshots a skeleton across the frame.
    _EXTRAP_MAX_S = 0.35

    def __init__(self, engine: PoseEngine) -> None:
        self.engine = engine
        self._cond = threading.Condition()
        self._pending: np.ndarray | None = None
        self._pending_ts = 0.0
        self._poses: list[Pose] = []
        self._result_ts = 0.0
        self._result_frame_ts = 0.0     # submit time of the frame behind _poses
        self._prev_poses: list[Pose] = []
        self._prev_frame_ts = 0.0
        self._smoothed: list[Pose] = []
        self._smooth_ts = 0.0
        self._stopped = False
        self._thread: threading.Thread | None = None

    def stop(self) -> None:
        """Tear the runner down NOW: exit the worker, drop the engine ref (its
        CUDA session frees on the next gc), clear cached poses. Idempotent —
        called by ``reset_detector()`` on backbone STOP / model change so pose
        VRAM never outlives the run."""
        with self._cond:
            self._stopped = True
            self._pending = None
            self._poses = []
            self._prev_poses = []
            self._smoothed = []
            self._cond.notify_all()
        t = self._thread
        if t is not None and t.is_alive():
            # Brief join only — the worker is a daemon and exits at its next
            # loop check; STOP must not pay for it. The engine ref drop below
            # releases the CUDA session on the following gc pass either way.
            t.join(timeout=0.2)
        self._thread = None
        self.engine = None

    @property
    def kpt_conf(self) -> float:
        eng = self.engine
        return eng.kpt_conf if eng is not None else 0.3

    def predict(self, frame_bgr: np.ndarray) -> list[Pose]:
        """Submit the frame for background inference; return the latest result."""
        now = time.time()
        with self._cond:
            if self._stopped:
                return []
            self._pending = frame_bgr
            self._pending_ts = now
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._run, daemon=True, name="pose-async")
                self._thread.start()
            self._cond.notify()
            if now - self._result_ts > self._STALE_S:
                self._smoothed = []
                return []
            # Motion-compensate the (aged) result to *now*, then smooth toward
            # that moving target — smoothing kills inter-result jitter without
            # trailing, because the target itself tracks the person.
            target = _extrapolate(
                self._poses, self._prev_poses,
                self._result_frame_ts - self._prev_frame_ts,
                now - self._result_frame_ts,
                max_age_s=self._EXTRAP_MAX_S, snap_px=self._SNAP_PX)
            self._smoothed = _advance_smoothing(
                self._smoothed, target, now - self._smooth_ts,
                tau_s=self._SMOOTH_TAU_S, snap_px=self._SNAP_PX)
            self._smooth_ts = now
            return list(self._smoothed)

    def draw(self, image: np.ndarray, poses: list[Pose]) -> None:
        eng = self.engine
        if eng is not None:
            eng.draw(image, poses)

    def _run(self) -> None:
        idle_since = time.time()
        while True:
            with self._cond:
                while self._pending is None:
                    if self._stopped:
                        return                      # torn down (reset_detector)
                    self._cond.wait(timeout=1.0)
                    if self._pending is None and time.time() - idle_since > self._IDLE_STOP_S:
                        return                      # viewer gone — free the GPU
                if self._stopped:
                    return
                frame = self._pending
                frame_ts = self._pending_ts
                self._pending = None
                engine = self.engine
            idle_since = time.time()
            if engine is None:
                return
            try:
                poses = engine.predict(frame)
            except Exception:
                logger.warning("pose overlay: async predict failed", exc_info=True)
                poses = []
            with self._cond:
                if self._stopped:
                    return
                # Keep the previous result + its frame time: the pair is the
                # velocity estimate that motion-compensates the render.
                self._prev_poses = self._poses
                self._prev_frame_ts = self._result_frame_ts
                self._poses = poses
                self._result_frame_ts = frame_ts
                self._result_ts = time.time()


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
