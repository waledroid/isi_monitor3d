"""Live capture session — per-camera auto-snap threads + the MJPEG frame buffer.

`CaptureManager` holds the single active `CaptureSession` (one rig at a time). A
session opens each configured camera (RTSP/USB via backbone.ingestion), runs a
detection+gate loop per camera, keeps the latest ANNOTATED JPEG for the live view,
and auto-snaps RAW frames to the phase directory:

  * intrinsic — each camera snaps INDEPENDENTLY into intrinsic/<cam>/.
  * extrinsic — a SYNCHRONIZED pair is snapped (all cams gate-pass within a short
    window) into extrinsic/<cam>/, so the multi-AprilGrid views line up.

The source/detector imports are lazy so tests can inject a stub source and so the
module imports without GStreamer/Multical present.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from .detect import (
    AprilTagDetector,
    CharucoBoardDetector,
    SnapGate,
    draw_hud,
    preprocess_for_tags,
)

_PAIR_WINDOW_S = 0.5      # both cameras must gate-pass within this window for a pair
FLOOR_TARGET = 3          # distinct flat ChArUco placements (synchronized pairs) wanted

# Targetless texture-quality readout tunables (cheap per-frame; NOT SuperPoint).
_TEXTURE_MIN_FEATURES = 150   # below this a view is flagged low-texture
_TEXTURE_MIN_BLUR_VAR = 60.0  # Laplacian variance floor (sharpness)


def texture_score(frame_bgr: np.ndarray, *, max_features: int = 800) -> dict:
    """Cheap per-frame texture/feature readout for the targetless live view.

    A rich, sharp scene gives the SuperPoint+LightGlue solve something to match, so
    the operator needs feedback WHILE aiming — but running the heavy ONNX matcher per
    live frame is far too slow. Instead we use OpenCV FAST corner count (feature
    density) + Laplacian variance (sharpness), both microsecond-cheap. Returns
    ``{"features": int, "blur_var": float, "ok": bool}`` where ``ok`` means the view
    looks textured + sharp enough to be worth capturing.
    """
    gray = (cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            if frame_bgr.ndim == 3 else frame_bgr)
    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    try:
        fast = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)
        kps = fast.detect(gray, None)
        features = min(len(kps), max_features)
    except Exception:                                    # pragma: no cover - cv2 guard
        features = 0
    ok = features >= _TEXTURE_MIN_FEATURES and blur_var >= _TEXTURE_MIN_BLUR_VAR
    return {"features": int(features), "blur_var": round(blur_var, 1), "ok": bool(ok)}


def _charuco_detector(charuco_spec, cap) -> CharucoBoardDetector:
    """Build a ChArUco detector with the capture spec's CLAHE knobs (default ON).

    Reuses the same ``tag_clahe*`` knobs the AprilTag path uses — a CLAHE contrast
    boost helps a flat, distant, low-contrast floor board detect (the "0/4" gate)
    without moving corners, so it is safe for the floor-anchor solve too.
    """
    return CharucoBoardDetector(
        charuco_spec,
        clahe=getattr(cap, "tag_clahe", True),
        clahe_clip=getattr(cap, "tag_clahe_clip", 2.0),
        clahe_grid=getattr(cap, "tag_clahe_grid", 8),
    )


def _open_source(cam_spec, camera_id: str):
    """Open a frame source for a CameraSpec (lazy import; RTSP or USB)."""
    if cam_spec.type == "usb":
        from backbone.ingestion.v4l2 import V4l2FrameSource
        return V4l2FrameSource(camera_id=camera_id, device=cam_spec.device)
    from backbone.ingestion.rtsp import RtspFrameSource
    return RtspFrameSource(camera_id=camera_id, url=cam_spec.url, latency_ms=100)


def wipe_phase_captures(project_dir: Path, phase: str, cameras: list[str]) -> int:
    """Delete captured images for a phase (used by Restart). Returns files removed.

    Floor restart wipes only the NEW ``floor/<cam>/*.jpg`` dir layout — a legacy
    single ``floor/<cam>.jpg`` file is left untouched (additive: never lose data).
    """
    sub = {"intrinsic": "intrinsic", "floor": "floor"}.get(phase, "extrinsic")
    removed = 0
    for cid in cameras:
        for f in (Path(project_dir) / sub / cid).glob("*.jpg"):
            f.unlink(missing_ok=True)
            removed += 1
    return removed


def grab_floor_shot(project_dir: Path, cfg, camera_id: str, *,
                    source_factory=_open_source, settle_frames: int = 8) -> dict:
    """Grab ONE ChArUco-on-floor shot for a camera → floor/<cam>.jpg (world anchor).

    Opens the source, lets it settle a few frames, then keeps the first frame with
    enough ChArUco corners (≥4 — the floor anchor's requirement). Raises if no board
    is detected. Used by the Extrinsic phase's "capture floor shot" button.
    """
    from ..core.project import charuco_spec
    detector = _charuco_detector(charuco_spec(cfg.board), cfg.capture)
    cam_spec = cfg.cameras[camera_id]
    source = source_factory(cam_spec, camera_id)
    source.start()
    best = None
    try:
        for i, frame in enumerate(source.frames()):
            if i < settle_frames:
                continue
            det = detector.detect(frame.image)
            if det.n >= 4:
                best = (frame.image, det.n)
                break
            if i > settle_frames + 60:        # ~a couple seconds of trying
                break
    finally:
        try:
            source.stop()
        except Exception:
            pass
    if best is None:
        raise ValueError("no ChArUco board detected on the floor — lay the ChArUco board FLAT on "
                         "the floor (it defines the ground plane); keep it flat, well-lit and free "
                         "of glare, and move it closer if it is not detected, then try again")
    out = Path(project_dir) / "floor" / f"{camera_id}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), best[0])
    return {"camera": camera_id, "corners": best[1], "path": str(out)}


class FloorPreview:
    """A single-camera live ChArUco preview for aiming the floor-anchor shot.

    Opens ONE camera via `_open_source`, runs the ChArUco detector in a daemon
    thread (keeping the latest annotated JPEG for the MJPEG view AND the latest
    raw frame), and exposes `grab()` to write floor/<cam>.jpg from the same source
    — so the live preview and the grab never double-open the camera (the source of
    the old 409 deadlock). The operator sees corner feedback while laying the board
    FLAT on the floor, then captures.
    """

    def __init__(self, project_dir: Path, cfg, camera_id: str, *,
                 source_factory=None) -> None:
        from ..core.project import charuco_spec
        if source_factory is None:
            source_factory = _open_source           # resolve module global at call time
        self.project_dir = Path(project_dir)
        self.cfg = cfg
        self.camera_id = camera_id
        self._detector = _charuco_detector(charuco_spec(cfg.board), cfg.capture)
        self._cam_spec = cfg.cameras[camera_id]
        self._source_factory = source_factory
        self._latest_jpeg: bytes | None = None
        self._latest_good: tuple[np.ndarray, int] | None = None   # raw frame w/ ≥4 corners
        self.status = "starting"
        self.last_det_n = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name=f"isical-floor-{camera_id}")

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._latest_jpeg

    def grab(self) -> dict:
        """Write the latest well-detected frame to floor/<cam>.jpg (≥4 corners)."""
        with self._lock:
            good = self._latest_good
        if good is None:
            raise ValueError("no ChArUco board detected on the floor — lay the ChArUco board FLAT on "
                             "the floor (it defines the ground plane); keep it flat, well-lit and free "
                             "of glare, and move it closer if it is not detected, then try again")
        out = self.project_dir / "floor" / f"{self.camera_id}.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), good[0])
        return {"camera": self.camera_id, "corners": int(good[1]), "path": str(out)}

    def _run(self) -> None:
        try:
            source = self._source_factory(self._cam_spec, self.camera_id)
            source.start()
        except Exception as exc:
            self.status = f"camera error: {exc}"
            return
        self.status = "live"
        try:
            for frame in source.frames():
                if self._stop.is_set():
                    break
                raw = frame.image
                det = self._detector.detect(raw)
                self.last_det_n = det.n
                ok_board = det.n >= 4
                annotated = self._detector.annotate(raw, det)
                annotated = draw_hud(annotated, count=det.n, target=4,
                                     status=("board OK — ready to capture" if ok_board
                                             else "lay the ChArUco FLAT on the floor"),
                                     ok=ok_board)
                ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                with self._lock:
                    if ok_board:
                        self._latest_good = (raw, det.n)
                    if ok:
                        self._latest_jpeg = buf.tobytes()
        except Exception as exc:                                # source died mid-stream
            self.status = f"stream ended: {exc}"
        finally:
            try:
                source.stop()
            except Exception:
                pass
            if not self.status.startswith("camera error"):
                self.status = "stopped"


def _write_shot_meta(jpg_path: Path, det) -> None:
    """Persist per-shot quality next to the jpg (powers the Studio gallery).

    Schema: {"corners": int, "centroid": [x, y] | null, "blur_var": float},
    centroid normalized to [0, 1]. Best-effort: never raises into the capture loop.
    """
    centroid = getattr(det, "centroid", None)
    meta = {
        "corners": int(getattr(det, "n", 0) or 0),
        "centroid": [float(centroid[0]), float(centroid[1])] if centroid else None,
        "blur_var": float(getattr(det, "blur_var", 0.0) or 0.0),
    }
    try:
        jpg_path.with_suffix(".json").write_text(json.dumps(meta))
    except OSError:
        pass


class _CamWorker:
    """One camera's capture thread: stream → detect → annotate → gate → (snap)."""

    def __init__(self, session: CaptureSession, camera_id: str, cam_spec,
                 out_dir: Path, source_factory) -> None:
        self.session = session
        self.camera_id = camera_id
        self.cam_spec = cam_spec
        self.out_dir = out_dir
        self._source_factory = source_factory
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.count = len(list(self.out_dir.glob("*.jpg")))
        self.status = "starting"
        self.last_det_n = 0
        self._latest_jpeg: bytes | None = None
        self._ready: tuple[float, np.ndarray] | None = None   # extrinsic pair staging
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name=f"isical-cap-{camera_id}")

        cap = session.cfg.capture
        if session.phase == "intrinsic":
            self._detector = _charuco_detector(session.charuco_spec, cap)
            self._min = cap.min_charuco_corners
            self.target = cap.target_per_camera
        elif session.phase == "floor":
            # Floor anchor: synchronized ChArUco pairs (board flat on the floor,
            # both cams see the SAME placement). Gate on ≥4 ChArUco corners (the
            # floor-anchor solve's requirement), snap a novel placement each time.
            self._detector = _charuco_detector(session.charuco_spec, cap)
            self._min = max(4, cap.min_charuco_corners)
            self.target = session.floor_target
        else:
            self._detector = AprilTagDetector(
                quad_decimate=getattr(cap, "tag_quad_decimate", 1.0),
                clahe=getattr(cap, "tag_clahe", True),
                clahe_clip=getattr(cap, "tag_clahe_clip", 2.0),
                clahe_grid=getattr(cap, "tag_clahe_grid", 8),
            )
            self._min = cap.min_april_tags
            self.target = cap.extrinsic_target
        self._gate = SnapGate(min_detections=self._min, blur_min_var=cap.blur_min_var,
                              steady_max_motion=cap.steady_max_motion,
                              novelty_min_dist=cap.novelty_min_dist)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._latest_jpeg

    # ---- extrinsic pair coordination ----
    def take_ready(self) -> tuple[float, np.ndarray] | None:
        with self._lock:
            r, self._ready = self._ready, None
            return r

    def peek_ready_ts(self) -> float | None:
        with self._lock:
            return self._ready[0] if self._ready else None

    def _save(self, raw: np.ndarray, det) -> None:
        idx = self.count
        path = self.out_dir / f"{self.camera_id}_{idx:03d}.jpg"
        cv2.imwrite(str(path), self._image_for_disk(raw))
        _write_shot_meta(path, det)
        self.count += 1
        self._gate.note_kept(det)

    def _image_for_disk(self, raw: np.ndarray) -> np.ndarray:
        """Image actually written to disk. For extrinsic shots, persist the SAME
        CLAHE/grayscale preprocessing the gate detected on, so Multical re-detects
        tags on identical pixels (what you capture is what solves). Intrinsic shots
        stay raw BGR (ChArUco intrinsics want the original image)."""
        if self.session.phase != "extrinsic":
            return raw
        cap = self.session.cfg.capture
        if not getattr(cap, "tag_clahe", True):
            return raw
        return preprocess_for_tags(
            raw, clahe=True,
            clip=getattr(cap, "tag_clahe_clip", 2.0),
            grid=getattr(cap, "tag_clahe_grid", 8),
        )

    def _run(self) -> None:
        try:
            source = self._source_factory(self.cam_spec, self.camera_id)
            source.start()
        except Exception as exc:
            self.status = f"camera error: {exc}"
            return
        self.status = "live"
        try:
            for frame in source.frames():
                if self._stop.is_set():
                    break
                raw = frame.image
                det = self._detector.detect(raw)
                self.last_det_n = det.n
                verdict = self._gate.evaluate(det)
                done = self.count >= self.target
                if verdict.snap and not done:
                    if self.session.phase == "intrinsic":
                        self._save(raw, det)
                        status = f"snap! {det.n} corners"
                    else:                                   # stage for a synchronized pair
                        with self._lock:
                            self._ready = (time.time(), raw)
                        self.session.try_snap_pair()
                        unit = "corners" if self.session.phase == "floor" else "tags"
                        status = f"ready ({det.n} {unit})"
                else:
                    status = "captured ✓" if done else verdict.reason
                annotated = self._detector.annotate(raw, det)
                annotated = draw_hud(annotated, count=self.count, target=self.target,
                                     status=status, ok=verdict.snap or done)
                ok, buf = cv2.imencode(".jpg", annotated,
                                       [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    with self._lock:
                        self._latest_jpeg = buf.tobytes()
        except Exception as exc:                            # source died mid-stream
            self.status = f"stream ended: {exc}"
        finally:
            try:
                source.stop()
            except Exception:
                pass
            if not self.status.startswith("camera error"):
                self.status = "stopped"


class CaptureSession:
    """A live capture run for one project + phase across its configured cameras."""

    def __init__(self, project_dir: Path, cfg, phase: str, *,
                 cameras: list[str] | None = None, source_factory=_open_source) -> None:
        from ..core.project import charuco_spec
        self.project_dir = Path(project_dir)
        self.cfg = cfg
        self.phase = phase                                  # "intrinsic" | "extrinsic" | "floor"
        self.charuco_spec = charuco_spec(cfg.board)
        # Small target of distinct flat placements for the floor plane fit (both cams).
        self.floor_target = FLOOR_TARGET
        self._pair_lock = threading.Lock()
        self.pair_count = 0
        configured = cfg.configured_cameras()
        cams = [c for c in (cameras or configured) if c in configured]
        sub = {"intrinsic": "intrinsic", "floor": "floor"}.get(phase, "extrinsic")
        self.workers: dict[str, _CamWorker] = {
            cid: _CamWorker(self, cid, cfg.cameras[cid],
                            self.project_dir / sub / cid, source_factory)
            for cid in cams
        }
        if phase in ("extrinsic", "floor"):
            self.pair_count = min((w.count for w in self.workers.values()), default=0)

    def start(self) -> None:
        for w in self.workers.values():
            w.start()

    def stop(self) -> None:
        for w in self.workers.values():
            w.stop()

    def latest_jpeg(self, camera_id: str) -> bytes | None:
        w = self.workers.get(camera_id)
        return w.latest_jpeg() if w else None

    def try_snap_pair(self) -> None:
        """Extrinsic/floor: if ALL cameras are staged within the window, commit the pair."""
        if self.phase not in ("extrinsic", "floor") or len(self.workers) < 2:
            # single-camera extrinsic is degenerate; treat each ready as a solo save
            for w in self.workers.values():
                r = w.take_ready()
                if r:
                    w._save(r[1], _Stub())
            return
        with self._pair_lock:
            tss = [w.peek_ready_ts() for w in self.workers.values()]
            if any(t is None for t in tss):
                return
            if max(tss) - min(tss) > _PAIR_WINDOW_S:
                return
            for w in self.workers.values():
                r = w.take_ready()
                if r:
                    w._save(r[1], _Stub())
            self.pair_count += 1

    def status(self) -> dict:
        return {
            "phase": self.phase,
            "pair_count": self.pair_count,
            "cameras": {cid: {"count": w.count, "target": w.target,
                              "status": w.status, "detections": w.last_det_n}
                        for cid, w in self.workers.items()},
        }


class _Stub:
    """A no-detection stand-in for pair saves (centroid novelty already gated)."""
    centroid = None


class _TargetlessWorker:
    """One camera's live thread for the targetless flow: stream → texture readout.

    Unlike the board ``_CamWorker`` there is NO board detector and NO auto-snap gate
    — targetless scenes have no calibration target. The worker only keeps the latest
    RAW frame (for the manual pair capture) and the latest ANNOTATED JPEG with a
    live texture/feature readout HUD. Snapping is driven externally by
    ``TargetlessSession.capture_pair`` (a manual, synchronized both-cam trigger).
    """

    def __init__(self, camera_id: str, cam_spec, out_dir: Path, source_factory) -> None:
        self.camera_id = camera_id
        self.cam_spec = cam_spec
        self.out_dir = out_dir
        self._source_factory = source_factory
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.count = len(list(self.out_dir.glob("*.jpg")))
        self.status = "starting"
        self.texture: dict = {"features": 0, "blur_var": 0.0, "ok": False}
        self._latest_jpeg: bytes | None = None
        self._latest_raw: np.ndarray | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name=f"isical-tl-{camera_id}")

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._latest_jpeg

    def latest_raw(self) -> np.ndarray | None:
        with self._lock:
            return None if self._latest_raw is None else self._latest_raw.copy()

    def save(self, raw: np.ndarray) -> Path:
        path = self.out_dir / f"{self.camera_id}_{self.count:03d}.jpg"
        cv2.imwrite(str(path), raw)
        self.count += 1
        return path

    def _run(self) -> None:
        try:
            source = self._source_factory(self.cam_spec, self.camera_id)
            source.start()
        except Exception as exc:
            self.status = f"camera error: {exc}"
            return
        self.status = "live"
        try:
            for frame in source.frames():
                if self._stop.is_set():
                    break
                raw = frame.image
                tex = texture_score(raw)
                label = (f"texture OK — {tex['features']} features"
                         if tex["ok"] else
                         f"low texture — {tex['features']} features (aim at a richer scene)")
                annotated = draw_hud(raw.copy(), count=self.count, target=self.count,
                                     status=label, ok=tex["ok"])
                ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                with self._lock:
                    self.texture = tex
                    self._latest_raw = raw
                    if ok:
                        self._latest_jpeg = buf.tobytes()
        except Exception as exc:                            # source died mid-stream
            self.status = f"stream ended: {exc}"
        finally:
            try:
                source.stop()
            except Exception:
                pass
            if not self.status.startswith("camera error"):
                self.status = "stopped"


class TargetlessSession:
    """A live targetless-extrinsics capture: both cameras live + manual pair snaps.

    Fully independent of the board/AprilGrid ``CaptureSession`` — it never touches
    ``extrinsic/``. Shows BOTH cameras with a live texture readout and, on a manual
    ``capture_pair`` trigger, snaps the latest raw frame from every camera at once
    into ``targetless/{cam}/`` (a synchronized stereo pair). Move the rig / change
    the scene between snaps to build a set of textured pairs for the solve.
    """

    def __init__(self, project_dir: Path, cfg, *, source_factory=_open_source) -> None:
        self.project_dir = Path(project_dir)
        self.cfg = cfg
        self.phase = "targetless"
        self._pair_lock = threading.Lock()
        configured = cfg.configured_cameras()
        self.workers: dict[str, _TargetlessWorker] = {
            cid: _TargetlessWorker(cid, cfg.cameras[cid],
                                   self.project_dir / "targetless" / cid, source_factory)
            for cid in configured
        }
        self.pair_count = min((w.count for w in self.workers.values()), default=0)

    def start(self) -> None:
        for w in self.workers.values():
            w.start()

    def stop(self) -> None:
        for w in self.workers.values():
            w.stop()

    def latest_jpeg(self, camera_id: str) -> bytes | None:
        w = self.workers.get(camera_id)
        return w.latest_jpeg() if w else None

    def capture_pair(self) -> dict:
        """Snap the latest raw frame from every camera at once (one stereo pair).

        Manual trigger (no board gate). If any camera has no frame yet, nothing is
        written and a human ``reason`` is returned so the UI can prompt.
        """
        with self._pair_lock:
            grabbed = {cid: w.latest_raw() for cid, w in self.workers.items()}
            missing = [cid for cid, raw in grabbed.items() if raw is None]
            if missing:
                return {"captured": False, "pair_count": self.pair_count,
                        "reason": f"no frame yet from {missing} — wait for the live view"}
            files = {cid: str(self.workers[cid].save(raw))
                     for cid, raw in grabbed.items()}
            self.pair_count += 1
            return {"captured": True, "pair_count": self.pair_count, "files": files}

    def status(self) -> dict:
        return {
            "phase": self.phase,
            "pair_count": self.pair_count,
            "cameras": {cid: {"count": w.count, "status": w.status,
                              "texture": w.texture}
                        for cid, w in self.workers.items()},
        }


class CaptureManager:
    """Holds the single active CaptureSession (one rig calibrated at a time)."""

    def __init__(self) -> None:
        self._session: CaptureSession | None = None
        self._key: tuple[str, str] | None = None          # (project, phase)
        self._floor: FloorPreview | None = None
        self._floor_key: tuple[str, str] | None = None    # (project, camera_id)
        self._targetless: TargetlessSession | None = None
        self._targetless_key: str | None = None           # project
        self._lock = threading.Lock()

    def start(self, project: str, project_dir: Path, cfg, phase: str,
              cameras: list[str] | None = None, **kw) -> dict:
        with self._lock:
            if self._session is not None:
                self._session.stop()
            self._stop_targetless_locked()
            self._session = CaptureSession(project_dir, cfg, phase, cameras=cameras, **kw)
            self._key = (project, phase)
            self._session.start()
            return self._session.status()

    def stop_current(self) -> None:
        with self._lock:
            if self._session is not None:
                self._session.stop()
            self._session = None
            self._key = None

    def stop_all(self) -> None:
        self.stop_current()
        self.stop_floor()
        self.stop_targetless()

    # ---- targetless capture (self-contained textured-scene stereo pairs) ----
    def start_targetless(self, project: str, project_dir: Path, cfg, **kw) -> dict:
        """Open both cameras for targetless capture. Stops any board session /
        floor preview first (cameras are exclusive)."""
        with self._lock:
            if self._session is not None:
                self._session.stop()
                self._session = None
                self._key = None
            if self._floor is not None:
                self._floor.stop()
                self._floor = None
                self._floor_key = None
            self._stop_targetless_locked()
            self._targetless = TargetlessSession(project_dir, cfg, **kw)
            self._targetless_key = project
            self._targetless.start()
            return self._targetless.status()

    def _stop_targetless_locked(self) -> None:
        if self._targetless is not None:
            self._targetless.stop()
        self._targetless = None
        self._targetless_key = None

    def stop_targetless(self) -> None:
        with self._lock:
            self._stop_targetless_locked()

    def targetless(self, project: str) -> TargetlessSession | None:
        with self._lock:
            if self._targetless is None or self._targetless_key != project:
                return None
            return self._targetless

    def targetless_mjpeg(self, project: str, camera_id: str) -> Iterator[bytes]:
        """Multipart MJPEG generator for a targetless live view (annotated frames)."""
        boundary = b"--frame"
        while True:
            sess = self.targetless(project)
            if sess is None:
                break
            jpeg = sess.latest_jpeg(camera_id)
            if jpeg:
                yield (boundary + b"\r\nContent-Type: image/jpeg\r\n"
                       + f"Content-Length: {len(jpeg)}\r\n\r\n".encode() + jpeg + b"\r\n")
            time.sleep(0.05)

    # ---- floor-anchor live preview (extrinsic): a single-camera aiming view ----
    def start_floor(self, project: str, project_dir: Path, cfg, camera_id: str,
                    **kw) -> None:
        """Open a single-camera ChArUco preview for floor aiming. Stops the full
        capture session first (cameras are exclusive). Re-targeting a different
        camera replaces the previous preview."""
        with self._lock:
            if self._session is not None:
                self._session.stop()
                self._session = None
                self._key = None
            if self._floor is not None:
                self._floor.stop()
            self._floor = FloorPreview(project_dir, cfg, camera_id, **kw)
            self._floor_key = (project, camera_id)
            self._floor.start()

    def stop_floor(self) -> None:
        with self._lock:
            if self._floor is not None:
                self._floor.stop()
            self._floor = None
            self._floor_key = None

    def floor(self, project: str, camera_id: str | None = None) -> FloorPreview | None:
        with self._lock:
            if self._floor is None or self._floor_key is None:
                return None
            if self._floor_key[0] != project:
                return None
            if camera_id is not None and self._floor_key[1] != camera_id:
                return None
            return self._floor

    def floor_mjpeg(self, project: str, camera_id: str) -> Iterator[bytes]:
        """Multipart MJPEG generator for the floor-aiming preview (annotated)."""
        boundary = b"--frame"
        while True:
            fp = self.floor(project, camera_id)
            if fp is None:
                break
            jpeg = fp.latest_jpeg()
            if jpeg:
                yield (boundary + b"\r\nContent-Type: image/jpeg\r\n"
                       + f"Content-Length: {len(jpeg)}\r\n\r\n".encode() + jpeg + b"\r\n")
            time.sleep(0.05)

    def active(self, project: str, phase: str | None = None) -> CaptureSession | None:
        with self._lock:
            if self._session is None or self._key is None:
                return None
            if self._key[0] != project:
                return None
            if phase is not None and self._key[1] != phase:
                return None
            return self._session

    def mjpeg(self, project: str, camera_id: str) -> Iterator[bytes]:
        """Multipart MJPEG generator for the live view (annotated frames)."""
        boundary = b"--frame"
        while True:
            sess = self.active(project)
            if sess is None:
                break
            jpeg = sess.latest_jpeg(camera_id)
            if jpeg:
                yield (boundary + b"\r\nContent-Type: image/jpeg\r\n"
                       + f"Content-Length: {len(jpeg)}\r\n\r\n".encode() + jpeg + b"\r\n")
            time.sleep(0.05)
