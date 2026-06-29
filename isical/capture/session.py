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


def _open_source(cam_spec, camera_id: str):
    """Open a frame source for a CameraSpec (lazy import; RTSP or USB)."""
    if cam_spec.type == "usb":
        from backbone.ingestion.v4l2 import V4l2FrameSource
        return V4l2FrameSource(camera_id=camera_id, device=cam_spec.device)
    from backbone.ingestion.rtsp import RtspFrameSource
    return RtspFrameSource(camera_id=camera_id, url=cam_spec.url, latency_ms=100)


def wipe_phase_captures(project_dir: Path, phase: str, cameras: list[str]) -> int:
    """Delete captured images for a phase (used by Restart). Returns files removed."""
    sub = "intrinsic" if phase == "intrinsic" else "extrinsic"
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
    detector = CharucoBoardDetector(charuco_spec(cfg.board))
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
        raise ValueError("no ChArUco board detected on the floor — place the ChArUco board on "
                         "the floor LEANED ~20-40° (not flat — a leaned board gives a better "
                         "PnP pose and detects more reliably at distance) and try again")
    out = Path(project_dir) / "floor" / f"{camera_id}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), best[0])
    return {"camera": camera_id, "corners": best[1], "path": str(out)}


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
            self._detector = CharucoBoardDetector(session.charuco_spec)
            self._min = cap.min_charuco_corners
            self.target = cap.target_per_camera
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
                        status = f"ready ({det.n} tags)"
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
        self.phase = phase                                  # "intrinsic" | "extrinsic"
        self.charuco_spec = charuco_spec(cfg.board)
        self._pair_lock = threading.Lock()
        self.pair_count = 0
        configured = cfg.configured_cameras()
        cams = [c for c in (cameras or configured) if c in configured]
        sub = "intrinsic" if phase == "intrinsic" else "extrinsic"
        self.workers: dict[str, _CamWorker] = {
            cid: _CamWorker(self, cid, cfg.cameras[cid],
                            self.project_dir / sub / cid, source_factory)
            for cid in cams
        }
        if phase == "extrinsic":
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
        """Extrinsic: if ALL cameras are staged within the window, commit the pair."""
        if self.phase != "extrinsic" or len(self.workers) < 2:
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


class CaptureManager:
    """Holds the single active CaptureSession (one rig calibrated at a time)."""

    def __init__(self) -> None:
        self._session: CaptureSession | None = None
        self._key: tuple[str, str] | None = None          # (project, phase)
        self._lock = threading.Lock()

    def start(self, project: str, project_dir: Path, cfg, phase: str,
              cameras: list[str] | None = None, **kw) -> dict:
        with self._lock:
            if self._session is not None:
                self._session.stop()
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
