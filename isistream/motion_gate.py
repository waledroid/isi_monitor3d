"""Motion gate — skip inference on static scenes, keep emitting the truth.

A warehouse is static most of the time. The nano models are launch-overhead-
bound, so the real lever is running them LESS, not faster: per camera, a
cheap grayscale signature decides whether anything visibly changed since the
last inference — unchanged ⇒ the detector is skipped and the last detections
(boxes AND mask polygons) are re-emitted with the new frame's capture_ts, so
downstream sees an uninterrupted observation stream (tracks stay alive,
occupancy stays voted, panels keep their masks). A forced re-inference every
``refresh_s`` self-heals anything the signature misses (gradual light drift,
sub-threshold creep).

Two independent gates per camera:
- **objects** — one signature per ZONE CROP (motion outside the zones must
  not wake the object detector);
- **pose** — one full-frame signature (people can be anywhere).

Signature: the region downscaled to ``sig_px``² gray int16; "changed" =
more than ``frac`` of pixels moved by more than ``pix_delta`` gray levels
(robust to sensor/compression noise). ~0.1 ms per region — three orders of
magnitude cheaper than the inference it saves.
"""

from __future__ import annotations

import cv2
import numpy as np

SIG_PX = 32
PIX_DELTA = 15
FRAC = 0.02


def _signature(region: np.ndarray) -> np.ndarray | None:
    if region.size == 0 or region.shape[0] < 4 or region.shape[1] < 4:
        return None
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if region.ndim == 3 else region
    return cv2.resize(gray, (SIG_PX, SIG_PX),
                      interpolation=cv2.INTER_AREA).astype(np.int16)


def _changed(a: np.ndarray | None, b: np.ndarray | None) -> bool:
    if a is None or b is None or a.shape != b.shape:
        return True
    return float(np.mean(np.abs(a - b) > PIX_DELTA)) > FRAC


class MotionGate:
    """Per-camera object/pose inference gate."""

    def __init__(
        self,
        crop_boxes: dict[str, list],          # cam_id → [(zone_name, (x0,y0,x1,y1))] in CALIBRATION px
        calib_wh: dict[str, tuple[int, int]],
        *,
        refresh_s: float = 2.0,
    ) -> None:
        self._boxes = crop_boxes
        self._calib_wh = calib_wh
        self._refresh_s = float(refresh_s)
        # cam → list of sigs at last OBJECT inference / full-frame sig at last POSE
        self._obj_sigs: dict[str, list] = {}
        self._pose_sig: dict[str, np.ndarray | None] = {}
        self._obj_last: dict[str, float] = {}
        self._pose_last: dict[str, float] = {}
        # Operator-visible counters.
        self.obj_skips = 0
        self.pose_skips = 0

    def _crop_sigs(self, cam_id: str, image: np.ndarray) -> list:
        fh, fw = image.shape[:2]
        cw, ch = self._calib_wh.get(cam_id, (fw, fh))
        sx, sy = fw / float(cw), fh / float(ch)
        sigs = []
        for _zone, (x0, y0, x1, y1) in self._boxes.get(cam_id, []):
            fx0, fy0 = max(0, int(x0 * sx)), max(0, int(y0 * sy))
            fx1, fy1 = min(fw, int(x1 * sx) + 1), min(fh, int(y1 * sy) + 1)
            sigs.append(_signature(image[fy0:fy1, fx0:fx1]))
        return sigs

    def objects_due(self, cam_id: str, image: np.ndarray, now_s: float) -> bool:
        """True ⇒ run the object detector on this camera (and the gate arms
        the new reference signatures); False ⇒ re-emit the cached detections."""
        if now_s - self._obj_last.get(cam_id, 0.0) >= self._refresh_s:
            self._obj_sigs[cam_id] = self._crop_sigs(cam_id, image)
            self._obj_last[cam_id] = now_s
            return True
        sigs = self._crop_sigs(cam_id, image)
        ref = self._obj_sigs.get(cam_id)
        if ref is None or len(ref) != len(sigs) or any(
                _changed(a, b) for a, b in zip(ref, sigs, strict=False)):
            self._obj_sigs[cam_id] = sigs
            self._obj_last[cam_id] = now_s
            return True
        self.obj_skips += 1
        return False

    def pose_due(self, cam_id: str, image: np.ndarray, now_s: float) -> bool:
        if now_s - self._pose_last.get(cam_id, 0.0) >= self._refresh_s:
            self._pose_sig[cam_id] = _signature(image)
            self._pose_last[cam_id] = now_s
            return True
        sig = _signature(image)
        if _changed(self._pose_sig.get(cam_id), sig):
            self._pose_sig[cam_id] = sig
            self._pose_last[cam_id] = now_s
            return True
        self.pose_skips += 1
        return False
