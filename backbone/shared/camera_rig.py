"""``CameraRig`` — the single source of truth for camera geometry at runtime.

Everything downstream of calibration (homography projection, cross-camera
fusion, triangulation, reprojection gating) reads ``K, D, R, t, H, P`` from
here. If a value is wrong, the bug is upstream in ``calibration.json``, not
in the consumer.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from calibration.schema import CalibrationFile, CameraCalibration


class CameraRig:
    """Loaded calibration for all cameras in one node.

    Hold a single instance per process; consumers receive references, never
    copies. The class is intentionally read-only: there is no setter for
    matrices, no recalibration at runtime — that path goes through
    ``calibrate.py`` and a process restart.
    """

    def __init__(self, calibration: CalibrationFile) -> None:
        self._raw = calibration
        self._cameras: dict[str, _CameraView] = {
            cam_id: _CameraView(cal) for cam_id, cal in calibration.cameras.items()
        }

    @classmethod
    def from_file(cls, path: str | Path) -> CameraRig:
        return cls(CalibrationFile.read(Path(path)))

    @property
    def camera_ids(self) -> tuple[str, ...]:
        return tuple(self._cameras.keys())

    @property
    def floor_anchor_method(self) -> str:
        return self._raw.floor_anchor_method

    @property
    def calibration_mode(self) -> str:
        """``"multical_full"`` (Mode 2) or ``"single_cam_4pt"`` (Mode 1)."""
        return self._raw.calibration_mode

    def __contains__(self, camera_id: object) -> bool:
        return camera_id in self._cameras

    def __getitem__(self, camera_id: str) -> _CameraView:
        try:
            return self._cameras[camera_id]
        except KeyError as exc:
            raise KeyError(
                f"camera_id {camera_id!r} not in calibration "
                f"(available: {list(self._cameras)})"
            ) from exc

    def items(self) -> Mapping[str, _CameraView]:
        return self._cameras


class _CameraView:
    """Read-only NumPy view over one camera's calibration entry.

    All arrays are cached as ``float64`` NumPy arrays on first access and
    handed out by reference. Do not mutate the returned arrays — they are
    shared across all consumers.
    """

    __slots__ = ("_D", "_H", "_K", "_P", "_R", "_cal", "_t")

    def __init__(self, cal: CameraCalibration) -> None:
        self._cal = cal
        self._K = cal.K_np()
        self._D = cal.D_np()
        self._R = cal.R_np()
        self._t = cal.t_np()
        self._H = cal.H_np()
        self._P = cal.P_np()
        self._K.setflags(write=False)
        self._D.setflags(write=False)
        self._R.setflags(write=False)
        self._t.setflags(write=False)
        self._H.setflags(write=False)
        self._P.setflags(write=False)

    @property
    def camera_id(self) -> str:
        return self._cal.camera_id

    @property
    def image_size_wh(self) -> tuple[int, int]:
        return self._cal.image_size_wh

    @property
    def K(self) -> np.ndarray:
        return self._K

    @property
    def D(self) -> np.ndarray:
        return self._D

    @property
    def R(self) -> np.ndarray:
        return self._R

    @property
    def t(self) -> np.ndarray:
        return self._t

    @property
    def H(self) -> np.ndarray:
        return self._H

    @property
    def P(self) -> np.ndarray:
        return self._P

    @property
    def reprojection_rms_px(self) -> float:
        return self._cal.reprojection_rms_px
