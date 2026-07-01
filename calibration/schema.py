"""Schema for ``calibration.json``.

Produced once per node by ``calibrate.py``. Consumed at startup by
``backbone.shared.camera_rig.CameraRig``. Holds everything both geometric
methods (homography and triangulation) need; the two cannot drift independently
because they are derived from the same intrinsics + extrinsics here.

The on-disk format is intentionally plain JSON (no pickle, no h5) so that
calibrations survive Python upgrades and can be diffed in git.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

CALIBRATION_VERSION = 1
"""Bumped on any breaking change to this schema. Loaders must reject older versions."""


@dataclass(slots=True)
class CameraCalibration:
    """All per-camera quantities needed by the Backbone at runtime.

    ``H`` is the homography from pixels to the shared floor frame (meters).
    ``P = K @ [R | t]`` is the full projection matrix used by triangulation.
    Both are derived from the same ``K, D, R, t`` and stored explicitly so
    the runtime never has to recompute them.
    """

    camera_id: str
    image_size_wh: tuple[int, int]
    K: list[list[float]]            # 3x3 intrinsic matrix
    D: list[float]                  # distortion coefficients (k1, k2, p1, p2, k3, ...)
    R: list[list[float]]            # 3x3 rotation, world ← camera
    t: list[float]                  # 3-vector translation, world ← camera (meters)
    H: list[list[float]]            # 3x3 floor-plane homography, pixel → world (meters)
    P: list[list[float]]            # 3x4 full projection, world → pixel
    reprojection_rms_px: float

    def K_np(self) -> np.ndarray:
        return np.asarray(self.K, dtype=np.float64)

    def D_np(self) -> np.ndarray:
        return np.asarray(self.D, dtype=np.float64)

    def R_np(self) -> np.ndarray:
        return np.asarray(self.R, dtype=np.float64)

    def t_np(self) -> np.ndarray:
        return np.asarray(self.t, dtype=np.float64).reshape(3)

    def H_np(self) -> np.ndarray:
        return np.asarray(self.H, dtype=np.float64)

    def P_np(self) -> np.ndarray:
        return np.asarray(self.P, dtype=np.float64)


CALIBRATION_MODE_MULTICAL_FULL = "multical_full"
CALIBRATION_MODE_SINGLE_CAM_4PT = "single_cam_4pt"
"""How this calibration was produced. Two operational modes live here:

* ``multical_full`` — joint multi-camera bundle adjustment via Multical. All
  per-camera ``K``, ``D``, ``R``, ``t``, ``H``, ``P`` matrices are real;
  triangulation works.
* ``single_cam_4pt`` — single-camera 4-point floor-plane homography. Only
  ``H`` is real; ``K`` defaults to identity, ``D`` to zeros, ``R`` to identity,
  ``t`` to zeros, ``P`` to ``K @ [R|t]``. Downstream homography (``pixel_to_floor``)
  works untouched because ``undistort_points`` with ``D=0`` is identity.
  Triangulation is unavailable in this mode — the orchestrator skips it.
"""


@dataclass(slots=True)
class CalibrationFile:
    """Top-level structure of ``calibration.json``."""

    version: int
    created_at: str
    floor_anchor_method: str        # "charuco_floor" | "planefit" | "tape_measured_points" | "4pt_floor"
    floor_origin_note: str          # human-readable, e.g. "ChArUco board, lower-left corner"
    cameras: dict[str, CameraCalibration] = field(default_factory=dict)
    calibration_mode: str = CALIBRATION_MODE_MULTICAL_FULL

    def to_json(self) -> str:
        return json.dumps(_to_jsonable(self), indent=2, sort_keys=False)

    def write(self, path: Path) -> None:
        path.write_text(self.to_json())

    @classmethod
    def read(cls, path: Path) -> CalibrationFile:
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def from_dict(cls, d: dict) -> CalibrationFile:
        version = int(d.get("version", 0))
        if version != CALIBRATION_VERSION:
            raise CalibrationVersionError(
                f"calibration.json version {version}, expected {CALIBRATION_VERSION}"
            )
        cameras = {
            cam_id: CameraCalibration(
                camera_id=cam_id,
                image_size_wh=tuple(cam["image_size_wh"]),  # type: ignore[arg-type]
                K=cam["K"],
                D=cam["D"],
                R=cam["R"],
                t=cam["t"],
                H=cam["H"],
                P=cam["P"],
                reprojection_rms_px=float(cam["reprojection_rms_px"]),
            )
            for cam_id, cam in d["cameras"].items()
        }
        return cls(
            version=version,
            created_at=d["created_at"],
            floor_anchor_method=d["floor_anchor_method"],
            floor_origin_note=d.get("floor_origin_note", ""),
            cameras=cameras,
            calibration_mode=d.get("calibration_mode", CALIBRATION_MODE_MULTICAL_FULL),
        )


class CalibrationVersionError(ValueError):
    """Raised when calibration.json was written by an incompatible schema version."""


def _to_jsonable(obj):
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj
