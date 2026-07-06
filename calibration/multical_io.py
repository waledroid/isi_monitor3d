"""Parse Multical's output ``calibration.json``.

Multical (Oliver Batchelor, https://github.com/oliver-batchelor/multical) runs a
joint multi-camera ChArUco bundle adjustment and writes a JSON file whose
shape is documented here implicitly — by the schema this parser pins.

Multical's output structure (v0.4.0):

    {
      "cameras": {
        "<cam>": {
          "model": "standard",
          "image_size": [W, H],
          "K": 3x3 list,
          "dist": [k1, k2, p1, p2, k3]
        }
      },
      "camera_poses": {
        # When `master` is specified at export time:
        "<master>":              {"R": 3x3, "T": [3]},   # identity for master
        "<other>_to_<master>":   {"R": 3x3, "T": [3]},
        # When no `master`: absolute poses keyed by cam name.
        "<cam>": {"R": 3x3, "T": [3]}
      },
      "image_sets": { ... }    # ignored here
    }

**Pose convention (load-bearing).** Multical's exported ``camera_poses`` are
OpenCV-style **camera ← rig** extrinsics: ``p_cam = R @ p_rig + T`` (pinned
against multical's own ``project_points``, which projects world points through
``camera_pose`` with zero rvec/tvec). :class:`CameraInRig` stores the INVERSE —
**rig ← camera** (``p_rig = R_in_rig @ p_cam + t_in_rig``) — because that is
what every consumer (``_board_pose_in_rig``, ``compose_camera_in_world``, the
targetless path) composes with. The inversion happens once, in
``_resolve_poses``. Storing multical's matrices verbatim silently inverts every
non-master camera's pose: per-camera reprojection RMS cannot catch it, but the
same physical point observed by two cameras maps metres apart in world
coordinates (the c1 rig showed a constant 1.65 m cross-camera offset).

Multical does *not* write per-camera reprojection RMS into the JSON — it logs
it during optimization. RMS values are captured separately from Multical's
stdout/stderr in :mod:`calibration.calibrate`; this parser stores them only
if the caller provides them via :meth:`MultiCalSolution.with_rms`.

This module imports cleanly **without Multical installed** — its job is only
to read the JSON that Multical writes. It has no runtime dependency on
Multical itself.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

UNKNOWN_RMS_PX = -1.0
"""Sentinel: Multical's log did not yield a per-camera RMS for this camera."""


@dataclass(slots=True)
class CameraInRig:
    """One camera's intrinsics + pose in Multical's chosen rig frame.

    The rig frame is whichever camera Multical used as its master (or its
    bundle-adjustment reference if no master was passed at export). The
    floor-anchor phase composes ``(R_rig, t_rig)`` with the floor pose to
    yield the final world-frame ``(R, t)`` that goes into ``calibration.json``.
    """

    camera_id: str
    image_size_wh: tuple[int, int]
    K: np.ndarray
    D: np.ndarray
    R_in_rig: np.ndarray   # rig ← camera: p_rig = R_in_rig @ p_cam + t_in_rig
    t_in_rig: np.ndarray   # 3-vector (the camera's CENTER in rig coordinates)
    rms_px: float = UNKNOWN_RMS_PX


@dataclass(slots=True)
class MultiCalSolution:
    """Full parsed output of one Multical run."""

    master_camera: str
    cameras: dict[str, CameraInRig] = field(default_factory=dict)

    @property
    def camera_ids(self) -> tuple[str, ...]:
        return tuple(self.cameras.keys())

    def with_rms(self, rms_by_camera: dict[str, float]) -> MultiCalSolution:
        """Return a copy with RMS values filled in (from log parsing)."""
        new_cameras = {
            cam_id: replace(cam, rms_px=rms_by_camera.get(cam_id, cam.rms_px))
            for cam_id, cam in self.cameras.items()
        }
        return MultiCalSolution(master_camera=self.master_camera, cameras=new_cameras)


class MultiCalParseError(ValueError):
    """Raised when Multical's output JSON is missing fields we depend on."""


# ---------------------------------------------------------------------------
# JSON → MultiCalSolution
# ---------------------------------------------------------------------------


def parse(path: str | Path) -> MultiCalSolution:
    """Load and parse Multical's ``calibration.json``."""
    data = json.loads(Path(path).read_text())
    return from_dict(data)


def from_dict(data: dict) -> MultiCalSolution:
    """Parse an already-loaded Multical export dict.

    Strict about required fields; permissive about ``camera_poses`` key naming
    (handles both ``"<cam>"`` and ``"<other>_to_<master>"`` conventions).
    """
    if "cameras" not in data:
        raise MultiCalParseError("missing 'cameras' section in Multical output")
    if "camera_poses" not in data:
        raise MultiCalParseError("missing 'camera_poses' section in Multical output")

    cameras_block: dict = data["cameras"]
    poses_block: dict = data["camera_poses"]

    if not cameras_block:
        raise MultiCalParseError("Multical output has no cameras")

    poses_by_camera, master = _resolve_poses(set(cameras_block.keys()), poses_block)

    parsed: dict[str, CameraInRig] = {}
    for cam_id, cam in cameras_block.items():
        try:
            image_size = tuple(cam["image_size"])  # type: ignore[arg-type]
            K = np.asarray(cam["K"], dtype=np.float64)
            D = np.asarray(cam["dist"], dtype=np.float64).reshape(-1)
        except KeyError as exc:
            raise MultiCalParseError(f"camera {cam_id!r}: missing field {exc.args[0]!r}") from exc

        if K.shape != (3, 3):
            raise MultiCalParseError(f"camera {cam_id!r}: K is {K.shape}, expected (3, 3)")

        if cam_id not in poses_by_camera:
            raise MultiCalParseError(f"camera {cam_id!r}: no pose found in camera_poses")
        R, t = poses_by_camera[cam_id]

        parsed[cam_id] = CameraInRig(
            camera_id=cam_id,
            image_size_wh=(int(image_size[0]), int(image_size[1])),
            K=K,
            D=D,
            R_in_rig=R,
            t_in_rig=t,
        )

    return MultiCalSolution(master_camera=master, cameras=parsed)


def _invert_rt(R: np.ndarray, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Invert an OpenCV extrinsic: camera←rig ``(R, T)`` → rig←camera.

    Multical exports camera←rig; :class:`CameraInRig` stores rig←camera
    (see the module docstring — this single inversion is the convention seam).
    """
    Rt = R.T
    return Rt, -Rt @ T


def _resolve_poses(
    camera_ids: set[str],
    poses_block: dict,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], str]:
    """Normalize Multical's two key conventions into ``{cam_id: (R, t)}``,
    INVERTED into the rig←camera convention ``CameraInRig`` stores.

    Returns also the master camera identifier (the one whose pose is identity
    in the rig frame, or — when there is no master — an arbitrary deterministic
    pick from the available cameras).
    """
    suffix_re = re.compile(r"^(?P<other>.+)_to_(?P<master>.+)$")

    direct: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    relative: dict[str, tuple[str, np.ndarray, np.ndarray]] = {}

    for key, val in poses_block.items():
        R = np.asarray(val["R"], dtype=np.float64)
        T = np.asarray(val["T"], dtype=np.float64).reshape(3)
        if R.shape != (3, 3):
            raise MultiCalParseError(f"pose {key!r}: R is {R.shape}, expected (3, 3)")

        m = suffix_re.match(key)
        if m and m.group("master") in camera_ids and m.group("other") in camera_ids:
            relative[m.group("other")] = (m.group("master"), R, T)
        elif key in camera_ids:
            direct[key] = (R, T)
        # Unknown keys (e.g. board poses) are ignored.

    if relative:
        masters = {m for (m, _, _) in relative.values()}
        if len(masters) != 1:
            raise MultiCalParseError(
                f"camera_poses references multiple masters: {sorted(masters)}; expected one"
            )
        master = masters.pop()
        out: dict[str, tuple[np.ndarray, np.ndarray]] = {master: (np.eye(3), np.zeros(3))}
        for other, (_m, R, T) in relative.items():
            out[other] = _invert_rt(R, T)   # camera←rig → rig←camera
        return out, master

    if not direct:
        raise MultiCalParseError("camera_poses contains no usable entries")
    master = sorted(direct.keys())[0]
    return {cid: _invert_rt(R, T) for cid, (R, T) in direct.items()}, master


# ---------------------------------------------------------------------------
# Log → per-camera RMS
# ---------------------------------------------------------------------------


_RMS_LINE_RE = re.compile(
    r"(?P<cam>[\w\-.]+)\s*-\s*RMS:\s*(?P<rms>[0-9]+\.?[0-9]*)",
    re.IGNORECASE,
)

# The joint `multical calibrate` BA logs a single overall figure per iteration
# (`... reprojection RMS=1.621 (1.621), n=768 ...`) rather than the per-camera
# `<cam> - RMS:` lines the intrinsic solve emits. We take the final occurrence.
_JOINT_RMS_LINE_RE = re.compile(
    r"reprojection\s+RMS=(?P<rms>[0-9]+\.?[0-9]*)",
    re.IGNORECASE,
)


def parse_joint_rms_from_log(log_text: str) -> float | None:
    """Return the final overall reprojection RMS from a joint `calibrate` log.

    Multical's joint bundle adjustment reports one overall RMS (not per-camera),
    so the extrinsic solve shares this figure across cameras. Returns ``None`` if
    no joint-RMS line is present (e.g. an intrinsic-only log).
    """
    last: float | None = None
    for m in _JOINT_RMS_LINE_RE.finditer(log_text):
        last = float(m.group("rms"))
    return last


def parse_rms_from_log(log_text: str, camera_ids: tuple[str, ...]) -> dict[str, float]:
    """Best-effort extraction of per-camera reprojection RMS from Multical's log.

    Multical emits lines like ``INFO - <name> - RMS: 0.3142 quantiles: [...]``
    in its log output. We grep for those, keep only the ones whose ``<name>``
    is a known camera id, and return the **last** (i.e. final-iteration) value
    per camera. Cameras with no match are simply absent from the result —
    callers see :data:`UNKNOWN_RMS_PX` via the dataclass default.

    Args:
        log_text: combined stdout + stderr (or log-file contents) from Multical.
        camera_ids: known camera names; lines whose ``<name>`` is not in this
            set are ignored (they're typically board / pose entries).

    Returns:
        Mapping of camera id → final RMS in pixels.
    """
    valid = set(camera_ids)
    final: dict[str, float] = {}
    for line in log_text.splitlines():
        m = _RMS_LINE_RE.search(line)
        if not m:
            continue
        name = m.group("cam")
        if name not in valid:
            continue
        final[name] = float(m.group("rms"))   # later occurrences overwrite earlier
    return final
