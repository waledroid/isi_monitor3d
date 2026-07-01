"""Offline calibration tool — produces ``calibration.json``.

Primary path is :func:`calibrate_all` (CLI: ``calibrate-all``): one shot, all
cameras, joint bundle adjustment via **Multical** (in its own venv at
``calibration/.venv-multical/``, bootstrapped by ``setup_multical.sh``). The
flow is:

    ChArUco shots (one folder per camera)
        ↓  Multical  (joint intrinsics + extrinsics BA, outlier rejection)
        ↓  multical_io.parse  (load Multical's JSON)
        ↓  floor-anchor shot per camera + cv2.solvePnP  (rig → world)
        ↓  derive H, P from K, R, t  (backbone.shared.geometry)
        ↓  CalibrationFile.write

For 2-camera rigs there is also a **two-stage** path (:func:`calibrate_two_stage`,
CLI ``calibrate-2cam``): per-camera intrinsics from a ChArUco board
(``multical intrinsic``) followed by joint extrinsics from a multi-AprilGrid
target with those intrinsics fixed (``multical calibrate --fix_intrinsic``). It
uses each board where it is strong — ChArUco's dense corners for intrinsics, a
wide AprilGrid target both cameras see at once for extrinsics — and still writes
the same ``calibration.json``. Generate printable A4 boards with ``gen-boards``
(:func:`generate_board_images`).

Operators can inspect any result in Multical's built-in 3D viewer (camera + board
poses, per-view reprojection): pass ``--vis`` to ``calibrate-all`` / ``calibrate-2cam``
to auto-open it after the solve, or re-open a saved run with ``vis --workspace
<work_dir>`` (:func:`run_multical_vis`). The viewer needs the Qt/PyVista deps
(``setup_multical.sh`` installs them) and a display; it never affects the written
``calibration.json``.

The ``intrinsics-single-cam`` subcommand exists only as a debug helper for
single-camera sanity checks; it does NOT feed the production flow. The
single-stage ``calibrate-all`` requires joint BA — running per-camera OpenCV
intrinsics and plugging them into the BA is strictly worse than letting Multical
handle both at once. Never call ``calibrate_intrinsics`` in production.

Multical 0.4.0 pins ``opencv-contrib-python <=4.7.0``, which conflicts with
the Backbone runtime's OpenCV (4.13+). The isolated venv is the only sane
install path; ``find_multical_binary`` enforces it.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from backbone.shared.geometry import (
    floor_homography_from_K_R_t,
    projection_from_K_R_t,
)
from calibration.multical_io import (
    UNKNOWN_RMS_PX,
    CameraInRig,
    MultiCalSolution,
    parse_rms_from_log,
)
from calibration.multical_io import (
    parse as parse_multical_output,
)
from calibration.schema import (
    CALIBRATION_VERSION,
    CalibrationFile,
    CameraCalibration,
)

REPROJECTION_RMS_HARD_LIMIT_PX = 0.5
"""Per-camera reprojection RMS above which we refuse to write calibration.json."""

DEFAULT_VENV_MULTICAL = Path(__file__).resolve().parent / ".venv-multical"


# ---------------------------------------------------------------------------
# Board specs (ChArUco + AprilGrid) → Multical boards.yaml
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CharucoBoardSpec:
    squares_x: int
    squares_y: int
    square_length_m: float
    marker_length_m: float
    dict_name: str = "DICT_5X5_50"

    def aruco_dict(self) -> cv2.aruco.Dictionary:
        attr = getattr(cv2.aruco, self.dict_name)
        return cv2.aruco.getPredefinedDictionary(attr)

    def board(self) -> cv2.aruco.CharucoBoard:
        return cv2.aruco.CharucoBoard(
            size=(self.squares_x, self.squares_y),
            squareLength=self.square_length_m,
            markerLength=self.marker_length_m,
            dictionary=self.aruco_dict(),
        )

    def multical_yaml_block(self, name: str) -> str:
        """Render this board as a Multical boards.yaml entry (2-space indented)."""
        # Multical builds the name as ``getattr(cv2.aruco, f"DICT_{aruco_dict}")``,
        # so strip our cv2-style ``DICT_`` prefix (else it asks for DICT_DICT_…).
        multical_dict = self.dict_name.removeprefix("DICT_")
        return (
            f"  {name}:\n"
            f"    _type_: charuco\n"
            f"    size: [{self.squares_x}, {self.squares_y}]\n"
            f"    square_length: {self.square_length_m}\n"
            f"    marker_length: {self.marker_length_m}\n"
            f'    aruco_dict: "{multical_dict}"\n'
        )


@dataclass(slots=True)
class AprilGridBoardSpec:
    """One AprilGrid board (Kalibr-compatible) for Multical extrinsic calibration.

    A grid of ``tags_x`` by ``tags_y`` AprilTags. ``tag_spacing`` is the gap-to-tag
    *ratio* (Kalibr convention: physical gap = ``tag_spacing * tag_length``).
    ``start_id`` is the family id of the first tag — **must be unique per board**
    so the 6 boards in a target don't share tag ids (use
    :func:`make_aprilgrid_target`, which offsets them automatically).
    """

    tags_x: int = 6
    tags_y: int = 6
    tag_length_m: float = 0.06
    tag_spacing: float = 0.3
    tag_family: str = "t36h11"
    start_id: int = 0

    def tag_count(self) -> int:
        return self.tags_x * self.tags_y

    def multical_yaml_block(self, name: str) -> str:
        # Keys must match Multical's AprilConfig schema (size, start_id,
        # tag_family, tag_length, tag_spacing) — extra keys are rejected.
        return (
            f"  {name}:\n"
            f"    _type_: aprilgrid\n"
            f"    size: [{self.tags_x}, {self.tags_y}]\n"
            f"    tag_length: {self.tag_length_m}\n"
            f"    tag_spacing: {self.tag_spacing}\n"
            f"    tag_family: {self.tag_family}\n"
            f"    start_id: {self.start_id}\n"
        )


def make_aprilgrid_target(
    n_boards: int = 6,
    template: AprilGridBoardSpec | None = None,
) -> dict[str, AprilGridBoardSpec]:
    """Build a multi-board AprilGrid target: ``n_boards`` copies of ``template``
    with **disjoint** ``start_id`` ranges so no tag id repeats across boards.

    Returns ``{board_name: spec}`` ordered ``april_0 .. april_{n-1}``.
    """
    base = template or AprilGridBoardSpec()
    target: dict[str, AprilGridBoardSpec] = {}
    next_id = base.start_id
    for i in range(n_boards):
        target[f"april_{i}"] = replace(base, start_id=next_id)
        next_id += base.tag_count()
    return target


def write_boards_yaml(
    boards: dict[str, CharucoBoardSpec | AprilGridBoardSpec],
    path: Path,
) -> Path:
    """Write a Multical ``boards.yaml`` from one or more named board specs."""
    body = "boards:\n" + "".join(
        spec.multical_yaml_block(name) for name, spec in boards.items()
    )
    path.write_text(body)
    return path


# ---------------------------------------------------------------------------
# Phase: Multical (primary path)
# ---------------------------------------------------------------------------


def find_multical_binary(
    venv_dir: Path = DEFAULT_VENV_MULTICAL,
    allow_path_fallback: bool = False,
) -> Path:
    """Locate the Multical binary, preferring the isolated venv.

    Args:
        venv_dir: where ``setup_multical.sh`` installed Multical.
        allow_path_fallback: if True and the venv isn't present, fall back to
            whatever ``multical`` is on ``PATH``. Defaults to False because
            picking up a globally-installed Multical risks dragging in its
            stale OpenCV pin.
    """
    candidate = venv_dir / "bin" / "multical"
    if candidate.is_file():
        return candidate
    if allow_path_fallback:
        which = shutil.which("multical")
        if which:
            return Path(which)
    raise RuntimeError(
        f"Multical not found at {candidate}. Run `bash calibration/setup_multical.sh` "
        f"to create the isolated venv. (Set allow_path_fallback=True only for tests.)"
    )


def _write_multical_boards_yaml(board: CharucoBoardSpec, out_dir: Path) -> Path:
    """Write the single-charuco boards.yaml Multical expects (the 1-board path)."""
    return write_boards_yaml({"charuco_main": board}, out_dir / "boards.yaml")


def _stage_image_dirs(image_dirs_by_camera: dict[str, Path], out_dir: Path) -> Path:
    """Symlink per-camera image directories under one root for Multical."""
    image_root = out_dir / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    for cam_id, src in image_dirs_by_camera.items():
        link = image_root / cam_id
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(Path(src).resolve())
    return image_root


def generate_board_images(
    boards: dict[str, CharucoBoardSpec | AprilGridBoardSpec],
    out_dir: Path,
    *,
    paper_size: str = "A4",
    pixels_mm: int = 10,
    margin_mm: int = 20,
    multical_binary: Path | None = None,
) -> Path:
    """Generate printable board images (one PNG per board) via ``multical boards``.

    Writes a ``boards.yaml`` from ``boards`` then invokes Multical's generator at
    the given paper size. Print the PNGs at **100% scale, zero margins** so the
    metric ``square_length`` / ``tag_length`` on paper matches the spec — the
    calibration is only as accurate as the printed board's true dimensions.

    ``margin_mm`` is Multical's border (default 20). Multical requires
    ``board + 2*margin <= paper``, so a board that nearly fills the sheet needs a
    smaller margin (e.g. a tall 1x2 AprilGrid on A4 → ``margin_mm=0``).

    Returns the directory containing the generated images.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    boards_yaml = write_boards_yaml(boards, out_dir / "boards.yaml")
    binary = multical_binary or find_multical_binary()
    cmd = [
        str(binary), "boards",
        "--boards", str(boards_yaml),
        "--paper_size", paper_size,
        "--pixels_mm", str(pixels_mm),
        "--margin_mm", str(margin_mm),
        "--write", str(out_dir),
    ]
    print(f"[gen-boards] invoking: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    print((result.stdout or "") + "\n" + (result.stderr or ""), flush=True)
    print(f"[gen-boards] wrote board images to {out_dir} "
          f"(print at 100% scale, 0 margins, {paper_size})", flush=True)
    return out_dir


def run_multical_calibration(
    image_dirs_by_camera: dict[str, Path],
    board: CharucoBoardSpec,
    work_dir: Path,
    multical_binary: Path | None = None,
    name: str = "calibration",
    vis: bool = False,
) -> MultiCalSolution:
    """Run Multical end-to-end and return the parsed solution with RMS attached.

    Returns:
        :class:`MultiCalSolution` with per-camera intrinsics, rig-frame poses,
        and best-effort RMS extracted from Multical's log output.

    ``vis=True`` adds ``--vis True`` so Multical's 3D viewer opens after the BA
    (interactive + blocking until the window is closed); JSON + log are written
    first, so RMS parsing is unaffected. Never enable in a headless run.

    Raises:
        subprocess.CalledProcessError: if Multical exits non-zero.
        FileNotFoundError: if Multical did not produce the expected output JSON.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    boards_yaml = _write_multical_boards_yaml(board, work_dir)
    image_root = _stage_image_dirs(image_dirs_by_camera, work_dir)

    binary = multical_binary or find_multical_binary()
    cmd = [
        str(binary),
        "calibrate",
        "--image_path", str(image_root),
        "--boards", str(boards_yaml),
        "--output_path", str(work_dir),
        "--name", name,
    ]
    if vis:
        cmd += ["--vis", "True"]
    print(f"[calibrate] invoking: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    log_text = (result.stdout or "") + "\n" + (result.stderr or "")
    print(log_text, flush=True)

    output_json = work_dir / f"{name}.json"
    if not output_json.exists():
        raise FileNotFoundError(
            f"Multical did not produce {output_json}; check the log above."
        )

    sol = parse_multical_output(output_json)
    rms = parse_rms_from_log(log_text, sol.camera_ids)
    return sol.with_rms(rms)


# ---------------------------------------------------------------------------
# Phase: Two-stage (intrinsics ChArUco → extrinsics AprilGrid, fixed K)
# ---------------------------------------------------------------------------
#
# Workflow the 2-camera rig uses (per the operator's plan):
#   1. INTRINSICS  — each camera shoots an A4 ChArUco board from many angles;
#      `multical intrinsic` solves per-camera K, D → intrinsic.json.
#   2. EXTRINSICS  — both cameras view a 6-AprilGrid target spread across the
#      shared volume; `multical calibrate --calibration intrinsic.json
#      --fix_intrinsic` solves the rig poses with K held fixed.
# Stage 1 wants dense corners (ChArUco); stage 2 wants a wide, multi-board target
# both cameras can see at once (AprilGrid) — using each board where it's strong.


def run_multical_intrinsics(
    image_dirs_by_camera: dict[str, Path],
    board: CharucoBoardSpec,
    work_dir: Path,
    multical_binary: Path | None = None,
) -> Path:
    """Stage 1 — per-camera intrinsics from a ChArUco board (``multical intrinsic``).

    Returns the path to the produced ``intrinsic.json`` (consumed by stage 2).
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    boards_yaml = write_boards_yaml({"charuco_main": board},
                                    work_dir / "intrinsic_boards.yaml")
    image_root = _stage_image_dirs(image_dirs_by_camera, work_dir / "intrinsic_images")

    binary = multical_binary or find_multical_binary()
    cmd = [
        str(binary), "intrinsic",
        "--image_path", str(image_root),
        "--boards", str(boards_yaml),
        "--output_path", str(work_dir),
        "--name", "intrinsic",            # → intrinsic.json (also Multical's default)
    ]
    print(f"[calibrate:intrinsics] invoking: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    print((result.stdout or "") + "\n" + (result.stderr or ""), flush=True)

    # Multical writes intrinsic.json to --output_path; older builds drop it next
    # to the images. Accept either.
    for candidate in (work_dir / "intrinsic.json", image_root / "intrinsic.json"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"multical intrinsic did not produce intrinsic.json under {work_dir} "
        f"or {image_root}; check the log above."
    )


def run_multical_extrinsics(
    image_dirs_by_camera: dict[str, Path],
    boards: dict[str, AprilGridBoardSpec],
    work_dir: Path,
    intrinsic_json: Path,
    multical_binary: Path | None = None,
    name: str = "calibration",
    vis: bool = False,
) -> MultiCalSolution:
    """Stage 2 — joint extrinsics from a multi-AprilGrid target with K fixed.

    ``intrinsic_json`` is stage 1's output; ``--fix_intrinsic`` keeps it frozen so
    the BA only solves the rig poses. ``vis=True`` opens Multical's 3D viewer after
    the solve (interactive/blocking). Returns the parsed solution (with RMS).
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    boards_yaml = write_boards_yaml(boards, work_dir / "extrinsic_boards.yaml")
    image_root = _stage_image_dirs(image_dirs_by_camera, work_dir / "extrinsic_images")

    binary = multical_binary or find_multical_binary()
    cmd = [
        str(binary), "calibrate",
        "--image_path", str(image_root),
        "--boards", str(boards_yaml),
        "--calibration", str(intrinsic_json),
        "--fix_intrinsic", "True",        # Multical bool flag: hold stage-1 K fixed
        "--output_path", str(work_dir),
        "--name", name,
    ]
    if vis:
        cmd += ["--vis", "True"]
    print(f"[calibrate:extrinsics] invoking: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    log_text = (result.stdout or "") + "\n" + (result.stderr or "")
    print(log_text, flush=True)

    output_json = work_dir / f"{name}.json"
    if not output_json.exists():
        raise FileNotFoundError(
            f"Multical did not produce {output_json}; check the log above."
        )
    sol = parse_multical_output(output_json)
    rms = parse_rms_from_log(log_text, sol.camera_ids)
    return sol.with_rms(rms)


def run_multical_vis(workspace: Path, multical_binary: Path | None = None) -> None:
    """Open Multical's built-in 3D viewer on a saved workspace (``multical vis``).

    ``workspace`` is either the ``work_dir`` of a prior calibrate run (Multical
    resolves ``<dir>/calibration.pkl``) or an explicit ``*.pkl``. The viewer is
    **interactive and blocking** — it returns when the operator closes the window
    — so we inherit the terminal instead of capturing output. Needs the viewer
    deps (qtpy/pyvista/pyvistaqt/PyQt5) + a display (WSLg/X); the workspace's
    images must still be present on disk.
    """
    binary = multical_binary or find_multical_binary()
    cmd = [str(binary), "vis", "--workspace_file", str(workspace)]
    print(f"[calibrate:vis] invoking: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Phase: Floor anchor — establish the world frame
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FloorAnchor:
    """Definition of the world frame's Z=0 plane and origin.

    The world frame is defined to lie on the floor: Z=0 IS the floor, X/Y
    span the floor plane. The anchor specifies where the world origin sits.
    """

    method: str           # "charuco_floor" | "planefit" | "tape_measured_points"
    note: str
    R_world_from_rig: np.ndarray  # 3x3 rotation, rig → world
    t_world_from_rig: np.ndarray  # 3-vector translation, rig → world


def _detect_charuco_in_image(
    image_path: Path,
    board: CharucoBoardSpec,
    *,
    clahe: bool = True,
    clahe_clip: float = 2.0,
    clahe_grid: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect ChArUco corners in one image. Returns (corners_uv, object_xy).

    A CLAHE contrast pass (default ON) is applied before detection to match the
    isical capture gate, so a flat, distant, low-contrast floor board detects as
    reliably at solve time as it did when captured. CLAHE only remaps intensities
    (no geometric warp), so the recovered corner positions stay sub-pixel-faithful.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"could not read image: {image_path}")
    if clahe:
        g = max(1, int(clahe_grid))
        img = cv2.createCLAHE(clipLimit=float(clahe_clip),
                              tileGridSize=(g, g)).apply(img)
    detector = cv2.aruco.CharucoDetector(board.board())
    ch_corners, ch_ids, _, _ = detector.detectBoard(img)
    if ch_corners is None or ch_ids is None or len(ch_ids) < 4:
        raise RuntimeError(
            f"floor anchor: too few ChArUco corners in {image_path} "
            f"(found {0 if ch_ids is None else len(ch_ids)}, need ≥4)"
        )
    object_points = board.board().getChessboardCorners()[ch_ids.flatten()]
    return ch_corners.reshape(-1, 2), object_points.reshape(-1, 3)


def estimate_floor_anchor_charuco(
    floor_shot_by_camera: dict[str, Path],
    solution: MultiCalSolution,
    board: CharucoBoardSpec,
) -> FloorAnchor:
    """Recover the rig → world transform from a ChArUco board on the floor.

    Each camera takes one shot of a ChArUco board placed flat on the floor.
    ``cv2.solvePnP`` recovers the board's pose in that camera's frame; combined
    with the rig-frame pose from Multical, every camera independently estimates
    the board's pose in rig coordinates. We average them (a few cameras' worth
    of votes is enough) and invert to get rig → world.

    The world frame ends up with its origin at the board's anchor corner,
    Z=0 the floor plane, X/Y aligned with the board's axes.
    """
    if not floor_shot_by_camera:
        raise RuntimeError("floor anchor needs at least one floor shot")

    board_in_rig_R: list[np.ndarray] = []
    board_in_rig_t: list[np.ndarray] = []

    for cam_id, shot_path in floor_shot_by_camera.items():
        if cam_id not in solution.cameras:
            raise RuntimeError(
                f"floor shot references unknown camera {cam_id!r}; "
                f"Multical solution has {sorted(solution.cameras)}"
            )
        cam = solution.cameras[cam_id]
        corners_uv, object_xyz = _detect_charuco_in_image(shot_path, board)
        ok, rvec, tvec = cv2.solvePnP(object_xyz, corners_uv, cam.K, cam.D)
        if not ok:
            raise RuntimeError(f"cv2.solvePnP failed for floor shot {shot_path}")
        R_cam_board, _ = cv2.Rodrigues(rvec)
        t_cam_board = tvec.reshape(3)

        # Board pose in camera frame is (R_cam_board, t_cam_board).
        # Camera pose in rig frame is (cam.R_in_rig, cam.t_in_rig)  [world←camera convention].
        # Board pose in rig frame:
        #   R_rig_board = R_rig_cam @ R_cam_board
        #   t_rig_board = R_rig_cam @ t_cam_board + t_rig_cam
        R_rig_board = cam.R_in_rig @ R_cam_board
        t_rig_board = cam.R_in_rig @ t_cam_board + cam.t_in_rig
        board_in_rig_R.append(R_rig_board)
        board_in_rig_t.append(t_rig_board)

    R_avg = _average_rotation(board_in_rig_R)
    t_avg = np.mean(np.stack(board_in_rig_t), axis=0)

    # world ← rig is the inverse of board-in-rig (the world frame IS the board).
    R_world_rig = R_avg.T
    t_world_rig = -R_world_rig @ t_avg

    return FloorAnchor(
        method="charuco_floor",
        note=f"ChArUco floor board across {len(floor_shot_by_camera)} cameras",
        R_world_from_rig=R_world_rig,
        t_world_from_rig=t_world_rig,
    )


def _average_rotation(rotations: list[np.ndarray]) -> np.ndarray:
    """Naïve rotation averaging via SVD of stacked matrices.

    Good enough for the floor anchor: the cameras' independent estimates of the
    board pose should disagree only by sub-degree noise. For larger spreads we
    would switch to quaternion averaging, but at the noise level expected from
    a well-calibrated ChArUco shot this is fine.
    """
    stacked = np.mean(np.stack(rotations), axis=0)
    U, _S, Vt = np.linalg.svd(stacked)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    return R


def compose_camera_in_world(
    cam: CameraInRig,
    anchor: FloorAnchor,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the floor anchor to one camera's rig-frame pose."""
    R_world_cam = anchor.R_world_from_rig @ cam.R_in_rig
    t_world_cam = anchor.R_world_from_rig @ cam.t_in_rig + anchor.t_world_from_rig
    return R_world_cam, t_world_cam


# ---------------------------------------------------------------------------
# Phase: Assemble calibration.json
# ---------------------------------------------------------------------------


def assemble_calibration(
    solution: MultiCalSolution,
    anchor: FloorAnchor,
    *,
    allow_unknown_rms: bool = False,
) -> CalibrationFile:
    """Combine a parsed Multical solution + a floor anchor into the on-disk schema.

    Refuses to assemble if any camera's reprojection RMS exceeds the hard limit.
    If a camera's RMS is unknown (Multical's log was unparseable), the call
    fails unless ``allow_unknown_rms=True`` — the rest of the system trusts
    ``calibration.json`` blindly, and "we don't know how accurate this is"
    is not the same as "this is accurate enough".
    """
    cameras: dict[str, CameraCalibration] = {}
    for cam_id, cam in solution.cameras.items():
        rms = cam.rms_px
        if rms == UNKNOWN_RMS_PX:
            if not allow_unknown_rms:
                raise RuntimeError(
                    f"{cam_id}: per-camera RMS not parsed from Multical log. "
                    f"Pass allow_unknown_rms=True if you accept an un-gated calibration."
                )
        elif rms > REPROJECTION_RMS_HARD_LIMIT_PX:
            raise RuntimeError(
                f"{cam_id}: reprojection RMS {rms:.3f} px exceeds "
                f"hard limit {REPROJECTION_RMS_HARD_LIMIT_PX} px — refusing to write calibration.json"
            )

        R_world, t_world = compose_camera_in_world(cam, anchor)
        P = projection_from_K_R_t(cam.K, R_world, t_world)
        H = floor_homography_from_K_R_t(cam.K, R_world, t_world)

        cameras[cam_id] = CameraCalibration(
            camera_id=cam_id,
            image_size_wh=cam.image_size_wh,
            K=cam.K.tolist(),
            D=cam.D.tolist(),
            R=R_world.tolist(),
            t=t_world.reshape(3).tolist(),
            H=H.tolist(),
            P=P.tolist(),
            reprojection_rms_px=float(rms),
        )

    return CalibrationFile(
        version=CALIBRATION_VERSION,
        created_at=_dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        floor_anchor_method=anchor.method,
        floor_origin_note=anchor.note,
        cameras=cameras,
    )


# ---------------------------------------------------------------------------
# Phase: Single-camera ChArUco intrinsics (DEBUG ONLY)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class IntrinsicsResult:
    K: np.ndarray
    D: np.ndarray
    image_size_wh: tuple[int, int]
    reprojection_rms_px: float


def calibrate_intrinsics(
    images_dir: Path,
    board: CharucoBoardSpec,
) -> IntrinsicsResult:
    """**DEBUG ONLY.** Per-camera ChArUco intrinsic calibration.

    Not part of the production calibration flow — :func:`run_multical_calibration`
    handles intrinsics jointly across all cameras for better accuracy. This
    helper exists so a single camera can be sanity-checked in isolation when
    diagnosing a calibration issue.
    """
    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        raise FileNotFoundError(f"no images found in {images_dir}")

    charuco_board = board.board()
    detector = cv2.aruco.CharucoDetector(charuco_board)

    all_corners: list[np.ndarray] = []
    all_ids: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None

    for img_path in images:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape[:2]
        if image_size is None:
            image_size = (w, h)
        elif image_size != (w, h):
            raise RuntimeError(
                f"image size mismatch in {images_dir}: {image_size} vs {(w, h)} ({img_path.name})"
            )

        ch_corners, ch_ids, _, _ = detector.detectBoard(img)
        if ch_corners is not None and ch_ids is not None and len(ch_ids) >= 6:
            all_corners.append(ch_corners)
            all_ids.append(ch_ids)

    if len(all_corners) < 8:
        raise RuntimeError(
            f"only {len(all_corners)} usable views in {images_dir}; need >=8 for stable intrinsics"
        )
    assert image_size is not None

    rms, K, D, _rvecs, _tvecs = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=all_corners,
        charucoIds=all_ids,
        board=charuco_board,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None,
    )
    return IntrinsicsResult(
        K=np.asarray(K, dtype=np.float64),
        D=np.asarray(D, dtype=np.float64).reshape(-1),
        image_size_wh=image_size,
        reprojection_rms_px=float(rms),
    )


# ---------------------------------------------------------------------------
# High-level orchestration — the primary entry point
# ---------------------------------------------------------------------------


def calibrate_all(
    image_dirs_by_camera: dict[str, Path],
    floor_shot_by_camera: dict[str, Path],
    board: CharucoBoardSpec,
    work_dir: Path,
    output_path: Path,
    *,
    multical_binary: Path | None = None,
    allow_unknown_rms: bool = False,
    vis: bool = False,
) -> CalibrationFile:
    """End-to-end calibration: Multical → floor anchor → calibration.json.

    ``vis=True`` opens Multical's 3D viewer right after the bundle adjustment.
    """
    solution = run_multical_calibration(
        image_dirs_by_camera=image_dirs_by_camera,
        board=board,
        work_dir=work_dir,
        multical_binary=multical_binary,
        vis=vis,
    )
    anchor = estimate_floor_anchor_charuco(floor_shot_by_camera, solution, board)
    calibration = assemble_calibration(solution, anchor, allow_unknown_rms=allow_unknown_rms)
    calibration.write(output_path)
    print(f"[calibrate] wrote {output_path}", flush=True)
    for cam_id, cam in calibration.cameras.items():
        print(f"[calibrate]   {cam_id}: RMS={cam.reprojection_rms_px:.3f} px", flush=True)
    return calibration


def calibrate_two_stage(
    intrinsic_dirs_by_camera: dict[str, Path],
    extrinsic_dirs_by_camera: dict[str, Path],
    floor_shot_by_camera: dict[str, Path],
    charuco_board: CharucoBoardSpec,
    aprilgrid_target: dict[str, AprilGridBoardSpec],
    work_dir: Path,
    output_path: Path,
    *,
    multical_binary: Path | None = None,
    allow_unknown_rms: bool = False,
    vis: bool = False,
) -> CalibrationFile:
    """Two-stage 2-camera calibration → calibration.json.

    Stage 1 solves per-camera intrinsics from ChArUco shots; stage 2 solves the
    rig extrinsics from a multi-AprilGrid target with those intrinsics fixed; then
    the ChArUco floor shot anchors the rig to the world frame and we assemble the
    same ``calibration.json`` the single-stage path writes (K, D, R, t, H, P).

    ``vis=True`` opens Multical's 3D viewer after the extrinsic solve (the run that
    holds all cameras + boards in one frame).
    """
    intrinsic_json = run_multical_intrinsics(
        intrinsic_dirs_by_camera, charuco_board, work_dir,
        multical_binary=multical_binary,
    )
    solution = run_multical_extrinsics(
        extrinsic_dirs_by_camera, aprilgrid_target, work_dir, intrinsic_json,
        multical_binary=multical_binary, vis=vis,
    )
    anchor = estimate_floor_anchor_charuco(floor_shot_by_camera, solution, charuco_board)
    calibration = assemble_calibration(solution, anchor, allow_unknown_rms=allow_unknown_rms)
    calibration.write(output_path)
    print(f"[calibrate] wrote {output_path}", flush=True)
    for cam_id, cam in calibration.cameras.items():
        print(f"[calibrate]   {cam_id}: RMS={cam.reprojection_rms_px:.3f} px", flush=True)
    return calibration


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_board_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--squares-x", type=int, default=8)
    p.add_argument("--squares-y", type=int, default=6)
    p.add_argument("--square-length", type=float, default=0.04)
    p.add_argument("--marker-length", type=float, default=0.03)
    p.add_argument("--dict", default="DICT_5X5_50")


def _board_from_args(args: argparse.Namespace) -> CharucoBoardSpec:
    return CharucoBoardSpec(
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_length_m=args.square_length,
        marker_length_m=args.marker_length,
        dict_name=args.dict,
    )


def _add_aprilgrid_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--n-boards", type=int, default=6,
                   help="number of AprilGrid boards in the extrinsic target")
    p.add_argument("--april-tags-x", type=int, default=6)
    p.add_argument("--april-tags-y", type=int, default=6)
    p.add_argument("--tag-length", type=float, default=0.06,
                   help="AprilTag side length in metres (printed size)")
    p.add_argument("--tag-spacing", type=float, default=0.3,
                   help="gap/tag ratio (Kalibr: gap = tag_spacing * tag_length)")
    p.add_argument("--tag-family", default="t36h11")


def _aprilgrid_target_from_args(args: argparse.Namespace) -> dict[str, AprilGridBoardSpec]:
    return make_aprilgrid_target(
        n_boards=args.n_boards,
        template=AprilGridBoardSpec(
            tags_x=args.april_tags_x,
            tags_y=args.april_tags_y,
            tag_length_m=args.tag_length,
            tag_spacing=args.tag_spacing,
            tag_family=args.tag_family,
        ),
    )


def _parse_camera_dir_map(spec: list[str]) -> dict[str, Path]:
    """Parse repeated --camera-dir cam_id=path arguments into a dict."""
    out: dict[str, Path] = {}
    for entry in spec:
        if "=" not in entry:
            raise argparse.ArgumentTypeError(f"expected 'cam_id=path', got {entry!r}")
        cam_id, _, path = entry.partition("=")
        out[cam_id] = Path(path)
    return out


def _add_multical_binary_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--multical-binary", default=None,
        help=f"override Multical binary (default: {DEFAULT_VENV_MULTICAL}/bin/multical)",
    )


def _add_solve_common_args(p: argparse.ArgumentParser) -> None:
    """Args shared by the calibrate-all / calibrate-2cam solvers."""
    p.add_argument("--work-dir", required=True, help="scratch dir for Multical output")
    p.add_argument("--output", required=True, help="output calibration.json path")
    _add_multical_binary_arg(p)
    p.add_argument(
        "--allow-unknown-rms", action="store_true",
        help="proceed even if RMS could not be parsed from Multical's log (DANGER)",
    )
    p.add_argument(
        "--vis", action="store_true",
        help="open Multical's 3D viewer after the solve (interactive; needs a display)",
    )


def _binary_from_args(args: argparse.Namespace) -> Path | None:
    """The Multical binary override (or None → the isolated venv default)."""
    return Path(args.multical_binary) if args.multical_binary else None


def _cmd_calibrate_all(args: argparse.Namespace) -> int:
    board = _board_from_args(args)
    image_dirs = _parse_camera_dir_map(args.camera_dir)
    floor_shots = _parse_camera_dir_map(args.floor_shot)
    binary = _binary_from_args(args)
    calibrate_all(
        image_dirs_by_camera=image_dirs,
        floor_shot_by_camera=floor_shots,
        board=board,
        work_dir=Path(args.work_dir),
        output_path=Path(args.output),
        multical_binary=binary,
        allow_unknown_rms=args.allow_unknown_rms,
        vis=args.vis,
    )
    return 0


def _cmd_gen_boards(args: argparse.Namespace) -> int:
    charuco = _board_from_args(args)
    target: dict[str, CharucoBoardSpec | AprilGridBoardSpec] = {"charuco_main": charuco}
    target.update(_aprilgrid_target_from_args(args))
    binary = _binary_from_args(args)
    generate_board_images(
        target, Path(args.output_dir),
        paper_size=args.paper_size, pixels_mm=args.pixels_mm, margin_mm=args.margin_mm,
        multical_binary=binary,
    )
    return 0


def _cmd_calibrate_2cam(args: argparse.Namespace) -> int:
    charuco = _board_from_args(args)
    target = _aprilgrid_target_from_args(args)
    binary = _binary_from_args(args)
    calibrate_two_stage(
        intrinsic_dirs_by_camera=_parse_camera_dir_map(args.intrinsic_dir),
        extrinsic_dirs_by_camera=_parse_camera_dir_map(args.extrinsic_dir),
        floor_shot_by_camera=_parse_camera_dir_map(args.floor_shot),
        charuco_board=charuco,
        aprilgrid_target=target,
        work_dir=Path(args.work_dir),
        output_path=Path(args.output),
        multical_binary=binary,
        allow_unknown_rms=args.allow_unknown_rms,
        vis=args.vis,
    )
    return 0


def _cmd_vis(args: argparse.Namespace) -> int:
    binary = _binary_from_args(args)
    run_multical_vis(Path(args.workspace), multical_binary=binary)
    return 0


def _cmd_intrinsics_single_cam(args: argparse.Namespace) -> int:
    board = _board_from_args(args)
    res = calibrate_intrinsics(Path(args.images_dir), board)
    out = {
        "camera_id": args.camera_id,
        "image_size_wh": list(res.image_size_wh),
        "K": res.K.tolist(),
        "D": res.D.tolist(),
        "reprojection_rms_px": res.reprojection_rms_px,
    }
    Path(args.output).write_text(json.dumps(out, indent=2))
    rms_marker = "OK" if res.reprojection_rms_px <= REPROJECTION_RMS_HARD_LIMIT_PX else "FAIL"
    print(
        f"[intrinsics-single-cam:{args.camera_id}] RMS={res.reprojection_rms_px:.3f} px "
        f"({rms_marker}); wrote {args.output}",
        flush=True,
    )
    return 0 if rms_marker == "OK" else 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="calibrate", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)

    pa = sub.add_parser(
        "calibrate-all",
        help="primary path: Multical joint BA + floor anchor + write calibration.json",
    )
    pa.add_argument(
        "--camera-dir",
        action="append",
        required=True,
        help="cam_id=path/to/charuco/shots. Repeat per camera.",
    )
    pa.add_argument(
        "--floor-shot",
        action="append",
        required=True,
        help="cam_id=path/to/floor_anchor.jpg. Repeat per camera.",
    )
    _add_solve_common_args(pa)
    _add_board_args(pa)
    pa.set_defaults(func=_cmd_calibrate_all)

    # gen-boards — generate printable A4 boards (ChArUco intrinsics + N AprilGrids).
    pg = sub.add_parser(
        "gen-boards",
        help="generate printable A4 board PNGs (ChArUco intrinsics + AprilGrid target) via Multical",
    )
    pg.add_argument("--output-dir", required=True, help="where to write board PNGs + boards.yaml")
    pg.add_argument("--paper-size", default="A4")
    pg.add_argument("--pixels-mm", type=int, default=10, help="render resolution (px per mm)")
    pg.add_argument("--margin-mm", type=int, default=20,
                    help="border in mm; lower it for boards that nearly fill the sheet (0 = none)")
    _add_multical_binary_arg(pg)
    _add_board_args(pg)
    _add_aprilgrid_args(pg)
    pg.set_defaults(func=_cmd_gen_boards)

    # calibrate-2cam — two-stage: ChArUco intrinsics → AprilGrid extrinsics (fixed K).
    p2 = sub.add_parser(
        "calibrate-2cam",
        help="2-camera two-stage: ChArUco intrinsics + multi-AprilGrid extrinsics → calibration.json",
    )
    p2.add_argument("--intrinsic-dir", action="append", required=True,
                    help="cam_id=path/to/charuco/intrinsic/shots. Repeat per camera.")
    p2.add_argument("--extrinsic-dir", action="append", required=True,
                    help="cam_id=path/to/aprilgrid/extrinsic/shots. Repeat per camera.")
    p2.add_argument("--floor-shot", action="append", required=True,
                    help="cam_id=path/to/floor_anchor.jpg (ChArUco on the floor). Repeat per camera.")
    _add_solve_common_args(p2)
    _add_board_args(p2)
    _add_aprilgrid_args(p2)
    p2.set_defaults(func=_cmd_calibrate_2cam)

    # vis — re-open Multical's 3D viewer on a saved workspace (no recalibration).
    pv = sub.add_parser(
        "vis",
        help="open Multical's 3D viewer on a saved workspace (work-dir or calibration.pkl)",
    )
    pv.add_argument("--workspace", required=True,
                    help="a prior --work-dir (resolves calibration.pkl) or an explicit *.pkl")
    _add_multical_binary_arg(pv)
    pv.set_defaults(func=_cmd_vis)

    pi = sub.add_parser(
        "intrinsics-single-cam",
        help="DEBUG ONLY: per-camera ChArUco intrinsics. Not for production.",
    )
    pi.add_argument("--camera-id", required=True)
    pi.add_argument("--images-dir", required=True)
    pi.add_argument("--output", required=True)
    _add_board_args(pi)
    pi.set_defaults(func=_cmd_intrinsics_single_cam)

    # Mode 1 — single-camera 4-point floor-plane fit (no Multical).
    from .calibrate_single_cam import add_single_cam_subparser
    add_single_cam_subparser(sub)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
