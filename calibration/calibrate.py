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

The ``intrinsics-single-cam`` subcommand exists only as a debug helper for
single-camera sanity checks; it does NOT feed the production flow. The
architecture requires joint BA — running per-camera OpenCV intrinsics and
plugging them into the BA is strictly worse than letting Multical handle both
at once. Never call ``calibrate_intrinsics`` in production.

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
from dataclasses import dataclass
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
# ChArUco board specification
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
    """Write the boards.yaml Multical expects."""
    path = out_dir / "boards.yaml"
    path.write_text(
        f"""boards:
  charuco_main:
    type: charuco
    size: [{board.squares_x}, {board.squares_y}]
    square_length: {board.square_length_m}
    marker_length: {board.marker_length_m}
    aruco_dict: "{board.dict_name}"
"""
    )
    return path


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


def run_multical_calibration(
    image_dirs_by_camera: dict[str, Path],
    board: CharucoBoardSpec,
    work_dir: Path,
    multical_binary: Path | None = None,
    name: str = "calibration",
) -> MultiCalSolution:
    """Run Multical end-to-end and return the parsed solution with RMS attached.

    Returns:
        :class:`MultiCalSolution` with per-camera intrinsics, rig-frame poses,
        and best-effort RMS extracted from Multical's log output.

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
# Phase: Floor anchor — establish the world frame
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FloorAnchor:
    """Definition of the world frame's Z=0 plane and origin.

    The world frame is defined to lie on the floor: Z=0 IS the floor, X/Y
    span the floor plane. The anchor specifies where the world origin sits.
    """

    method: str           # "charuco_floor" | "tape_measured_points"
    note: str
    R_world_from_rig: np.ndarray  # 3x3 rotation, rig → world
    t_world_from_rig: np.ndarray  # 3-vector translation, rig → world


def _detect_charuco_in_image(
    image_path: Path,
    board: CharucoBoardSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect ChArUco corners in one image. Returns (corners_uv, object_xy)."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"could not read image: {image_path}")
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
) -> CalibrationFile:
    """End-to-end calibration: Multical → floor anchor → calibration.json."""
    solution = run_multical_calibration(
        image_dirs_by_camera=image_dirs_by_camera,
        board=board,
        work_dir=work_dir,
        multical_binary=multical_binary,
    )
    anchor = estimate_floor_anchor_charuco(floor_shot_by_camera, solution, board)
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


def _parse_camera_dir_map(spec: list[str]) -> dict[str, Path]:
    """Parse repeated --camera-dir cam_id=path arguments into a dict."""
    out: dict[str, Path] = {}
    for entry in spec:
        if "=" not in entry:
            raise argparse.ArgumentTypeError(f"expected 'cam_id=path', got {entry!r}")
        cam_id, _, path = entry.partition("=")
        out[cam_id] = Path(path)
    return out


def _cmd_calibrate_all(args: argparse.Namespace) -> int:
    board = _board_from_args(args)
    image_dirs = _parse_camera_dir_map(args.camera_dir)
    floor_shots = _parse_camera_dir_map(args.floor_shot)
    binary = Path(args.multical_binary) if args.multical_binary else None
    calibrate_all(
        image_dirs_by_camera=image_dirs,
        floor_shot_by_camera=floor_shots,
        board=board,
        work_dir=Path(args.work_dir),
        output_path=Path(args.output),
        multical_binary=binary,
        allow_unknown_rms=args.allow_unknown_rms,
    )
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
    pa.add_argument("--work-dir", required=True, help="scratch dir for Multical output")
    pa.add_argument("--output", required=True, help="output calibration.json path")
    pa.add_argument(
        "--multical-binary",
        default=None,
        help=f"override Multical binary path (default: {DEFAULT_VENV_MULTICAL}/bin/multical)",
    )
    pa.add_argument(
        "--allow-unknown-rms",
        action="store_true",
        help="proceed even if RMS could not be parsed from Multical's log (DANGER)",
    )
    _add_board_args(pa)
    pa.set_defaults(func=_cmd_calibrate_all)

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
