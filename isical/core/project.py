"""Calibration project — `data/<name>/calib.yaml`, validated by pydantic.

A *project* is one calibration session (a named camera rig): up to two cameras
(cam_a, cam_b — RTSP or USB), the board geometry (defaults to the printed
tools/boards_print boards), and per-phase capture thresholds. Mirrors isiGen's
ProjectConfig/create_project. Data dir layout:

    data/<name>/calib.yaml
    data/<name>/intrinsic/{cam_a,cam_b}/*.jpg
    data/<name>/extrinsic/{cam_a,cam_b}/*.jpg
    data/<name>/floor/{cam_a,cam_b}.jpg
    data/<name>/work/                 # multical workspace (intrinsic.json, logs)
    data/<name>/calibration.json      # Export output (schema.CalibrationFile)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

# Floor for the operator-settable extrinsic pair target — fewer than this and the
# rig BA is too poorly conditioned to trust.
EXTRINSIC_TARGET_MIN = 4

CALIB_YAML = "calib.yaml"
_ISICAL_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = _ISICAL_ROOT / "configs" / "calib_template.yaml"

# The two fixed camera slots (Mode-2 rig). cam_b empty ⇒ single-camera capture only.
CAMERA_IDS = ("cam_a", "cam_b")


class CameraSpec(BaseModel):
    """One camera source. ``type`` rtsp → ``url``; usb → ``device`` (/dev/videoN)."""

    model_config = ConfigDict(extra="allow")
    id: str
    type: Literal["rtsp", "usb"] = "rtsp"
    url: str = ""              # rtsp://user:pass@ip:554/...
    device: str = ""          # /dev/video0 (usb)

    def source_ref(self) -> str:
        return self.url if self.type == "rtsp" else self.device

    def configured(self) -> bool:
        return bool(self.source_ref().strip())


class BoardSpec(BaseModel):
    """Board geometry (defaults match tools/boards_print/boards.yaml)."""

    model_config = ConfigDict(extra="allow")
    # ChArUco (intrinsics + floor anchor)
    squares_x: int = 5
    squares_y: int = 7
    square_length_m: float = 0.035
    marker_length_m: float = 0.026
    dict_name: str = "DICT_5X5_50"
    # AprilGrid target (extrinsics) — 6 boards, disjoint tag ids
    n_aprilgrids: int = 6
    april_tags_x: int = 1
    april_tags_y: int = 2
    tag_length_m: float = 0.18
    tag_spacing: float = 0.3
    tag_family: str = "t36h11"


# --- AprilGrid board-measurement conversion (operator measures in cm) ---
# The operator measures the printed tag with a ruler in centimetres; isical derives
# the Multical/Kalibr board config. tag_spacing is the Kalibr *ratio*: the physical
# inter-tag gap equals tag_spacing * tag_length, so spacing = gap_cm / tag_length_cm.
#
#   18 cm tag, 5.4 cm gap  →  tag_length_m 0.18, tag_spacing 0.30
#
# Sane ruler ranges (reject/clamp out-of-range input at the API boundary).
TAG_LENGTH_CM_MIN, TAG_LENGTH_CM_MAX = 1.0, 100.0
TAG_GAP_CM_MIN, TAG_GAP_CM_MAX = 0.0, 50.0


def board_config_from_cm(tag_length_cm: float, tag_gap_cm: float) -> dict[str, float]:
    """cm measurements → persisted board config (tag_length_m, tag_spacing).

    Raises ``ValueError`` for non-positive / out-of-range input."""
    tl = float(tag_length_cm)
    gap = float(tag_gap_cm)
    if not (TAG_LENGTH_CM_MIN <= tl <= TAG_LENGTH_CM_MAX):
        raise ValueError(f"tag_length_cm must be in [{TAG_LENGTH_CM_MIN}, {TAG_LENGTH_CM_MAX}]")
    if not (TAG_GAP_CM_MIN <= gap <= TAG_GAP_CM_MAX):
        raise ValueError(f"tag_gap_cm must be in [{TAG_GAP_CM_MIN}, {TAG_GAP_CM_MAX}]")
    if tl <= 0:
        raise ValueError("tag_length_cm must be > 0")
    return {"tag_length_m": tl / 100.0, "tag_spacing": gap / tl}


def board_cm_from_config(tag_length_m: float, tag_spacing: float) -> dict[str, float]:
    """Reverse: persisted board config → cm inputs for the operator form.

    tag_length_cm = tag_length_m * 100 ; tag_gap_cm = tag_spacing * tag_length_m * 100."""
    tlm = float(tag_length_m)
    return {"tag_length_cm": tlm * 100.0,
            "tag_gap_cm": float(tag_spacing) * tlm * 100.0}


class CaptureSpec(BaseModel):
    """Auto-snap thresholds for the live capture loop."""

    model_config = ConfigDict(extra="allow")
    target_per_camera: int = 25       # intrinsic shots wanted per camera
    extrinsic_target: int = 10        # synchronized extrinsic pairs wanted (operator-settable)
    min_charuco_corners: int = 12     # auto-snap only with ≥ this many ChArUco corners
    min_april_tags: int = 4           # auto-snap only with ≥ this many AprilTags (per cam)
    blur_min_var: float = 80.0        # Laplacian variance floor (reject blur)
    steady_max_motion: float = 2.5    # max mean board-corner motion (px) to count "steady"
    novelty_min_dist: float = 0.06    # min normalized board-centroid move vs kept shots
    # --- detection-boost for small/far AprilTags (favours detection over speed) ---
    tag_clahe: bool = True            # grayscale + CLAHE before AprilTag detection
    tag_clahe_clip: float = 2.0       # CLAHE clip limit
    tag_clahe_grid: int = 8           # CLAHE tile grid (NxN)
    tag_quad_decimate: float = 1.0    # AprilTag quad_decimate (1.0 = no downscale → small tags)

    @field_validator("extrinsic_target")
    @classmethod
    def _floor_extrinsic_target(cls, v: int) -> int:
        return max(EXTRINSIC_TARGET_MIN, int(v))


class CalibConfig(BaseModel):
    """The whole calib.yaml."""

    model_config = ConfigDict(extra="allow")
    name: str
    cameras: dict[str, CameraSpec]
    board: BoardSpec = Field(default_factory=BoardSpec)
    capture: CaptureSpec = Field(default_factory=CaptureSpec)
    # Extrinsic method: "aprilgrid" (default/fallback, target-based) or
    # "targetless" (SuperPoint+LightGlue, experimental).
    extrinsic_method: Literal["aprilgrid", "targetless"] = "aprilgrid"

    @field_validator("cameras")
    @classmethod
    def _known_cameras(cls, v: dict[str, CameraSpec]) -> dict[str, CameraSpec]:
        for cid in v:
            if cid not in CAMERA_IDS:
                raise ValueError(f"unknown camera id {cid!r} (allowed: {CAMERA_IDS})")
        return v

    # ---- helpers ----
    def configured_cameras(self) -> list[str]:
        return [cid for cid in CAMERA_IDS
                if cid in self.cameras and self.cameras[cid].configured()]

    def is_mode2(self) -> bool:
        return len(self.configured_cameras()) >= 2


def load_project(project_dir: Path) -> CalibConfig:
    data = yaml.safe_load((Path(project_dir) / CALIB_YAML).read_text()) or {}
    return CalibConfig.model_validate(data)


def save_project(project_dir: Path, cfg: CalibConfig) -> None:
    project_dir = Path(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / CALIB_YAML).write_text(
        yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False, allow_unicode=True))


def create_project(data_dir: Path, name: str, cameras: dict[str, CameraSpec]) -> Path:
    """New calibration project from the template + the given cameras."""
    project_dir = Path(data_dir) / name
    if (project_dir / CALIB_YAML).exists():
        raise FileExistsError(f"calibration {name!r} already exists at {project_dir}")
    template = yaml.safe_load(TEMPLATE_PATH.read_text()) if TEMPLATE_PATH.exists() else {}
    cfg = CalibConfig.model_validate({
        **(template or {}), "name": name,
        "cameras": {cid: c.model_dump() for cid, c in cameras.items()},
    })
    save_project(project_dir, cfg)
    for cid in CAMERA_IDS:
        (project_dir / "intrinsic" / cid).mkdir(parents=True, exist_ok=True)
        (project_dir / "extrinsic" / cid).mkdir(parents=True, exist_ok=True)
    (project_dir / "floor").mkdir(parents=True, exist_ok=True)
    (project_dir / "work").mkdir(parents=True, exist_ok=True)
    return project_dir


def list_projects(data_dir: Path) -> list[str]:
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return []
    return sorted(p.parent.name for p in data_dir.glob(f"*/{CALIB_YAML}"))


def delete_project(data_dir: Path, name: str, runs_dir: Path | None = None) -> None:
    import shutil
    data_dir = Path(data_dir).resolve()
    project_dir = (data_dir / name).resolve()
    if project_dir.parent != data_dir or not (project_dir / CALIB_YAML).is_file():
        raise FileNotFoundError(f"calibration {name!r} not found under {data_dir}")
    shutil.rmtree(project_dir)
    if runs_dir is not None:
        for f in (Path(runs_dir) / "jobs").glob(f"*_{name}_*.log"):
            f.unlink(missing_ok=True)


# ---- board-spec adapters (CalibConfig.board → calibration.calibrate dataclasses) ----

def charuco_spec(board: BoardSpec):
    """Build a calibration.calibrate.CharucoBoardSpec from the project's board."""
    from calibration.calibrate import CharucoBoardSpec
    return CharucoBoardSpec(
        squares_x=board.squares_x, squares_y=board.squares_y,
        square_length_m=board.square_length_m, marker_length_m=board.marker_length_m,
        dict_name=board.dict_name,
    )


def aprilgrid_target(board: BoardSpec):
    """Build the 6-board AprilGrid target dict from the project's board."""
    from calibration.calibrate import AprilGridBoardSpec, make_aprilgrid_target
    template = AprilGridBoardSpec(
        tags_x=board.april_tags_x, tags_y=board.april_tags_y,
        tag_length_m=board.tag_length_m, tag_spacing=board.tag_spacing,
        tag_family=board.tag_family, start_id=0,
    )
    return make_aprilgrid_target(n_boards=board.n_aprilgrids, template=template)
