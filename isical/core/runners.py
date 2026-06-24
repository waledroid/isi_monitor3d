"""Phase runners — the solves the JobRunner submits (one impl per phase).

Drives the existing Multical backend (calibration/calibrate.py) on the images the
capture session collected. Mirrors isiGen's core/runners pattern (resumable, logs,
progress.report). Splits calibrate_two_stage across the 3 phase cards:

  * run_intrinsic — multical intrinsic per camera → work/intrinsic.json + per-cam RMS
  * run_extrinsic — multical calibrate --fix_intrinsic + floor anchor + assemble
                    → data/<name>/calibration.json  (needs intrinsic.json from phase 1)
  * run_export    — present + optionally INSTALL calibration.json to config/mode2/
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from . import progress
from .project import (
    aprilgrid_target,
    charuco_spec,
    load_project,
)

logger = logging.getLogger(__name__)

_MIN_INTRINSIC_SHOTS = 8       # multical/opencv floor for a usable intrinsics solve
_INTRINSIC_JSON = "work/intrinsic.json"
_CALIBRATION_JSON = "calibration.json"


def _imgdirs(project_dir: Path, sub: str, cameras: list[str]) -> dict[str, Path]:
    """{cam: dir} for the per-camera subdirs that actually contain images."""
    out = {}
    for cid in cameras:
        d = project_dir / sub / cid
        if d.is_dir() and any(d.glob("*.jpg")):
            out[cid] = d
    return out


def run_intrinsic(project_dir: Path) -> dict:
    """Phase 1 — per-camera intrinsics from the captured ChArUco shots.

    Runs ``multical intrinsic`` on whatever cameras have ≥8 shots (so cam_a can be
    solved now, cam_b later), producing ``work/intrinsic.json``. Also runs the
    OpenCV ChArUco solver per camera for a quick per-camera reprojection RMS readout.
    """
    from calibration.calibrate import (
        REPROJECTION_RMS_HARD_LIMIT_PX,
        calibrate_intrinsics,
        run_multical_intrinsics,
    )
    project_dir = Path(project_dir)
    cfg = load_project(project_dir)
    board = charuco_spec(cfg.board)
    cams = cfg.configured_cameras()
    dirs = _imgdirs(project_dir, "intrinsic", cams)
    ready = {cid: d for cid, d in dirs.items()
             if len(list(d.glob("*.jpg"))) >= _MIN_INTRINSIC_SHOTS}
    if not ready:
        raise ValueError(
            f"need ≥{_MIN_INTRINSIC_SHOTS} ChArUco shots in at least one camera "
            f"(have: {{cam: {{len}}}} = "
            f"{ {c: len(list(d.glob('*.jpg'))) for c, d in dirs.items()} })")

    work = project_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    logger.info("intrinsic: solving %s (%s)", list(ready),
                {c: len(list(d.glob('*.jpg'))) for c, d in ready.items()})
    progress.report(0, len(ready) + 1, "intrinsic:multical")
    intrinsic_json = run_multical_intrinsics(ready, board, work)
    # copy to the stable name we look for elsewhere (no-op if already there)
    stable = project_dir / _INTRINSIC_JSON
    if Path(intrinsic_json).resolve() != stable.resolve():
        shutil.copy2(intrinsic_json, stable)

    # per-camera RMS readout (OpenCV solver on the same shots) for the UI
    rms: dict[str, float] = {}
    for i, (cid, d) in enumerate(ready.items(), 1):
        progress.report(i, len(ready) + 1, f"intrinsic:rms:{cid}")
        try:
            res = calibrate_intrinsics(d, board)
            rms[cid] = round(float(res.reprojection_rms_px), 4)
            logger.info("intrinsic[%s]: RMS=%.4f px (%s)", cid, rms[cid],
                        "OK" if rms[cid] <= REPROJECTION_RMS_HARD_LIMIT_PX else "HIGH")
        except Exception as exc:                       # readout only — don't fail the phase
            logger.warning("intrinsic[%s]: RMS readout failed (%s)", cid, exc)
    return {"cameras_solved": list(ready), "rms": rms,
            "intrinsic_json": str(project_dir / _INTRINSIC_JSON)}


def run_extrinsic(project_dir: Path) -> dict:
    """Phase 2 — joint extrinsics (AprilGrid, K fixed) + floor anchor + assemble.

    Needs Phase 1's ``work/intrinsic.json``. Writes ``data/<name>/calibration.json``
    (K, D, R, t, H, P per camera), RMS-gated at 0.5 px by ``assemble_calibration``.
    """
    from calibration.calibrate import (
        assemble_calibration,
        estimate_floor_anchor_charuco,
        run_multical_extrinsics,
    )
    project_dir = Path(project_dir)
    cfg = load_project(project_dir)
    cams = cfg.configured_cameras()
    intrinsic_json = project_dir / _INTRINSIC_JSON
    if not intrinsic_json.exists():
        raise ValueError("no work/intrinsic.json — run the Intrinsic phase first")
    extr = _imgdirs(project_dir, "extrinsic", cams)
    if len(extr) < len(cams):
        raise ValueError(f"extrinsic shots missing for some cameras (have {list(extr)}, "
                         f"need {cams})")
    floors = {cid: project_dir / "floor" / f"{cid}.jpg" for cid in cams}
    missing = [cid for cid, p in floors.items() if not p.exists()]
    if missing:
        raise ValueError(f"floor shots missing for {missing} — capture one ChArUco-on-floor "
                         f"shot per camera into floor/<cam>.jpg")

    board = charuco_spec(cfg.board)
    target = aprilgrid_target(cfg.board)
    work = project_dir / "work"
    logger.info("extrinsic: solving rig %s with K fixed", cams)
    progress.report(0, 3, "extrinsic:multical")
    solution = run_multical_extrinsics(extr, target, work, intrinsic_json)
    progress.report(1, 3, "extrinsic:floor-anchor")
    anchor = estimate_floor_anchor_charuco(floors, solution, board)
    progress.report(2, 3, "extrinsic:assemble")
    calibration = assemble_calibration(solution, anchor)
    out = project_dir / _CALIBRATION_JSON
    calibration.write(out)
    rms = {cid: round(float(c.reprojection_rms_px), 4)
           for cid, c in calibration.cameras.items()}
    logger.info("extrinsic: wrote %s | RMS=%s", out, rms)
    progress.report(3, 3, "extrinsic:done")
    return {"calibration_json": str(out), "rms": rms, "cameras": list(calibration.cameras)}


def run_export(project_dir: Path, *, install: bool = False) -> dict:
    """Phase 3 — present the calibration.json; optionally INSTALL it to the live system.

    ``install=True`` copies ``calibration.json`` to ``config/mode2/calibration.json``
    (what the Backbone + monitor_web load) and stamps ``backbone.yaml``'s
    ``calibration_path`` at it. The path is taken from isical Settings.
    """
    from ..config import Settings
    project_dir = Path(project_dir)
    src = project_dir / _CALIBRATION_JSON
    if not src.exists():
        raise ValueError("no calibration.json — run the Extrinsic phase first")
    data = json.loads(src.read_text())
    rms = {cid: cam.get("reprojection_rms_px")
           for cid, cam in (data.get("cameras") or {}).items()}
    out: dict = {"calibration_json": str(src), "rms": rms,
                 "cameras": list((data.get("cameras") or {}).keys()), "installed": False}
    if install:
        cfg = Settings()
        dst = cfg.mode2_calibration_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        out["installed"] = True
        out["installed_path"] = str(dst)
        logger.info("export: installed calibration → %s", dst)
        _stamp_backbone_yaml(cfg.backbone_config_path, dst)
        out["backbone_stamped"] = str(cfg.backbone_config_path)
    return out


def _stamp_backbone_yaml(backbone_yaml: Path, calibration_path: Path) -> None:
    """Point backbone.yaml's calibration_path at the installed file (best-effort)."""
    try:
        import yaml
        if not backbone_yaml.exists():
            return
        doc = yaml.safe_load(backbone_yaml.read_text()) or {}
        doc["calibration_path"] = str(calibration_path)
        backbone_yaml.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))
        logger.info("export: stamped %s calibration_path=%s", backbone_yaml, calibration_path)
    except Exception as exc:
        logger.warning("export: could not stamp backbone.yaml (%s)", exc)


def calibration_summary(project_dir: Path) -> dict | None:
    """Vital calibration facts from calibration.json (the SOLVE output): per-camera
    reprojection RMS + focal length / principal point / distortion / world position,
    and the camera baseline (their physical separation). Returns None if not solved.
    """
    src = Path(project_dir) / _CALIBRATION_JSON
    if not src.exists():
        return None
    data = json.loads(src.read_text())
    cams = data.get("cameras") or {}
    out_cams = {}
    centers = {}
    for cid, c in cams.items():
        K = c.get("K") or [[0, 0, 0]] * 3
        D = c.get("D") or []
        t = c.get("t") or [0, 0, 0]
        wh = c.get("image_size_wh") or [0, 0]
        centers[cid] = t
        out_cams[cid] = {
            "reprojection_rms_px": c.get("reprojection_rms_px"),
            "image_size": f"{int(wh[0])}x{int(wh[1])}",
            "focal_px": [round(float(K[0][0]), 1), round(float(K[1][1]), 1)],   # fx, fy
            "principal_px": [round(float(K[0][2]), 1), round(float(K[1][2]), 1)],  # cx, cy
            "distortion": [round(float(v), 4) for v in D[:5]],
            "position_m": [round(float(v), 3) for v in t],                       # world-frame
        }
    summary = {
        "cameras": out_cams,
        "rms_gate_px": 0.5,
        "floor_anchor": data.get("floor_anchor_method"),
        "calibration_mode": data.get("calibration_mode"),
        "created_at": data.get("created_at"),
    }
    if len(centers) == 2:
        a, b = list(centers)
        import math
        d = math.dist(centers[a], centers[b])
        summary["baseline_m"] = round(d, 3)            # camera separation
    return summary


# ---- status helpers (for the phase board) ----

def phase_status(project_dir: Path) -> dict:
    """Per-phase done-state + counts for the board (no heavy work)."""
    project_dir = Path(project_dir)
    cfg = load_project(project_dir)
    cams = cfg.configured_cameras()
    intr = {cid: len(list((project_dir / "intrinsic" / cid).glob("*.jpg"))) for cid in cams}
    extr = {cid: len(list((project_dir / "extrinsic" / cid).glob("*.jpg"))) for cid in cams}
    floors = {cid: (project_dir / "floor" / f"{cid}.jpg").exists() for cid in cams}
    intrinsic_done = (project_dir / _INTRINSIC_JSON).exists()
    calibration = project_dir / _CALIBRATION_JSON
    extrinsic_done = calibration.exists()
    installed = False
    rms = {}
    if extrinsic_done:
        try:
            data = json.loads(calibration.read_text())
            rms = {cid: cam.get("reprojection_rms_px")
                   for cid, cam in (data.get("cameras") or {}).items()}
        except Exception:
            pass
        from ..config import Settings
        installed = Settings().mode2_calibration_path.exists()
    t_intr = cfg.capture.target_per_camera
    t_extr = cfg.capture.extrinsic_target
    intrinsic_captured = bool(cams) and all(intr.get(c, 0) >= t_intr for c in cams)
    extrinsic_captured = bool(cams) and all(extr.get(c, 0) >= t_extr for c in cams)
    return {
        "cameras": cams, "mode2": cfg.is_mode2(),
        "intrinsic_counts": intr, "extrinsic_counts": extr, "floor": floors,
        "intrinsic_done": intrinsic_done,
        "extrinsic_done": extrinsic_done, "rms": rms,
        "calibration_json": str(calibration) if extrinsic_done else None,
        "installed": installed,
        "intrinsic_captured": intrinsic_captured,
        "extrinsic_captured": extrinsic_captured,
        "targets": {"intrinsic": t_intr, "extrinsic": t_extr},
    }
