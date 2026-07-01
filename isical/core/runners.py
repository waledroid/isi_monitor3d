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

# Acceptance gate for the intrinsic-results badge (≤2 px = usable intrinsics).
# Distinct from REPROJECTION_RMS_HARD_LIMIT_PX (0.5 px, the solver's "excellent" log
# threshold) — do NOT conflate them.
INTRINSIC_RMS_GATE_PX: float = 2.0


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

    # persist RMS sidecar so the UI panel can read it without re-running the solver
    try:
        rms_path = project_dir / "work" / "intrinsic_rms.json"
        rms_path.write_text(json.dumps(rms))
    except Exception as exc:
        logger.warning("intrinsic: could not write intrinsic_rms.json (%s)", exc)

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


_SCALE_REFS_JSON = "work/scale_references.json"
_TARGETLESS_REPORT_JSON = "work/targetless_report.json"
_TARGETLESS_STAGE_DIR = "work/targetless_stages"


def _load_scale_references(project_dir: Path):
    """Load the operator-marked floor scale references → list[ScaleReference]."""
    from calibration.feature_extrinsics import ScaleReference
    path = Path(project_dir) / _SCALE_REFS_JSON
    if not path.exists():
        raise ValueError(
            "no scale references marked — mark ≥3 measured floor point-pairs on the "
            "targetless extrinsic page first (work/scale_references.json missing)")
    data = json.loads(path.read_text())
    refs = [ScaleReference(
        p1_a=tuple(r["p1_a"]), p1_b=tuple(r["p1_b"]),
        p2_a=tuple(r["p2_a"]), p2_b=tuple(r["p2_b"]),
        distance_m=float(r["distance_m"])) for r in data]
    if len(refs) < 3:
        raise ValueError(f"targetless needs ≥3 floor scale references, have {len(refs)}")
    return refs


def run_extrinsic_targetless(project_dir: Path) -> dict:
    """Phase 2 (targetless) — SuperPoint+LightGlue extrinsics + plane-fit floor.

    Uses Phase 1's ``work/intrinsic.json`` (K, D), the captured synchronized stereo
    pairs (``extrinsic/{cam}/*.jpg``), and the operator's ≥3 measured floor
    scale-references (``work/scale_references.json``) to solve stereo extrinsics
    without a physical target. Writes ``calibration.json``, the 3-level validation
    report, and the 5 key-stage images. The ONNX matcher weights must be vendored
    into ``models/`` — absent them this raises a clear MatcherWeightsMissing (on-rig).
    """
    from calibration.calibrate import run_targetless
    project_dir = Path(project_dir)
    cfg = load_project(project_dir)
    cams = cfg.configured_cameras()
    if len(cams) < 2:
        raise ValueError("targetless extrinsics needs both cameras configured")
    intrinsic_json = project_dir / _INTRINSIC_JSON
    if not intrinsic_json.exists():
        raise ValueError("no work/intrinsic.json — run the Intrinsic phase first")
    pair_a = project_dir / "extrinsic" / "cam_a"
    pair_b = project_dir / "extrinsic" / "cam_b"
    if not (pair_a.is_dir() and any(pair_a.glob("*.jpg"))
            and pair_b.is_dir() and any(pair_b.glob("*.jpg"))):
        raise ValueError("no synchronized stereo pairs captured for targetless extrinsics")

    references = _load_scale_references(project_dir)
    # Optional: diff against a prior AprilGrid calibration.json if one is present.
    ref_calib = None
    apr = project_dir / "calibration_aprilgrid.json"
    if apr.exists():
        ref_calib = json.loads(apr.read_text())

    out = project_dir / _CALIBRATION_JSON
    report = project_dir / _TARGETLESS_REPORT_JSON
    stages = project_dir / _TARGETLESS_STAGE_DIR
    logger.info("extrinsic(targetless): solving rig %s (%d scale refs)", cams, len(references))
    progress.report(0, 3, "targetless:solve")
    calibration = run_targetless(
        intrinsic_json=intrinsic_json, pair_dir_a=pair_a, pair_dir_b=pair_b,
        references=references, output_path=out,
        reference_calib=ref_calib, report_path=report,
        stage_image_dir=stages, render_stage_images=True,
    )
    progress.report(3, 3, "targetless:done")
    rms = {cid: round(float(c.reprojection_rms_px), 4)
           for cid, c in calibration.cameras.items()}
    logger.info("extrinsic(targetless): wrote %s | RMS=%s", out, rms)
    return {"calibration_json": str(out), "rms": rms,
            "cameras": list(calibration.cameras),
            "report_json": str(report), "method": "targetless"}


def targetless_report(project_dir: Path) -> dict | None:
    """The persisted 3-level validation report (or None if not solved)."""
    path = Path(project_dir) / _TARGETLESS_REPORT_JSON
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def calibration_matrices(project_dir: Path) -> dict | None:
    """Per-camera pose matrices from ``calibration.json`` for the Result cell.

    Returns ``{"floor_anchor_method", "calibration_mode", "cameras": {cam:
    {"R": 3x3, "t": [3], "reprojection_rms_px"}}}`` after the Extrinsic solve, or
    ``None`` when ``calibration.json`` doesn't exist yet (the notebook Result cell
    then stays in its "awaiting solve" placeholder). Read-only, never raises into
    the request.
    """
    path = Path(project_dir) / _CALIBRATION_JSON
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    cams: dict[str, dict] = {}
    for cid, cam in (data.get("cameras") or {}).items():
        cams[cid] = {
            "R": cam.get("R"),
            "t": cam.get("t"),
            "reprojection_rms_px": cam.get("reprojection_rms_px"),
        }
    return {
        "floor_anchor_method": data.get("floor_anchor_method"),
        "calibration_mode": data.get("calibration_mode"),
        "cameras": cams,
    }


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


def intrinsic_summary(project_dir: Path) -> dict:
    """Per-camera intrinsics from work/intrinsic.json.

    Returns ``{"rms_gate_px": <float>, "cameras": {<cam>: {image_size, fx, fy,
    cx, cy, K, dist, rms}}}`` when the file exists, or ``{"cameras": {}}``
    when it does not (UI hides the panel in that case).
    rms is null when work/intrinsic_rms.json is absent (old solve).
    """
    project_dir = Path(project_dir)
    intr_path = project_dir / "work" / "intrinsic.json"
    if not intr_path.exists():
        return {"cameras": {}}

    data = json.loads(intr_path.read_text())

    # load per-camera RMS sidecar (written by run_intrinsic; absent for old solves)
    rms_map: dict[str, float] = {}
    rms_path = project_dir / "work" / "intrinsic_rms.json"
    if rms_path.exists():
        try:
            rms_map = json.loads(rms_path.read_text())
        except Exception:
            pass

    out_cams: dict = {}
    for cid, c in (data.get("cameras") or {}).items():
        K = c.get("K") or [[0, 0, 0]] * 3
        # dist stored as [[k1,k2,p1,p2,k3]] (1x5 nested) or flat [k1,k2,p1,p2,k3]
        raw_dist = c.get("dist") or []
        if raw_dist and isinstance(raw_dist[0], list):
            dist = raw_dist[0]
        else:
            dist = raw_dist
        out_cams[cid] = {
            "image_size": list(c.get("image_size") or [0, 0]),
            "fx": round(float(K[0][0]), 4),
            "fy": round(float(K[1][1]), 4),
            "cx": round(float(K[0][2]), 4),
            "cy": round(float(K[1][2]), 4),
            "K": [[round(float(v), 4) for v in row] for row in K],
            "dist": [round(float(v), 6) for v in dist[:5]],
            "rms": rms_map.get(cid),   # None when absent
        }

    return {"rms_gate_px": INTRINSIC_RMS_GATE_PX, "cameras": out_cams}


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
    # Floor anchor shots are required before the extrinsic solve (run_extrinsic
    # raises without them). Solve-ready ⇒ pairs at target AND both floors present,
    # so the board never offers a green "Solve now" that will fail downstream.
    floor_done = bool(cams) and all(floors.get(c, False) for c in cams)
    missing_floor = [c for c in cams if not floors.get(c, False)]
    extrinsic_solve_ready = extrinsic_captured and floor_done
    return {
        "cameras": cams, "mode2": cfg.is_mode2(),
        "intrinsic_counts": intr, "extrinsic_counts": extr, "floor": floors,
        "intrinsic_done": intrinsic_done,
        "extrinsic_done": extrinsic_done, "rms": rms,
        "calibration_json": str(calibration) if extrinsic_done else None,
        "installed": installed,
        "intrinsic_captured": intrinsic_captured,
        "extrinsic_captured": extrinsic_captured,
        "extrinsic_floor_done": floor_done,
        "extrinsic_missing_floor": missing_floor,
        "extrinsic_solve_ready": extrinsic_solve_ready,
        "targets": {"intrinsic": t_intr, "extrinsic": t_extr},
    }
