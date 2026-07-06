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
    floor_present,
    floor_shots,
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
        EXTRINSIC_REPROJECTION_RMS_HARD_LIMIT_PX,
        assemble_calibration,
        check_extrinsic_floor_consistency,
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
    floors = {cid: floor_shots(project_dir, cid) for cid in cams}
    missing = [cid for cid, shots in floors.items() if not shots]
    if missing:
        raise ValueError(f"floor shots missing for {missing} — use the [FLOOR] button to "
                         f"auto-snap synchronized ChArUco-on-floor pairs (floor/<cam>/*.jpg)")

    board = charuco_spec(cfg.board)
    target = aprilgrid_target(cfg.board)
    work = project_dir / "work"
    logger.info("extrinsic: solving rig %s with K fixed", cams)
    progress.report(0, 3, "extrinsic:multical")
    solution = run_multical_extrinsics(extr, target, work, intrinsic_json)
    # Honesty gate: the floor pairs independently imply the relative camera
    # pose — refuse to write a calibration the two data sources disagree on
    # (catches board-scale mistakes and extrinsic-solve failures that
    # per-camera reprojection RMS is blind to).
    consistency = check_extrinsic_floor_consistency(floors, solution, board)
    if consistency.get("checked"):
        logger.info("extrinsic: floor-consistency %s", consistency)
    progress.report(1, 3, "extrinsic:floor-anchor")
    anchor = estimate_floor_anchor_charuco(floors, solution, board)
    progress.report(2, 3, "extrinsic:assemble")
    calibration = assemble_calibration(
        solution, anchor, rms_limit_px=EXTRINSIC_REPROJECTION_RMS_HARD_LIMIT_PX)
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
# The targetless method OWNS its output — separate from the board calibration.json,
# so a targetless solve can never clobber a trusted AprilGrid result (and vice versa).
_CALIBRATION_TARGETLESS_JSON = "calibration_targetless.json"


def _import_run_targetless():
    """Indirection so tests can monkeypatch the (ONNX-heavy) targetless solver."""
    from calibration.calibrate import run_targetless
    return run_targetless


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

    Fully self-contained + isolated from the board method: it reads its OWN captured
    textured-scene stereo pairs (``targetless/{cam}/*.jpg`` — NEVER the board
    ``extrinsic/`` captures) and writes its OWN output
    (``calibration_targetless.json`` — NEVER the board ``calibration.json``). It only
    SHARES the read-only intrinsics (``work/intrinsic.json``, K + D) and the operator's
    ≥3 measured floor scale-references (``work/scale_references.json``). Writes the
    3-level validation report + the 5 key-stage images too. The ONNX matcher weights
    must be vendored into ``models/`` — absent them this raises MatcherWeightsMissing.
    """
    run_targetless = _import_run_targetless()
    project_dir = Path(project_dir)
    cfg = load_project(project_dir)
    cams = cfg.configured_cameras()
    if len(cams) < 2:
        raise ValueError("targetless extrinsics needs both cameras configured")
    intrinsic_json = project_dir / _INTRINSIC_JSON
    if not intrinsic_json.exists():
        raise ValueError("no work/intrinsic.json — run the Intrinsic phase first")
    pair_a = project_dir / "targetless" / "cam_a"
    pair_b = project_dir / "targetless" / "cam_b"
    if not (pair_a.is_dir() and any(pair_a.glob("*.jpg"))
            and pair_b.is_dir() and any(pair_b.glob("*.jpg"))):
        raise ValueError("no targetless stereo pairs captured — capture textured-scene "
                         "pairs on the targetless page first (targetless/<cam>/*.jpg)")

    references = _load_scale_references(project_dir)
    # Optional: diff against a prior AprilGrid calibration.json if one is present.
    ref_calib = None
    apr = project_dir / "calibration_aprilgrid.json"
    if apr.exists():
        ref_calib = json.loads(apr.read_text())

    out = project_dir / _CALIBRATION_TARGETLESS_JSON
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


# The per-pair feature-match DIAGNOSTIC preview renders under work/ scratch — never
# a data/<name>/ root artifact and never a board file. It's a look-only aid so the
# operator can judge match quality per captured targetless pair BEFORE solving; it
# NEVER solves, writes calibration_targetless.json, or requires scale references.
_TARGETLESS_DIAG_DIR = "work/targetless_diag"


def _build_diag_matcher(models_dir: Path):
    """Construct the ONNX SuperPoint+LightGlue matcher for the diagnostic preview.

    Indirection so hermetic tests can monkeypatch in a FakeMatcher (no ONNX/weights).
    Raises MatcherWeightsMissing when the weights aren't vendored — the runner turns
    that into a clean, surfaced error rather than a crash.
    """
    from calibration.feature_extrinsics import OnnxSuperPointLightGlue
    # Lower match_threshold than the default (0.2) so the DIAGNOSTIC preview shows
    # denser matches — the operator judges pair quality from the RANSAC inlier
    # count, and more raw matches make good-vs-junk pairs easier to see.
    return OnnxSuperPointLightGlue(models_dir=models_dir, match_threshold=0.05)


def _stereo_pairs_by_index(project_dir: Path) -> list[tuple[str, Path, Path]]:
    """Sorted (pair_label, cam_a_jpg, cam_b_jpg) for the captured targetless pairs.

    Reads the targetless method's OWN captures (targetless/{cam}/*.jpg), pairing
    cam_a[i] with cam_b[i] by filename order — the same pairing the gallery + solve
    use. Trailing unmatched frames are dropped.
    """
    a_dir = project_dir / "targetless" / "cam_a"
    b_dir = project_dir / "targetless" / "cam_b"
    a = sorted(a_dir.glob("*.jpg")) if a_dir.is_dir() else []
    b = sorted(b_dir.glob("*.jpg")) if b_dir.is_dir() else []
    out: list[tuple[str, Path, Path]] = []
    for i, (pa, pb) in enumerate(zip(a, b, strict=False)):
        out.append((f"{i:03d}", pa, pb))
    return out


def preview_targetless_matches(project_dir: Path) -> dict:
    """DIAGNOSTIC — render per-pair feature-match previews for every targetless pair.

    For each captured stereo pair (``targetless/{cam}/*.jpg``) this runs the
    SuperPoint+LightGlue matcher + RANSAC (Essential-matrix) and renders two aids into
    ``work/targetless_diag/`` (scratch — never a root artifact, never a board file):

      * ``pair_<i>_matches.jpg`` — LightGlue matches with RANSAC inliers (green) vs
        outliers (red) + an "N matches, M RANSAC inliers" banner
        (:func:`calibration.feature_viz.draw_feature_matches`).
      * ``pair_<i>_keypoints.jpg`` — the matched keypoints drawn per camera, cam_a |
        cam_b side by side.

    It NEVER solves, writes ``calibration_targetless.json``, or needs scale references.
    Per-pair caching: a pair whose diag images already exist and are newer than BOTH
    its capture jpgs is skipped, so re-previewing after adding pairs only renders the
    new ones. If the matcher ONNX weights aren't vendored, raises MatcherWeightsMissing
    (surfaced by the JobRunner as a clean error, not a crash).
    """
    import cv2
    import numpy as np

    from calibration import feature_viz as viz
    from calibration.calibrate import _load_intrinsics_kd

    project_dir = Path(project_dir)
    cfg = load_project(project_dir)
    cams = cfg.configured_cameras()
    if len(cams) < 2:
        raise ValueError("targetless match preview needs both cameras configured")
    intrinsic_json = project_dir / _INTRINSIC_JSON
    if not intrinsic_json.exists():
        raise ValueError("no work/intrinsic.json — run the Intrinsic phase first")
    pairs = _stereo_pairs_by_index(project_dir)
    if not pairs:
        raise ValueError("no targetless stereo pairs captured — capture textured-scene "
                         "pairs on the targetless page first (targetless/<cam>/*.jpg)")

    kd = _load_intrinsics_kd(intrinsic_json)
    if "cam_a" not in kd:
        raise ValueError(f"cam_a intrinsics missing in {intrinsic_json}")
    K_a = np.asarray(kd["cam_a"][0], dtype=np.float64)

    diag = project_dir / _TARGETLESS_DIAG_DIR
    diag.mkdir(parents=True, exist_ok=True)

    def _fresh(dst: Path, *srcs: Path) -> bool:
        if not dst.exists():
            return False
        dmt = dst.stat().st_mtime
        return all(s.exists() and s.stat().st_mtime <= dmt for s in srcs)

    # Lazily build the (ONNX-heavy) matcher only when at least one pair needs rendering.
    matcher = None
    models_dir = Path(__file__).resolve().parents[2] / "models"

    rendered: list[dict] = []
    for label, pa_path, pb_path in pairs:
        match_name = f"pair_{label}_matches.jpg"
        kp_name = f"pair_{label}_keypoints.jpg"
        match_dst = diag / match_name
        kp_dst = diag / kp_name

        if _fresh(match_dst, pa_path, pb_path) and _fresh(kp_dst, pa_path, pb_path):
            meta_path = diag / f"pair_{label}.json"
            meta = {}
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                except (OSError, ValueError):
                    meta = {}
            rendered.append({"pair": label, "matches": match_name, "keypoints": kp_name,
                             "n_matches": meta.get("n_matches"),
                             "n_inliers": meta.get("n_inliers"), "cached": True})
            continue

        img_a = cv2.imread(str(pa_path))
        img_b = cv2.imread(str(pb_path))
        if img_a is None or img_b is None:
            logger.warning("targetless-diag: unreadable pair %s (%s / %s)", label, pa_path, pb_path)
            continue

        if matcher is None:
            matcher = _build_diag_matcher(models_dir)

        pts_a, pts_b, _scores = matcher.match(img_a, img_b)
        pts_a = np.asarray(pts_a, dtype=np.float64).reshape(-1, 2)
        pts_b = np.asarray(pts_b, dtype=np.float64).reshape(-1, 2)
        n_matches = len(pts_a)
        if n_matches >= 5:
            _E, inl = cv2.findEssentialMat(pts_a, pts_b, K_a, method=cv2.RANSAC,
                                           prob=0.999, threshold=1.5)
            mask = (inl.reshape(-1).astype(bool) if inl is not None
                    else np.zeros(n_matches, dtype=bool))
        else:
            mask = np.zeros(n_matches, dtype=bool)
        n_inliers = int(mask.sum())

        match_img = viz.draw_feature_matches(img_a, img_b, pts_a, pts_b, mask)
        cv2.imwrite(str(match_dst), match_img)
        kp_img = _draw_matched_keypoints(img_a, img_b, pts_a, pts_b, mask)
        cv2.imwrite(str(kp_dst), kp_img)
        try:
            (diag / f"pair_{label}.json").write_text(
                json.dumps({"n_matches": n_matches, "n_inliers": n_inliers}))
        except OSError:
            pass

        logger.info("targetless-diag[%s]: %d matches, %d RANSAC inliers",
                    label, n_matches, n_inliers)
        rendered.append({"pair": label, "matches": match_name, "keypoints": kp_name,
                         "n_matches": n_matches, "n_inliers": n_inliers, "cached": False})
        progress.report(len(rendered), len(pairs), f"targetless-diag:{label}")

    return {"pairs": rendered, "count": len(rendered),
            "diag_dir": str(diag), "method": "targetless_match_preview"}


def _draw_matched_keypoints(img_a, img_b, pts_a, pts_b, mask):
    """SuperPoint/LightGlue keypoints drawn per camera, cam_a | cam_b side by side.

    Reuses feature_viz's side-by-side canvas + banner; inlier keypoints are green,
    RANSAC outliers red (matching the match view's colour code)."""
    import cv2
    import numpy as np

    from calibration import feature_viz as viz

    canvas, xb = viz._hstack_pair(img_a, img_b)
    pa = np.asarray(pts_a, dtype=np.float64).reshape(-1, 2)
    pb = np.asarray(pts_b, dtype=np.float64).reshape(-1, 2)
    m = np.asarray(mask).reshape(-1).astype(bool)
    n = min(len(pa), len(pb), len(m))
    for i in range(n):
        color = viz._GREEN if m[i] else viz._RED
        cv2.circle(canvas, (round(pa[i, 0]), round(pa[i, 1])), 3, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, (round(pb[i, 0]) + xb, round(pb[i, 1])), 3, color, 1, cv2.LINE_AA)
    return viz._banner(canvas, [f"keypoints: {n} (cam_a | cam_b)"])


def targetless_calibration_matrices(project_dir: Path) -> dict | None:
    """Per-camera R/t/RMS from the TARGETLESS output (calibration_targetless.json).

    Mirrors :func:`calibration_matrices` but reads the targetless method's own file,
    so the targetless Result cell never reads (or depends on) the board result.
    Returns None when the targetless solve hasn't run yet. Read-only, never raises.
    """
    return _matrices_from(Path(project_dir) / _CALIBRATION_TARGETLESS_JSON)


def targetless_report(project_dir: Path) -> dict | None:
    """The persisted 3-level validation report (or None if not solved)."""
    path = Path(project_dir) / _TARGETLESS_REPORT_JSON
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _matrices_from(path: Path) -> dict | None:
    """Per-camera R/t/RMS + anchor from a calibration.json-shaped file, or None."""
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


def calibration_matrices(project_dir: Path) -> dict | None:
    """Per-camera pose matrices from the BOARD ``calibration.json`` for the Result
    cell. Returns ``None`` when the board Extrinsic solve hasn't run. Read-only.
    """
    return _matrices_from(Path(project_dir) / _CALIBRATION_JSON)


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


def extrinsic_summary(project_dir: Path) -> dict:
    """Per-camera extrinsics (the [R | t] pose) from calibration.json — the SOLVE
    output, shown as a matrix panel below the intrinsics.

    Returns ``{"rms_gate_px": <float>, "baseline_m": <float?>, "cameras":
    {<cam>: {R (3x3), t (3-vector), rms}}}`` when calibration.json exists, or
    ``{"cameras": {}}`` when it does not (UI hides the panel).
    ``R``/``t`` are the world→camera extrinsic (``P = K[R|t]``).
    """
    from calibration.calibrate import EXTRINSIC_REPROJECTION_RMS_HARD_LIMIT_PX

    src = Path(project_dir) / _CALIBRATION_JSON
    if not src.exists():
        return {"cameras": {}}
    data = json.loads(src.read_text())
    out_cams: dict = {}
    centers: dict = {}
    for cid, c in (data.get("cameras") or {}).items():
        R = c.get("R") or [[0, 0, 0]] * 3
        t = c.get("t") or [0, 0, 0]
        centers[cid] = t
        out_cams[cid] = {
            "R": [[round(float(v), 4) for v in row] for row in R],
            "t": [round(float(v), 4) for v in t],
            "rms": c.get("reprojection_rms_px"),
        }
    out: dict = {"rms_gate_px": EXTRINSIC_REPROJECTION_RMS_HARD_LIMIT_PX,
                 "cameras": out_cams}
    if len(centers) == 2:
        import math
        a, b = list(centers)
        out["baseline_m"] = round(math.dist(centers[a], centers[b]), 3)
    return out


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
    floors = {cid: floor_present(project_dir, cid) for cid in cams}
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
