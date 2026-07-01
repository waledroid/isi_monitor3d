"""Three-level validation report for the targetless stereo extrinsics (Stage 3).

The targetless method (:mod:`feature_extrinsics` + :mod:`floor_planefit`) stays
**experimental and opt-in** until it meets the **metric** KPIs — *not* reprojection
alone. This module produces the side-by-side report that gates adoption. Three
levels, per the plan:

1. **Geometric (pixel)** — always computable from the two ``calibration.json``
   results: rotation / translation difference vs the AprilGrid reference (when
   available), per-camera reprojection RMS, camera-baseline agreement, and the
   floor-anchor method used. Computed here, hermetically.
2. **Object-level metric (metres) — the real acceptance objective** — errors in
   metres: known inter-object distances, object dimensions / height, and
   triangulated-position error vs tape-measured ground truth. The fields are
   **structured** and computed **when the operator supplies measurements**; with no
   measurements this level is reported ``pending`` (it lights up on the rig).
3. **End-to-end perception** — run the real detection → cross-camera fusion → 3D and
   compare fused inter-object distances to reality. Requires **live detections**, so
   here it is a clearly-marked ``pending`` hook that the on-rig harness fills in.

**The acceptance gate is the metric KPIs (level 2/3), not reprojection alone.** The
report exposes ``accepted`` / ``acceptance_reason`` making that explicit: a report
with only the geometric level satisfied is ``accepted=False`` /
``acceptance_reason="metric_pending"``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

# KPI thresholds (mirroring CLAUDE.md's KPI table + the plan's gates).
REPROJECTION_RMS_GATE_PX = 2.0
HOMOGRAPHY_GATE_PX = 2.0
ROTATION_GATE_DEG = 1.0            # targetless-vs-AprilGrid rotation agreement
BASELINE_REL_GATE = 0.05          # 5 % baseline-length agreement
METRIC_DISTANCE_GATE_M = 0.05     # inter-object / dimension error acceptance (metres)
METRIC_POSITION_GATE_M = 0.10     # triangulated-position error acceptance (metres)


# ---------------------------------------------------------------------------
# Level 2 inputs — operator/tape-measured ground truth (metres)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ObjectMeasurement:
    """One tape-measured metric ground-truth item to validate against.

    Exactly one of the value fields is meaningful per ``kind``:

    * ``kind="distance"`` — ``measured_m`` is the real distance between two
      world points ``point_a`` / ``point_b`` (metres); ``estimated_m`` is the
      pipeline's estimate.
    * ``kind="dimension"`` / ``"height"`` — ``measured_m`` is the real object
      size; ``estimated_m`` the pipeline's estimate.
    * ``kind="position"`` — ``point_a`` is the tape-measured world position and
      ``estimated_point`` the pipeline's triangulated position; the error is the
      Euclidean distance between them.
    """

    label: str
    kind: str                              # distance | dimension | height | position
    measured_m: float | None = None
    estimated_m: float | None = None
    point_a: tuple[float, float, float] | None = None
    point_b: tuple[float, float, float] | None = None
    estimated_point: tuple[float, float, float] | None = None


# ---------------------------------------------------------------------------
# calibration.json helpers (consumer-side — no backbone.runtime import)
# ---------------------------------------------------------------------------


def _camera_center(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Camera centre in world coords: C = -R^T t (world←cam extrinsics)."""
    return -np.asarray(R, float).T @ np.asarray(t, float).reshape(3)


def _baseline_m(calib: dict) -> float | None:
    cams = calib.get("cameras") or {}
    if len(cams) != 2:
        return None
    centers = []
    for c in cams.values():
        centers.append(_camera_center(np.asarray(c["R"], float), np.asarray(c["t"], float)))
    return float(np.linalg.norm(centers[0] - centers[1]))


def _rotation_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Geodesic angle (deg) between two rotations."""
    Rd = np.asarray(Ra, float) @ np.asarray(Rb, float).T
    cos = (np.trace(Rd) - 1.0) / 2.0
    cos = max(-1.0, min(1.0, cos))
    return float(math.degrees(math.acos(cos)))


def _relative_pose(calib: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """cam_b ← cam_a relative pose from a 2-camera calibration.json."""
    cams = calib.get("cameras") or {}
    if "cam_a" not in cams or "cam_b" not in cams:
        return None
    Ra, ta = np.asarray(cams["cam_a"]["R"], float), np.asarray(cams["cam_a"]["t"], float).reshape(3)
    Rb, tb = np.asarray(cams["cam_b"]["R"], float), np.asarray(cams["cam_b"]["t"], float).reshape(3)
    R_rel = Rb @ Ra.T
    t_rel = tb - R_rel @ ta
    return R_rel, t_rel


# ---------------------------------------------------------------------------
# Level 1 — geometric
# ---------------------------------------------------------------------------


def geometric_level(
    targetless_calib: dict,
    *,
    reference_calib: dict | None = None,
    reprojection_rms_px: float | None = None,
) -> dict[str, Any]:
    """Pixel-level checks computable from calibration.json alone.

    ``reference_calib`` is the AprilGrid result on the same rig (when available) to
    diff R/t against. ``reprojection_rms_px`` is the targetless solve's BA RMS.
    """
    cams = targetless_calib.get("cameras") or {}
    per_cam_rms = {cid: c.get("reprojection_rms_px") for cid, c in cams.items()}
    worst_rms = max((v for v in per_cam_rms.values() if v is not None), default=None)

    out: dict[str, Any] = {
        "per_camera_rms_px": per_cam_rms,
        "worst_camera_rms_px": worst_rms,
        "bundle_adjustment_rms_px": reprojection_rms_px,
        "baseline_m": _baseline_m(targetless_calib),
        "floor_anchor_method": targetless_calib.get("floor_anchor_method"),
        "rms_gate_px": REPROJECTION_RMS_GATE_PX,
    }

    checks: dict[str, bool] = {}
    if worst_rms is not None:
        checks["reprojection_rms"] = worst_rms <= REPROJECTION_RMS_GATE_PX
    if reprojection_rms_px is not None:
        checks["homography_px"] = reprojection_rms_px <= HOMOGRAPHY_GATE_PX

    if reference_calib is not None:
        rel_t = _relative_pose(targetless_calib)
        rel_r = _relative_pose(reference_calib)
        if rel_t is not None and rel_r is not None:
            ang = _rotation_angle_deg(rel_t[0], rel_r[0])
            out["rotation_diff_deg"] = ang
            checks["rotation_agreement"] = ang <= ROTATION_GATE_DEG
        b_t = _baseline_m(targetless_calib)
        b_r = _baseline_m(reference_calib)
        if b_t is not None and b_r is not None and b_r > 1e-9:
            rel = abs(b_t - b_r) / b_r
            out["baseline_diff_rel"] = rel
            out["reference_baseline_m"] = b_r
            checks["baseline_agreement"] = rel <= BASELINE_REL_GATE
        out["has_reference"] = True
    else:
        out["has_reference"] = False

    out["checks"] = checks
    out["passed"] = bool(checks) and all(checks.values())
    return out


# ---------------------------------------------------------------------------
# Level 2 — object-level metric (metres)
# ---------------------------------------------------------------------------


def _measurement_error_m(m: ObjectMeasurement) -> float | None:
    if m.kind == "position":
        if m.point_a is None or m.estimated_point is None:
            return None
        return float(np.linalg.norm(np.asarray(m.point_a, float) - np.asarray(m.estimated_point, float)))
    if m.measured_m is None or m.estimated_m is None:
        return None
    return float(abs(m.measured_m - m.estimated_m))


def metric_level(measurements: list[ObjectMeasurement] | None) -> dict[str, Any]:
    """Object-level metric accuracy in metres — the real acceptance objective.

    With no measurements supplied this level is ``status="pending"`` (it requires
    on-rig, tape-measured ground truth). When measurements ARE supplied it computes
    per-item errors and gates them against the metric KPIs.
    """
    if not measurements:
        return {
            "status": "pending",
            "note": "no metric measurements supplied — provide tape-measured "
                    "ground truth on the rig (inter-object distances, dimensions, "
                    "heights, triangulated positions) to evaluate acceptance",
            "items": [],
            "passed": None,
        }

    items: list[dict[str, Any]] = []
    checks: list[bool] = []
    for m in measurements:
        err = _measurement_error_m(m)
        gate = METRIC_POSITION_GATE_M if m.kind == "position" else METRIC_DISTANCE_GATE_M
        ok = None if err is None else bool(err <= gate)
        if ok is not None:
            checks.append(ok)
        items.append({
            "label": m.label, "kind": m.kind,
            "measured_m": m.measured_m, "estimated_m": m.estimated_m,
            "error_m": err, "gate_m": gate, "passed": ok,
        })

    return {
        "status": "evaluated",
        "distance_gate_m": METRIC_DISTANCE_GATE_M,
        "position_gate_m": METRIC_POSITION_GATE_M,
        "items": items,
        "passed": bool(checks) and all(checks),
    }


# ---------------------------------------------------------------------------
# Level 3 — end-to-end perception (detection → fusion → 3D)
# ---------------------------------------------------------------------------


def end_to_end_level(fused_result: dict | None = None) -> dict[str, Any]:
    """End-to-end detection→fusion→3D vs real-world inter-object distances.

    This needs **live detections** from both cameras run through the actual
    homography-fusion + triangulation stack, so hermetically it is a ``pending``
    hook. The on-rig harness populates ``fused_result`` with
    ``{"distances": [{"label", "measured_m", "fused_m"}, ...]}`` and this evaluates
    it against the metric distance gate.
    """
    if not fused_result:
        return {
            "status": "pending",
            "note": "requires on-rig live detections through detection → "
                    "cross-camera fusion → 3D; run the end-to-end harness to "
                    "compare fused inter-object distances to real-world truth",
            "distances": [],
            "passed": None,
        }
    dists = fused_result.get("distances") or []
    items: list[dict[str, Any]] = []
    checks: list[bool] = []
    for d in dists:
        measured = d.get("measured_m")
        fused = d.get("fused_m")
        if measured is None or fused is None:
            items.append({**d, "error_m": None, "passed": None})
            continue
        err = abs(float(measured) - float(fused))
        ok = err <= METRIC_DISTANCE_GATE_M
        checks.append(ok)
        items.append({**d, "error_m": err, "gate_m": METRIC_DISTANCE_GATE_M, "passed": ok})
    return {
        "status": "evaluated",
        "distance_gate_m": METRIC_DISTANCE_GATE_M,
        "distances": items,
        "passed": bool(checks) and all(checks),
    }


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ValidationReport:
    geometric: dict[str, Any]
    metric: dict[str, Any]
    end_to_end: dict[str, Any]
    accepted: bool
    acceptance_reason: str
    summary_lines: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_validation_report(
    targetless_calib: dict,
    *,
    reference_calib: dict | None = None,
    reprojection_rms_px: float | None = None,
    measurements: list[ObjectMeasurement] | None = None,
    end_to_end_result: dict | None = None,
) -> ValidationReport:
    """Assemble the 3-level report and decide acceptance.

    **Acceptance is metric-first**: the targetless result is adopted only when the
    **metric** level (2) passes — geometric agreement alone is *not* sufficient. If
    the metric level is pending (no on-rig measurements), ``accepted=False`` with
    reason ``"metric_pending"``, even when the geometric checks all pass.
    """
    geo = geometric_level(
        targetless_calib, reference_calib=reference_calib,
        reprojection_rms_px=reprojection_rms_px,
    )
    met = metric_level(measurements)
    e2e = end_to_end_level(end_to_end_result)

    if met["passed"] is None:
        accepted, reason = False, "metric_pending"
    elif not met["passed"]:
        accepted, reason = False, "metric_failed"
    elif not geo["passed"]:
        accepted, reason = False, "geometric_failed"
    elif e2e["passed"] is False:
        accepted, reason = False, "end_to_end_failed"
    else:
        accepted, reason = True, "metric_kpis_met"

    lines = _summary_lines(geo, met, e2e, accepted, reason)
    return ValidationReport(
        geometric=geo, metric=met, end_to_end=e2e,
        accepted=accepted, acceptance_reason=reason, summary_lines=lines,
    )


def _summary_lines(geo, met, e2e, accepted, reason) -> list[str]:
    lines: list[str] = []
    worst = geo.get("worst_camera_rms_px")
    if worst is not None:
        lines.append(f"reproj RMS {worst:.3f} px (gate {REPROJECTION_RMS_GATE_PX})")
    if geo.get("baseline_m") is not None:
        lines.append(f"baseline {geo['baseline_m']:.3f} m")
    if "rotation_diff_deg" in geo:
        lines.append(f"vs AprilGrid: rot {geo['rotation_diff_deg']:.2f} deg")
    if met["status"] == "pending":
        lines.append("metric: PENDING (needs on-rig measurements)")
    else:
        lines.append(f"metric: {'PASS' if met['passed'] else 'FAIL'} ({len(met['items'])} items)")
    lines.append(f"end-to-end: {e2e['status'].upper()}")
    lines.append(f"ACCEPTED={accepted} ({reason})")
    return lines
