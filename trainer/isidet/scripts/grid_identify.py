"""Fit a grid to the etagere from its black pipe-joint clamps and cut cell crops.

The black clamp elements at post/rail junctions are the rack's structural
markers. Per frame:

  white-rack mask (adaptive brightness, largest blob)
  -> black joint detection: small dark compact blobs ON the white structure
  -> joints clustered into ROWS (shelf levels); each row line is fit through
     its joints; the LEFT/RIGHT outer bounds are fit through the per-row
     extreme joints
  -> columns: each row span divided in thirds (bins are equal width; there
     are no pipes between columns)
  -> cells = (visible joint rows - 1) x 3.  A frame does NOT need the full
     3x3: only the rows whose joints are visible are fit (partial grids).

Rejection gates:
  - fewer than 2 joint rows visible          -> nothing to fit
  - consecutive joint rows closer than
    --min-row-gap px (grazing view / bogus)  -> reject frame
  - rack span too narrow                     -> reject frame
  - any cell corner > --skew-max from 90 deg -> reject frame

A temporal pass over each video validates fits against their neighbours and
rescues failed frames by interpolation (only between same-shape grids).

Usage:
  python scripts/grid_identify.py --input data/etagere/images \
      --debug-dir data/etagere/grid_debug --limit 10 --debug-only
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Rack mask must cover at least this fraction of the frame.
MIN_MASK_FRAC = 0.10
# Black joint blob constraints (px at ~848x464).
JOINT_AREA = (25, 1500)
JOINT_MAX_SIDE = 70
# Fraction of white-rack pixels required in a ring around a joint blob.
JOINT_WHITE_RING_FRAC = 0.30
# Joint y-clustering: a row cluster spans at most this many px.
ROW_CLUSTER_SPAN_PX = 40
# A joint row needs at least this many joints (rails have joints at both posts).
MIN_JOINTS_PER_ROW = 2
# ...and real horizontal extent: a rail's joints sit at both posts. This
# drops degenerate vertical chains (e.g. striped tape along one post).
MIN_ROW_XSPREAD_PX = 70
# Rails are near-horizontal; clamp the fitted row-line slope.
MAX_ROW_SLOPE = 0.6
# Joint rows closer than this belong to the SAME shelf level (rail + brace)
# and are merged before the min-row-gap gate applies.
ROW_MERGE_PX = 55
# Minimum horizontal rack span (px) between the outer bounds.
MIN_SPAN_PX = 140
# Bins are wider than tall; a median cell aspect (w/h) below this means a
# rail's joints were missed and the "cell" spans two shelf levels.
MIN_CELL_ASPECT = 0.8


@dataclass
class FitResult:
    ok: bool
    reason: str
    corners: np.ndarray | None  # (R, 4, 2): R joint-row lines x 4 col positions
    joints: np.ndarray | None   # (N, 2) detected joint centroids (debug)
    max_skew: float

    @property
    def n_rows(self) -> int:
        return 0 if self.corners is None else self.corners.shape[0]


def _rack_mask(img: np.ndarray) -> tuple[np.ndarray | None, int]:
    """Largest bright low-saturation blob = the white rack + bins."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v_thresh, _ = cv2.threshold(hsv[:, :, 2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    v_thresh = max(165.0, float(v_thresh))  # only ever rises above the validated base
    raw = cv2.inRange(hsv, (0, 0, v_thresh), (180, 55, 255))
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(raw)
    if n < 2:
        return None, 0
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (labels == idx).astype(np.uint8)
    return mask, int(stats[idx, cv2.CC_STAT_AREA])


def _black_joints(img: np.ndarray, rack: np.ndarray) -> np.ndarray:
    """Centroids (N, 2) of small dark compact blobs sitting ON the white rack."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    dark = cv2.inRange(hsv, (0, 0, 0), (180, 255, 80))
    near = cv2.dilate(rack, np.ones((13, 13), np.uint8))
    dark = cv2.bitwise_and(dark, dark, mask=near)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(dark)
    joints = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if not (JOINT_AREA[0] <= area <= JOINT_AREA[1]):
            continue
        if w > JOINT_MAX_SIDE or h > JOINT_MAX_SIDE:
            continue
        # a joint is embedded in white pipework: its surrounding ring must be
        # substantially white (rejects dark bin contents / background holes)
        blob = (labels == i).astype(np.uint8)
        ring = cv2.dilate(blob, np.ones((11, 11), np.uint8)) - blob
        if rack[ring > 0].mean() < JOINT_WHITE_RING_FRAC:
            continue
        joints.append(centroids[i])
    return np.array(joints, dtype=np.float64).reshape(-1, 2)


def _joint_rows(joints: np.ndarray) -> list[np.ndarray]:
    """Cluster joints into shelf-level rows by y (span-bounded)."""
    order = joints[:, 1].argsort()
    rows, current = [], [joints[order[0]]]
    for idx in order[1:]:
        pt = joints[idx]
        if pt[1] - current[0][1] <= ROW_CLUSTER_SPAN_PX:
            current.append(pt)
        else:
            rows.append(np.array(current))
            current = [pt]
    rows.append(np.array(current))
    rows = [
        r for r in rows
        if len(r) >= MIN_JOINTS_PER_ROW and np.ptp(r[:, 0]) >= MIN_ROW_XSPREAD_PX
    ]
    # merge rows belonging to the same shelf level (rail + brace clamps)
    merged: list[np.ndarray] = []
    for r in rows:
        if merged and r[:, 1].mean() - merged[-1][:, 1].mean() < ROW_MERGE_PX:
            merged[-1] = np.vstack([merged[-1], r])
        else:
            merged.append(r)
    return merged


def _fit_row_line(row: np.ndarray) -> tuple[float, float]:
    """y = a*x + b through one row's joints (slope clamped: rails are ~horizontal)."""
    a, b = np.polyfit(row[:, 0], row[:, 1], 1)
    if abs(a) > MAX_ROW_SLOPE:
        a = 0.0
        b = float(row[:, 1].mean())
    return float(a), float(b)


def _fit_bound(points: list[np.ndarray]) -> tuple[float, float]:
    """x = a*y + b through the per-row extreme joints."""
    pts = np.array(points)
    if len(pts) == 1:
        return 0.0, float(pts[0, 0])
    a, b = np.polyfit(pts[:, 1], pts[:, 0], 1)
    return float(a), float(b)


def _row_x_bound_intersect(row_ab: tuple, bound_ab: tuple) -> np.ndarray:
    ar, br = row_ab  # y = ar*x + br
    ab, bb = bound_ab  # x = ab*y + bb
    y = (ar * bb + br) / (1.0 - ar * ab)
    x = ab * y + bb
    return np.array([x, y])


def _cell_quad(corners: np.ndarray, r: int, c: int) -> np.ndarray:
    return np.array([corners[r, c], corners[r, c + 1],
                     corners[r + 1, c + 1], corners[r + 1, c]])


def _max_corner_skew(corners: np.ndarray) -> float:
    """Worst deviation (deg) of any cell corner angle from 90."""
    worst = 0.0
    for r in range(corners.shape[0] - 1):
        for c in range(3):
            quad = _cell_quad(corners, r, c)
            for i in range(4):
                p0, p1, p2 = quad[i - 1], quad[i], quad[(i + 1) % 4]
                u, v = p0 - p1, p2 - p1
                cosang = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9)
                ang = math.degrees(math.acos(np.clip(cosang, -1, 1)))
                worst = max(worst, abs(ang - 90.0))
    return worst


def fit_grid(img: np.ndarray, skew_max: float, min_row_gap: float) -> FitResult:
    h_px, w_px = img.shape[:2]
    rack, area = _rack_mask(img)
    if rack is None or area < MIN_MASK_FRAC * w_px * h_px:
        return FitResult(False, "rack not found (mask too small)", None, None, 0.0)

    joints = _black_joints(img, rack)
    if len(joints) < 2 * MIN_JOINTS_PER_ROW:
        return FitResult(False, f"too few joints ({len(joints)})", None, joints, 0.0)

    rows = _joint_rows(joints)
    if len(rows) < 2:
        return FitResult(False, f"only {len(rows)} joint row(s)", None, joints, 0.0)

    row_lines = [_fit_row_line(r) for r in rows]
    # user gate: consecutive joint rows too close -> grazing view, don't fit
    mid = w_px / 2
    ys = [a * mid + b for a, b in row_lines]
    gaps = np.diff(ys)
    if gaps.min() < min_row_gap:
        return FitResult(
            False, f"joint rows too close ({gaps.min():.0f} px < {min_row_gap:.0f})",
            None, joints, 0.0,
        )

    left = _fit_bound([r[r[:, 0].argmin()] for r in rows])
    right = _fit_bound([r[r[:, 0].argmax()] for r in rows])

    corners = np.zeros((len(rows), 4, 2))
    for i, rl in enumerate(row_lines):
        p_l = _row_x_bound_intersect(rl, left)
        p_r = _row_x_bound_intersect(rl, right)
        for j, u in enumerate((0, 1 / 3, 2 / 3, 1)):
            corners[i, j] = p_l * (1 - u) + p_r * u

    span = float(np.linalg.norm(corners[0, 3] - corners[0, 0]))
    if span < MIN_SPAN_PX:
        return FitResult(False, f"rack span too narrow ({span:.0f} px)", corners, joints, 0.0)

    aspects = []
    for r in range(len(rows) - 1):
        for c in range(3):
            quad = _cell_quad(corners, r, c)
            cw = (np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3])) / 2
            ch = (np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1])) / 2
            aspects.append(cw / max(ch, 1e-6))
    med_aspect = float(np.median(aspects))
    if med_aspect < MIN_CELL_ASPECT:
        return FitResult(
            False, f"cells too tall (aspect {med_aspect:.2f} — missed rail?)",
            corners, joints, 0.0,
        )

    max_skew = _max_corner_skew(corners)
    if max_skew > skew_max:
        return FitResult(False, f"skew {max_skew:.1f} deg > {skew_max}", corners, joints, max_skew)
    return FitResult(True, "", corners, joints, max_skew)


# --- temporal pass over one video's ordered frames -------------------------
# Max frame-number gap bridged by interpolation (at 2 fps, 2 = one second).
RESCUE_MAX_GAP = 2
# An accepted fit whose corners deviate more than this (px) from the
# interpolation of its accepted neighbours is a temporal outlier.
OUTLIER_DEV_PX = 80.0
# Interpolation is only trusted when the bracketing fits moved less than
# this (px) between themselves (slow / steady camera).
RESCUE_MAX_MOTION_PX = 120.0


def _interp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a * (1.0 - t) + b * t


def temporal_pass(nums: list[int], fits: list[FitResult], skew_max: float) -> list[FitResult]:
    """Validate and rescue per-frame fits using the surrounding video slice.

    Only same-shape grids (equal joint-row count) are compared/interpolated.
    """
    ok_idx = [i for i, f in enumerate(fits) if f.ok]

    def bracket(i: int, pool: list[int], n_rows: int | None = None):
        def shape_ok(j):
            return n_rows is None or fits[j].n_rows == n_rows or fits[j].corners is None

        prev = [j for j in pool if j < i and nums[i] - nums[j] <= RESCUE_MAX_GAP and shape_ok(j)]
        nxt = [j for j in pool if j > i and nums[j] - nums[i] <= RESCUE_MAX_GAP and shape_ok(j)]
        if not prev or not nxt:
            return None
        return prev[-1], nxt[0]

    out = list(fits)
    # 1) validation (predictions always come from the original accepted set)
    for i in ok_idx:
        pool = [j for j in ok_idx if j != i and fits[j].n_rows == fits[i].n_rows]
        br = bracket(i, pool)
        if br is None:
            continue
        j, m = br
        t = (nums[i] - nums[j]) / (nums[m] - nums[j])
        pred = _interp(fits[j].corners, fits[m].corners, t)
        dev = float(np.abs(fits[i].corners - pred).max())
        if dev > OUTLIER_DEV_PX:
            out[i] = FitResult(False, f"temporal outlier (dev {dev:.0f} px)",
                               fits[i].corners, fits[i].joints, fits[i].max_skew)

    # 2) rescue, bracketed by same-shape fits that survived validation
    ok_after = [i for i, f in enumerate(out) if f.ok]
    for i, f in enumerate(out):
        if f.ok:
            continue
        for n_rows in {out[j].n_rows for j in ok_after}:
            pool = [j for j in ok_after if out[j].n_rows == n_rows]
            br = bracket(i, pool)
            if br is None:
                continue
            j, m = br
            motion = float(np.abs(out[j].corners - out[m].corners).max())
            if motion > RESCUE_MAX_MOTION_PX:
                continue
            t = (nums[i] - nums[j]) / (nums[m] - nums[j])
            corners = _interp(out[j].corners, out[m].corners, t)
            skew = _max_corner_skew(corners)
            if skew <= skew_max:
                out[i] = FitResult(True, "interp", corners, f.joints, skew)
                break
    return out


def draw_debug(img: np.ndarray, fit: FitResult) -> np.ndarray:
    out = img.copy()
    color = (0, 255, 0) if fit.ok else (0, 0, 255)
    if fit.joints is not None:
        for x, y in fit.joints.astype(int):
            cv2.circle(out, (x, y), 6, (255, 0, 255), 2)
    if fit.corners is not None:
        for r in range(fit.n_rows - 1):
            for c in range(3):
                quad = _cell_quad(fit.corners, r, c).astype(int)
                cv2.polylines(out, [quad], True, (0, 220, 255), 2)
                cx, cy = quad.mean(axis=0).astype(int)
                cv2.putText(out, f"r{r + 1}c{c + 1}", (cx - 22, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)
    if fit.ok:
        label = f"OK {fit.n_rows - 1}x3 skew {fit.max_skew:.1f}" + (
            " (interp)" if fit.reason == "interp" else "")
    else:
        label = f"REJECT: {fit.reason}"
    cv2.putText(out, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return out


def extract_cells(img: np.ndarray, fit: FitResult, stem: str, cells_dir: Path) -> int:
    n = 0
    h_px, w_px = img.shape[:2]
    for r in range(fit.n_rows - 1):
        for c in range(3):
            quad = _cell_quad(fit.corners, r, c)
            x0, y0 = quad.min(axis=0).astype(int)
            x1, y1 = quad.max(axis=0).astype(int)
            x0, y0 = max(x0, 0), max(y0, 0)
            x1, y1 = min(x1, w_px), min(y1, h_px)
            if x1 - x0 < 20 or y1 - y0 < 20:
                continue
            cv2.imwrite(str(cells_dir / f"{stem}_r{r + 1}c{c + 1}.jpg"), img[y0:y1, x0:x1])
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True, help="folder of frames")
    ap.add_argument("--debug-dir", type=Path, required=True)
    ap.add_argument("--cells-dir", type=Path, help="where cell crops go")
    ap.add_argument("--limit", type=int, default=0, help="sample N frames evenly (0 = all)")
    ap.add_argument("--skew-max", type=float, default=20.0, help="max corner deviation (deg)")
    ap.add_argument("--min-row-gap", type=float, default=45.0,
                    help="min px between joint rows; below this the frame is not fit")
    ap.add_argument("--debug-only", action="store_true", help="draw grids, no crops")
    ap.add_argument("--no-temporal", action="store_true",
                    help="disable the video-slice validation/rescue pass")
    args = ap.parse_args()

    files = sorted(args.input.glob("*.jpg"))
    if args.limit and len(files) > args.limit:
        idx = np.linspace(0, len(files) - 1, args.limit).astype(int)
        files = [files[i] for i in idx]

    args.debug_dir.mkdir(parents=True, exist_ok=True)
    if not args.debug_only and args.cells_dir:
        args.cells_dir.mkdir(parents=True, exist_ok=True)

    # pass 1: independent per-frame fits
    fits = {f: fit_grid(cv2.imread(str(f)), args.skew_max, args.min_row_gap) for f in files}

    # pass 2: temporal validation + rescue per video sequence
    if not args.no_temporal:
        groups: dict[str, list[tuple[int, Path]]] = {}
        for f in files:
            prefix, _, num = f.stem.rpartition("_")
            if prefix and num.isdigit():
                groups.setdefault(prefix, []).append((int(num), f))
        for items in groups.values():
            items.sort()
            nums = [n for n, _ in items]
            seq = [fits[f] for _, f in items]
            for (_, f), fit in zip(items, temporal_pass(nums, seq, args.skew_max), strict=True):
                fits[f] = fit

    ok_n = rej_n = interp_n = cells_n = 0
    for f in files:
        fit = fits[f]
        img = cv2.imread(str(f))
        cv2.imwrite(str(args.debug_dir / f.name), draw_debug(img, fit))
        if fit.ok:
            ok_n += 1
            interp_n += fit.reason == "interp"
            if not args.debug_only and args.cells_dir:
                cells_n += extract_cells(img, fit, f.stem, args.cells_dir)
        else:
            rej_n += 1
            print(f"reject {f.name}: {fit.reason}")
    print(f"\nframes={len(files)} ok={ok_n} (interp={interp_n}) rejected={rej_n} cells={cells_n}")


if __name__ == "__main__":
    main()
