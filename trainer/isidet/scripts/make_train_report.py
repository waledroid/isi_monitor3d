#!/usr/bin/env python3
"""Generate a human-readable Markdown report for a YOLO training run.

Assembles the artifacts Ultralytics already wrote into one ``report.md`` inside
the run directory: a metrics summary (best/final epoch), the training config and
augmentations (from ``args.yaml``), the saved plots (embedded via relative
links), and the export artifacts with sizes. Pure post-processing — nothing is
re-run.

Usage:
    python scripts/make_train_report.py                       # latest run
    python scripts/make_train_report.py --run runs/detect/models/yolo/<run>
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import yaml

ISIDET = Path(__file__).resolve().parent.parent
YOLO_RUNS = ISIDET / "runs" / "detect" / "models" / "yolo"

# Plots to embed if present: (filename, caption). confusion_matrix has two variants.
PLOTS = [
    ("results.png", "Training & validation curves"),
    ("BoxPR_curve.png", "Precision–Recall curve"),
    ("BoxF1_curve.png", "F1–confidence curve (peak = best deployment conf)"),
    ("confusion_matrix_normalized.png", "Confusion matrix (normalized)"),
    ("confusion_matrix.png", "Confusion matrix"),
    ("val_batch0_pred.jpg", "Sample validation predictions"),
    ("train_batch0.jpg", "Augmented training batch (what the model sees)"),
    ("labels.jpg", "Dataset label distribution"),
]

CONFIG_KEYS = [
    "model", "data", "imgsz", "batch", "epochs", "patience", "optimizer",
    "lr0", "lrf", "weight_decay", "cos_lr", "close_mosaic", "workers", "amp",
]
AUG_KEYS = [
    "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear",
    "perspective", "fliplr", "flipud", "mosaic", "mixup", "copy_paste", "erasing",
]

# results.csv metric column names (Ultralytics).
MAP = "metrics/mAP50-95(B)"
MAP50 = "metrics/mAP50(B)"
PREC = "metrics/precision(B)"
REC = "metrics/recall(B)"


def _overview(rows: list[dict], args: dict, best: dict, final: dict, total: int) -> list[str]:
    """Return ~3 expert-style summary lines derived from the run's metrics —
    headline outcome, the single most salient training event (instability vs
    overfitting vs clean), and the key caveat/next step."""
    map50s = [_f(r, MAP50) for r in rows]
    best_epoch = int(_f(best, "epoch"))
    best_map, best_map50 = _f(best, MAP), _f(best, MAP50)
    best_p, best_r = _f(best, PREC), _f(best, REC)
    final_map = _f(final, MAP)
    model = Path(str(args.get("model", "model"))).stem

    # 1) headline outcome
    if best_map >= 0.85:
        verdict = "excellent fit"
    elif best_map >= 0.70:
        verdict = "strong fit"
    elif best_map >= 0.50:
        verdict = "moderate fit"
    else:
        verdict = "weak fit (needs more/better data)"
    line1 = (f"**Outcome:** {verdict} — best at epoch {best_epoch}/{total}: "
             f"mAP@50 {best_map50:.3f}, mAP@50-95 {best_map:.3f}, "
             f"P {best_p:.3f} / R {best_r:.3f} (saved as best.pt).")

    # 2) most salient event: instability > overfitting > clean
    first_good = next((i for i, m in enumerate(map50s) if m > 0.5), None)
    collapse = [i for i, m in enumerate(map50s)
                if first_good is not None and i > first_good and m < 0.15]
    if len(collapse) >= 3:
        a = int(_f(rows[min(collapse)], "epoch"))
        b = int(_f(rows[max(collapse)], "epoch"))
        worst = min(map50s[i] for i in collapse)
        line2 = (f"**Watch:** training destabilized around epochs {a}–{b} "
                 f"(mAP@50 fell to ~{worst:.2f}, then recovered) — for `{model}` on a "
                 f"small set, try a smaller model, lighter mosaic, or a lower LR.")
    elif best_epoch <= 0.9 * total and (best_map - final_map) > 0.01:
        line2 = (f"**Watch:** peaked at epoch {best_epoch}, then mAP@50-95 eased to "
                 f"{final_map:.3f} by epoch {total} — mild overfitting; best.pt keeps the "
                 f"peak (consider earlier stopping or more data).")
    else:
        line2 = "**Stability:** smooth convergence — no collapses, final ≈ best."

    # 3) caveat / next step
    line3 = ("**Caveat:** scores are on the held-out val split only and can be "
             "optimistic on a small set — confirm on real deployment frames, and set the "
             "deployment confidence from the F1-curve peak.")
    return [line1, line2, line3]


def _latest_run() -> Path | None:
    if not YOLO_RUNS.is_dir():
        return None
    runs = [d for d in YOLO_RUNS.iterdir() if d.is_dir()]
    return max(runs, key=lambda d: d.stat().st_mtime) if runs else None


def _read_rows(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for raw in reader:
            # Ultralytics pads some headers/values with spaces.
            rows.append({(k or "").strip(): (v or "").strip() for k, v in raw.items()})
        return rows


def _f(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _human_size(path: Path) -> str:
    if not path.exists():
        return "—"
    mb = path.stat().st_size / 1024**2
    return f"{mb:.1f} MB"


def _fmt(v) -> str:
    return "—" if v is None else str(v)


def generate_report(run_dir: Path) -> Path | None:
    """Write ``<run_dir>/report.md``. Returns its path, or None if not a YOLO run."""
    run_dir = Path(run_dir)
    results_csv = run_dir / "results.csv"
    args_yaml = run_dir / "args.yaml"
    if not results_csv.exists() or not args_yaml.exists():
        print(f"make_train_report: {run_dir} is not a YOLO run (no results.csv/args.yaml) — skipped")
        return None

    args = yaml.safe_load(args_yaml.read_text()) or {}
    rows = _read_rows(results_csv)
    if not rows:
        print(f"make_train_report: {results_csv} is empty — skipped")
        return None

    best = max(rows, key=lambda r: _f(r, MAP))
    final = rows[-1]
    total_epochs = len(rows)
    duration_s = _f(final, "time")
    dur = f"{duration_s/60:.1f} min" if duration_s else "—"

    name = run_dir.name
    md: list[str] = []
    md.append(f"# Training report — `{name}`\n")
    md.append(f"- **Model:** `{_fmt(args.get('model'))}`")
    md.append(f"- **Dataset:** `{_fmt(args.get('data'))}`")
    md.append(f"- **Epochs run:** {total_epochs} (configured {_fmt(args.get('epochs'))}, "
              f"patience {_fmt(args.get('patience'))})")
    md.append(f"- **Duration:** {dur}")
    md.append(f"- **Run dir:** `{run_dir}`\n")

    # ---- overview (expert read, ~3 lines) ----
    md.append("## Overview (expert read)\n")
    for line in _overview(rows, args, best, final, total_epochs):
        md.append(f"- {line}")
    md.append("")

    # ---- summary ----
    md.append("## Results\n")
    md.append("| metric | best epoch | final epoch |")
    md.append("|---|---|---|")
    be = int(_f(best, "epoch"))
    fe = int(_f(final, "epoch"))
    md.append(f"| epoch | **{be}** | {fe} |")
    md.append(f"| mAP@50 | **{_f(best, MAP50):.4f}** | {_f(final, MAP50):.4f} |")
    md.append(f"| mAP@50-95 | **{_f(best, MAP):.4f}** | {_f(final, MAP):.4f} |")
    md.append(f"| precision | {_f(best, PREC):.4f} | {_f(final, PREC):.4f} |")
    md.append(f"| recall | {_f(best, REC):.4f} | {_f(final, REC):.4f} |\n")
    md.append(f"_Best epoch chosen by max mAP@50-95 → `weights/best.pt`._\n")

    # ---- config ----
    md.append("## Configuration\n")
    md.append("| key | value |")
    md.append("|---|---|")
    for k in CONFIG_KEYS:
        if k in args:
            md.append(f"| {k} | `{_fmt(args[k])}` |")
    md.append("")

    # ---- augmentations ----
    md.append("## Augmentations\n")
    md.append("| aug | value |")
    md.append("|---|---|")
    for k in AUG_KEYS:
        if k in args:
            md.append(f"| {k} | {_fmt(args[k])} |")
    md.append("")

    # ---- export artifacts ----
    w = run_dir / "weights"
    md.append("## Export artifacts\n")
    for label, p in [
        ("best.pt", w / "best.pt"),
        ("last.pt", w / "last.pt"),
        ("best.onnx (raw head, opset 17 — Backbone yolo_onnx)", w / "best.onnx"),
        ("openvino/model.xml", w / "openvino" / "model.xml"),
        ("openvino/model.bin", w / "openvino" / "model.bin"),
    ]:
        if p.exists():
            md.append(f"- `{p.relative_to(run_dir)}` — {_human_size(p)}")
    md.append("")

    # ---- plots (embed first variant that exists per kind) ----
    md.append("## Plots\n")
    seen_confusion = False
    for fname, caption in PLOTS:
        if fname.startswith("confusion_matrix"):
            if seen_confusion or not (run_dir / fname).exists():
                if (run_dir / fname).exists():
                    seen_confusion = True
                continue
            seen_confusion = True
        if (run_dir / fname).exists():
            md.append(f"**{caption}**\n")
            md.append(f"![{caption}]({fname})\n")

    # ---- notes ----
    md.append("## Notes\n")
    md.append("- Validation metrics are computed on the clean (un-augmented) val split.")
    md.append("- For deployment, read the optimal confidence off `BoxF1_curve.png` (its peak).")
    md.append("- A small val split makes single-number metrics optimistic — validate on "
              "real deployment frames before trusting them.")

    report = run_dir / "report.md"
    report.write_text("\n".join(md) + "\n")
    print(f"make_train_report: wrote {report}")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, default=None,
                    help="run dir (default: latest under runs/detect/models/yolo/)")
    args = ap.parse_args()
    run_dir = args.run or _latest_run()
    if run_dir is None:
        print("make_train_report: no run dir found")
        return 1
    return 0 if generate_report(run_dir) else 1


if __name__ == "__main__":
    sys.exit(main())
