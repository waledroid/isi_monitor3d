"""F5 — end-to-end latency under production load (G3 artifact).

Reads the 61 MQTT diagnostics heartbeats captured on 2026-07-20
(thesis/measurements/G3_mqtt_diagnostics_20260720.jsonl) and plots the
rolling-window p50/p95/p99 capture->publish latency against the 200 ms KPI.

Design (dataviz method): ordered percentiles of ONE quantity -> single-hue
ordinal blue ramp (validated #86b6ef / #2a78d6 / #0d366b on #fcfcfb, ordinal
mode, all checks PASS); KPI threshold as a recessive dashed gray reference
line; direct end-labels + compact legend; text in ink tokens, never series
color; y from 0 so the x2.6 margin is visible honestly.

Run:  conda activate monitor3d && python thesis/figures/src/f5_latency.py
Out:  thesis/figures/F5_latency.{pdf,png}
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]  # thesis/
RAW = ROOT / "measurements" / "G3_mqtt_diagnostics_20260720.jsonl"
OUT = ROOT / "figures"

INK = "#1a1a18"
INK_MUTED = "#6b6b66"
GRID = "#e8e7e3"
SURFACE = "#fcfcfb"
RAMP = {"p50": "#86b6ef", "p95": "#2a78d6", "p99": "#0d366b"}
KPI_MS = 200.0


def load() -> tuple[list[float], dict[str, list[float]]]:
    t0, times = None, []
    series: dict[str, list[float]] = {"p50": [], "p95": [], "p99": []}
    for line in RAW.read_text().splitlines():
        try:
            msg = json.loads(line.split(" ", 1)[1])
        except (IndexError, json.JSONDecodeError):
            continue
        lat = msg.get("latency_ms") or {}
        if not lat.get("n"):
            continue
        ts = float(msg["ts"])
        t0 = ts if t0 is None else t0
        times.append((ts - t0) / 60.0)
        for k in series:
            series[k].append(float(lat[k]))
    return times, series


def main() -> None:
    times, series = load()
    n = len(times)

    fig, ax = plt.subplots(figsize=(5.0, 2.9), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.axhline(KPI_MS, color=INK_MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.annotate(
        f"KPI target {KPI_MS:.0f} ms",
        xy=(times[-1], KPI_MS),
        xytext=(0, 4),
        textcoords="offset points",
        ha="right",
        fontsize=7.5,
        color=INK_MUTED,
    )

    for key in ("p99", "p95", "p50"):  # draw dark->light so light sits on top
        ax.plot(times, series[key], color=RAMP[key], lw=2.0, zorder=3, label=key)
        ax.annotate(
            f"{key}  {series[key][-1]:.0f} ms",
            xy=(times[-1], series[key][-1]),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=7.5,
            color=INK,
        )

    ax.set_xlim(0, times[-1] * 1.16)  # right margin for end labels
    ax.set_ylim(0, 215)
    ax.set_xlabel("time (min)", fontsize=8, color=INK)
    ax.set_ylabel("capture→publish latency (ms)", fontsize=8, color=INK)
    ax.tick_params(labelsize=7.5, colors=INK_MUTED, length=3)
    ax.grid(axis="y", color=GRID, lw=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK_MUTED)
        ax.spines[s].set_linewidth(0.8)
    ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=7.5,
        handlelength=1.4,
        labelcolor=INK,
        ncols=3,
        columnspacing=1.2,
    )
    ax.set_title(
        f"End-to-end latency, live production load ({n} heartbeats, 5 min, "
        "rolling window n=2048)",
        fontsize=8,
        color=INK,
        loc="left",
        pad=8,
    )

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"F5_latency.{ext}", facecolor=SURFACE, bbox_inches="tight")
    print(f"F5 written ({n} heartbeats): {OUT}/F5_latency.[pdf,png]")


if __name__ == "__main__":
    main()
