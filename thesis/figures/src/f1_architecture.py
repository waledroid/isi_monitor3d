"""F1 — system architecture: five modules, two frozen wire contracts.

Lanes: offline artifact production | runtime node (one PC, one GPU) | delivery.
Delivery is a vertical chain (broker -> gateway -> consumers). Titles anchor to
box tops so multi-line subtitles never collide. Neutral surfaces + one blue
accent family; text in ink tokens.

Run:  conda activate monitor3d && python thesis/figures/src/f1_architecture.py
Out:  thesis/figures/F1_architecture.{pdf,png}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1]

INK = "#1a1a18"
INK_MUTED = "#6b6b66"
SURFACE = "#fcfcfb"
BOX_FILL = "#f0efec"
BOX_EDGE = "#b9b8b3"
ACCENT = "#2a78d6"
ACCENT_DARK = "#0d366b"
ACCENT_FILL = "#e3eefb"


def box(ax, x, y, w, h, title, sub=None, accent=False, dashed=False):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.010,rounding_size=0.012",
            facecolor=ACCENT_FILL if accent else BOX_FILL,
            edgecolor=ACCENT if accent else BOX_EDGE,
            linewidth=1.1,
            linestyle=(0, (4, 2)) if dashed else "solid",
        )
    )
    ax.text(
        x + w / 2, y + h - 0.035, title, ha="center", va="center", fontsize=8, color=INK, weight="bold"
    )
    if sub:
        ax.text(
            x + w / 2,
            y + (h - 0.07) / 2,
            sub,
            ha="center",
            va="center",
            fontsize=6.5,
            color=INK_MUTED,
            linespacing=1.35,
        )


def arrow(ax, xy_a, xy_b, color=ACCENT, ls="solid", curve=0.0):
    ax.add_patch(
        FancyArrowPatch(
            xy_a,
            xy_b,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=1.2,
            color=color,
            linestyle=ls,
            shrinkA=2,
            shrinkB=2,
            connectionstyle=f"arc3,rad={curve}",
        )
    )


def label(ax, x, y, text, ha="center", va="center", fs=6.5, color=INK_MUTED):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, color=color, linespacing=1.3)


def main() -> None:
    fig, ax = plt.subplots(figsize=(6.9, 3.7), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # lane headers + separators
    label(ax, 0.125, 0.965, "OFFLINE — artifact production", fs=7, color=INK_MUTED)
    label(ax, 0.485, 0.965, "RUNTIME NODE (one PC, one GPU)", fs=7, color=INK_MUTED)
    label(ax, 0.865, 0.965, "DELIVERY", fs=7, color=INK_MUTED)
    for x in (0.255, 0.735):
        ax.plot([x, x], [0.05, 0.93], color=BOX_EDGE, lw=0.7, ls=(0, (1, 3)))

    # ---- offline lane ------------------------------------------------------
    box(ax, 0.015, 0.66, 0.22, 0.21, "isical", "calibration studio\nChArUco + AprilGrid\n→ bundle adjustment")
    box(ax, 0.015, 0.20, 0.22, 0.21, "isiGen + isidet", "synthetic data\n(SDXL + ControlNet + LoRA)\n→ detector training")
    box(ax, 0.030, 0.505, 0.19, 0.075, "calibration.json", dashed=True)
    box(ax, 0.030, 0.045, 0.19, 0.075, "model.onnx", dashed=True)
    arrow(ax, (0.125, 0.66), (0.125, 0.585), color=INK_MUTED)
    arrow(ax, (0.125, 0.20), (0.125, 0.125), color=INK_MUTED)

    # ---- runtime lane ------------------------------------------------------
    box(ax, 0.29, 0.62, 0.16, 0.21, "cameras", "2× RTSP\nH.264 / H.265")
    box(ax, 0.55, 0.62, 0.165, 0.21, "monitor_web", "operator dashboard\nSTART / STOP\nsupervision")
    box(ax, 0.29, 0.09, 0.185, 0.27, "isistream", "perception producer\ncapture · NVDEC\nzone-scoped detection\npose", accent=True)
    box(ax, 0.53, 0.09, 0.185, 0.27, "backbone", "metric engine\nhomography · triangulation\nByteTrack-in-meters\ngating", accent=True)

    arrow(ax, (0.37, 0.62), (0.3825, 0.365), color=INK_MUTED)  # cameras -> isistream
    label(ax, 0.368, 0.43, "RTSP", ha="right")

    arrow(ax, (0.4795, 0.16), (0.5265, 0.16))  # detection sets
    label(ax, 0.503, 0.135, "UDP :9010", va="top", fs=6.0)
    label(ax, 0.503, 0.075, "detection sets", va="top", fs=6.2)

    arrow(ax, (0.44, 0.365), (0.60, 0.615), ls=(0, (4, 2)), curve=-0.15)  # frame bus
    label(ax, 0.475, 0.545, "frames\n/dev/shm bus", ha="right")

    arrow(ax, (0.655, 0.365), (0.655, 0.615))  # tracks -> dashboard
    label(ax, 0.665, 0.49, "tracks · zones\nUDP :9001", ha="left")

    # artifacts feed the runtime (dashed gray; calibration lands between the
    # two consumers, model.onnx into the producer)
    arrow(ax, (0.225, 0.52), (0.575, 0.368), color=INK_MUTED, ls=(0, (4, 2)), curve=-0.10)
    arrow(ax, (0.225, 0.082), (0.29, 0.13), color=INK_MUTED, ls=(0, (4, 2)))

    # ---- delivery lane (vertical chain, bottom-up) -------------------------
    box(ax, 0.775, 0.09, 0.19, 0.15, "MQTT broker", "Mosquitto :1883")
    box(ax, 0.775, 0.40, 0.19, 0.16, "isicomms gateway", "aggregation\nREST :8080 · probe UI")
    box(ax, 0.775, 0.71, 0.19, 0.15, "consumers", "AGVs · dashboards", dashed=True)
    arrow(ax, (0.718, 0.25), (0.775, 0.20))  # backbone -> broker
    label(ax, 0.749, 0.27, "MQTT\nschema v6", va="bottom", fs=6.0)
    arrow(ax, (0.87, 0.245), (0.87, 0.395), color=ACCENT)
    label(ax, 0.882, 0.32, "subscribe", ha="left", fs=6.2)
    arrow(ax, (0.87, 0.565), (0.87, 0.705), color=ACCENT_DARK)
    label(ax, 0.882, 0.635, "REST poll", ha="left", fs=6.2)

    label(
        ax,
        0.5,
        0.012,
        "solid blue = live data plane (wire contracts) · dashed = file artifacts / shared memory · gray = inputs",
        fs=6.3,
    )

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"F1_architecture.{ext}", facecolor=SURFACE, bbox_inches="tight")
    print(f"F1 written: {OUT}/F1_architecture.[pdf,png]")


if __name__ == "__main__":
    main()
