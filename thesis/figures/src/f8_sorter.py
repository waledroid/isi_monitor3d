"""F8 - System B: from a conveyor frame to one correctly timed sorter event.

Top row: the per-frame path (capture, operator ROI, resize, detector, image-space
tracker). Bottom row: the decision layer that turns a detection stream into exactly
one event per parcel (leading-edge anchor, crossing latch OR line test, dedup gate,
gap-free sequence number) and the two delivery paths (UDP datagram to the sorter
controller, relay pulse on a wire) plus the audit CSV.

Style matches f1/f2/f7.

Run:  conda activate monitor3d && python thesis/figures/src/f8_sorter.py
Out:  thesis/figures/F8_sorter.{pdf,png}
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


def box(ax, x, y, w, h, text, gate=False, chip=False, fs=6.3, dashed=False):
    fc = ACCENT_DARK if chip else (ACCENT_FILL if gate else BOX_FILL)
    ec = ACCENT_DARK if chip else (ACCENT if gate else BOX_EDGE)
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.006,rounding_size=0.010",
            facecolor=fc, edgecolor=ec,
            linewidth=1.2 if gate or chip else 1.0,
            linestyle=(0, (4, 2)) if dashed else "solid",
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color="white" if chip else INK,
            weight="bold" if chip else "normal", linespacing=1.32)
    return x + w


def arr(ax, xy_a, xy_b, color=INK_MUTED, curve=0.0, ls="solid", lw=1.1):
    ax.add_patch(
        FancyArrowPatch(xy_a, xy_b, arrowstyle="-|>", mutation_scale=9,
                        color=color, linewidth=lw, linestyle=ls,
                        connectionstyle=f"arc3,rad={curve}", shrinkA=1.5, shrinkB=1.5)
    )


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.1, 3.05))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor(SURFACE)

    # ---------------------------------------------------------- row 1: per frame
    ax.text(0.008, 0.945, "EVERY FRAME: one camera, image space, no calibration",
            fontsize=6.7, color=ACCENT, weight="bold", ha="left")

    Y1, H1 = 0.735, 0.145
    xs, ws = [], []
    x = 0.010
    for w in (0.130, 0.108, 0.120, 0.128, 0.150, 0.128):
        xs.append(x)
        ws.append(w)
        x += w + 0.030
    labels = [
        "RTSP capture\nnewest frame only",
        "operator ROI\nrectangle",
        "resize to 320 px\n(area filter)",
        "glare filter\n(deployed OFF)",
        "parcel detector\ncarton / polybag",
        "ByteTrack\nin image space",
    ]
    for i, (bx, w, t) in enumerate(zip(xs, ws, labels)):
        box(ax, bx, Y1, w, H1, t, dashed=(i == 3), fs=6.1 if i == 3 else 6.3)
    for i in range(len(xs) - 1):
        arr(ax, (xs[i] + ws[i] + 0.004, Y1 + H1 / 2), (xs[i + 1] - 0.004, Y1 + H1 / 2))

    ax.text(xs[3] + ws[3] / 2, Y1 - 0.042, "measured to cost carton counts",
            fontsize=5.7, color=INK_MUTED, ha="center", style="italic")

    # ---------------------------------------------------------- row 2: decision
    ax.text(0.008, 0.585, "PER PARCEL: exactly one correctly timed event",
            fontsize=6.7, color=ACCENT, weight="bold", ha="left")

    Y2, H2 = 0.365, 0.165
    box(ax, 0.010, Y2, 0.170, H2,
        "leading-edge anchor\nfront of the parcel,\nnot its centre", gate=True, fs=6.1)
    box(ax, 0.205, Y2, 0.205, H2,
        "crossing test\ninstantaneous side flip\nOR latched crossing", gate=True, fs=6.1)
    box(ax, 0.435, Y2, 0.165, H2,
        "dedup gate\none per track id", gate=True, fs=6.1)
    box(ax, 0.625, Y2, 0.150, H2,
        "seq counter\ngap-free by\nconstruction", gate=True, fs=6.1)

    # connector: drop out of the tracker, run left on a rail, enter the decision row
    cx = xs[5] + ws[5] / 2
    rail_y = 0.650
    ax.plot([cx, cx], [Y1 - 0.004, rail_y], color=INK_MUTED, linewidth=1.1, solid_capstyle="round")
    ax.plot([cx, 0.095], [rail_y, rail_y], color=INK_MUTED, linewidth=1.1, solid_capstyle="round")
    arr(ax, (0.095, rail_y), (0.095, Y2 + H2 + 0.004))
    ax.text(cx - 0.02, rail_y + 0.022, "detections with track ids", fontsize=5.9,
            color=INK_MUTED, ha="right", style="italic")
    for a, b in ((0.180, 0.205), (0.410, 0.435), (0.600, 0.625)):
        arr(ax, (a + 0.004, Y2 + H2 / 2), (b - 0.004, Y2 + H2 / 2))

    ax.text(0.312, Y2 - 0.042,
            "the latch survives a dropped frame; both paths feed the same gate",
            fontsize=5.9, color=INK_MUTED, ha="center", style="italic")

    # ---------------------------------------------------------- delivery
    box(ax, 0.800, Y2 + 0.088, 0.192, 0.077,
        "UDP datagram to the\nsorter controller", chip=True, fs=6.0)
    box(ax, 0.800, Y2 - 0.005, 0.192, 0.077,
        "relay pulse on a wire\n(the electrical twin)", chip=True, fs=6.0)
    arr(ax, (0.777, Y2 + H2 / 2), (0.797, Y2 + 0.126), curve=-0.10)
    arr(ax, (0.777, Y2 + H2 / 2), (0.797, Y2 + 0.033), curve=0.10)

    box(ax, 0.435, 0.155, 0.340, 0.105,
        "event log, one row per crossing (timestamp, class, track id, seq)\n"
        "same seq as the wire, so the log reconciles against what was received", fs=6.0)
    arr(ax, (0.700, Y2 - 0.004), (0.640, 0.262))

    ax.text(0.5, 0.055,
            "the sorter acts on the event, so the binding constraint is WHEN it is sent, not only whether the parcel was classified",
            fontsize=6.2, color=ACCENT, ha="center", style="italic")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"F8_sorter.{ext}", facecolor=SURFACE, bbox_inches="tight", dpi=300)
    print(f"F8 written: {OUT}/F8_sorter.[pdf,png]")


if __name__ == "__main__":
    main()
