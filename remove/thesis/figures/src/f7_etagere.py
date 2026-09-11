"""F7 - the etagere (bin-rack) cell-occupancy path, single camera, image space.

Step 1: one camera view of the rack; the operator clicks four corners and the
grid auto-splits into 3 x 3 cells (each cell adjustable and rotatable).
Step 2: every cell is cropped upright with an 8 % margin and the nine crops
ride one batched inference through the two-class detector.
Step 3: per-cell verdicts are stabilized by a vote window and leave as one
retained message per rack.

Style matches f1/f2 (same palette, same rounded boxes). Squares are drawn
aspect-corrected so cells look square on the page.

Run:  conda activate monitor3d && python thesis/figures/src/f7_etagere.py
Out:  thesis/figures/F7_etagere.{pdf,png}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

OUT = Path(__file__).resolve().parents[1]

INK = "#1a1a18"
INK_MUTED = "#6b6b66"
SURFACE = "#fcfcfb"
BOX_FILL = "#f0efec"
BOX_EDGE = "#b9b8b3"
ACCENT = "#2a78d6"
ACCENT_DARK = "#0d366b"
ACCENT_FILL = "#e3eefb"

FIG_W, FIG_H = 7.1, 3.15
AR = FIG_W / FIG_H  # multiply an x-width by this to get an equal-looking y-height

# which of the nine cells hold a bin in the illustration (row-major, 3 x 3)
OCCUPANCY = [
    [True, True, False],
    [True, False, True],
    [True, True, True],
]


def box(ax, x, y, w, h, text, gate=False, chip=False, fs=6.4):
    fc = ACCENT_DARK if chip else (ACCENT_FILL if gate else BOX_FILL)
    ec = ACCENT_DARK if chip else (ACCENT if gate else BOX_EDGE)
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.006,rounding_size=0.010",
            facecolor=fc, edgecolor=ec,
            linewidth=1.2 if gate or chip else 1.0,
        )
    )
    ax.text(
        x + w / 2, y + h / 2, text, ha="center", va="center",
        fontsize=fs, color="white" if chip else INK,
        weight="bold" if chip else "normal", linespacing=1.35,
    )


def arr(ax, xy_a, xy_b, color=INK_MUTED, curve=0.0, ls="solid", lw=1.1):
    ax.add_patch(
        FancyArrowPatch(
            xy_a, xy_b, arrowstyle="-|>", mutation_scale=9,
            color=color, linewidth=lw, linestyle=ls,
            connectionstyle=f"arc3,rad={curve}", shrinkA=1.5, shrinkB=1.5,
        )
    )


def step_label(ax, x, y, n, text):
    ax.text(x, y, f"{n}", fontsize=6.6, color="white", ha="center", va="center",
            weight="bold", zorder=6,
            bbox=dict(boxstyle="circle,pad=0.28", facecolor=ACCENT_DARK, edgecolor="none"))
    ax.text(x + 0.022, y, text, fontsize=6.5, color=ACCENT, style="italic",
            ha="left", va="center")


def main() -> None:
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor(SURFACE)

    # ================================================================ step 1
    step_label(ax, 0.028, 0.955, 1, "camera view")

    ax.add_patch(Rectangle((0.020, 0.335), 0.250, 0.545, facecolor="#ffffff",
                           edgecolor=BOX_EDGE, linewidth=1.0))
    quad = [(0.055, 0.815), (0.238, 0.782), (0.246, 0.400), (0.043, 0.375)]
    ax.add_patch(Polygon(quad, closed=True, fill=False, edgecolor=ACCENT,
                         linewidth=1.3, linestyle=(0, (4, 2))))
    for (cx, cy) in quad:
        ax.plot([cx], [cy], marker="o", markersize=3.6, color=ACCENT_DARK, zorder=5)

    def q(u, v):
        top = (quad[0][0] + (quad[1][0] - quad[0][0]) * u,
               quad[0][1] + (quad[1][1] - quad[0][1]) * u)
        bot = (quad[3][0] + (quad[2][0] - quad[3][0]) * u,
               quad[3][1] + (quad[2][1] - quad[3][1]) * u)
        return (top[0] + (bot[0] - top[0]) * v, top[1] + (bot[1] - top[1]) * v)

    for k in (1 / 3, 2 / 3):
        ax.plot(*zip(q(k, 0.0), q(k, 1.0)), color=ACCENT, linewidth=0.9, alpha=0.85)
        ax.plot(*zip(q(0.0, k), q(1.0, k)), color=ACCENT, linewidth=0.9, alpha=0.85)
    for r in range(3):
        for c in range(3):
            cx, cy = q((c + 0.5) / 3, (r + 0.5) / 3)
            ax.text(cx, cy, f"r{r+1}c{c+1}", fontsize=5.0, color=INK_MUTED,
                    ha="center", va="center")

    ax.text(0.145, 0.300, "image space, no calibration:\noperator clicks 4 corners, the grid\n"
                          "auto-splits into 3 × 3 cells\n(adjustable, each rotatable)",
            fontsize=6.2, color=INK, ha="center", va="top", linespacing=1.4)

    # ================================================================ step 2
    arr(ax, (0.276, 0.607), (0.316, 0.607))
    step_label(ax, 0.335, 0.955, 2, "crops and inference")

    s = 0.044                 # crop side in x units
    sy = s * AR               # same side in y units
    gap = 0.013
    x0 = 0.330
    y_top = 0.800
    for r in range(3):
        for c in range(3):
            cx = x0 + c * (s + gap)
            cy = y_top - r * (sy + gap * AR) - sy
            ax.add_patch(Rectangle((cx, cy), s, sy, facecolor="#ffffff",
                                   edgecolor=BOX_EDGE, linewidth=0.8))
            if OCCUPANCY[r][c]:
                pad_x, pad_y = 0.008, 0.008 * AR
                ax.add_patch(Rectangle((cx + pad_x, cy + pad_y),
                                       s - 2 * pad_x, sy - 2 * pad_y,
                                       facecolor=ACCENT_FILL, edgecolor=ACCENT, linewidth=0.8))
    grid_w = 3 * s + 2 * gap
    mid2 = x0 + grid_w / 2

    ax.text(mid2, 0.310, "each cell cropped upright with an 8 % margin,\n"
                         "letterboxed to 320 px, batched together",
            fontsize=6.2, color=INK, ha="center", va="top", linespacing=1.4)
    box(ax, x0 - 0.030, 0.075, grid_w + 0.060, 0.105,
        "yolo26n detector, 2 classes\nempty_box / filled_box", chip=True, fs=6.1)
    arr(ax, (mid2, 0.372), (mid2, 0.185))

    # ================================================================ step 3
    arr(ax, (x0 + grid_w + 0.020, 0.607), (0.600, 0.607))
    step_label(ax, 0.610, 0.955, 3, "state and delivery")

    box(ax, 0.600, 0.640, 0.180, 0.215,
        "cell-state stabilizer\nflip needs ≥ 70 % of a\n15-vote window,\nunknown after 5 s",
        gate=True, fs=6.2)

    # resulting matrix, aspect-corrected cells with visible separators
    ms = 0.036
    msy = ms * AR
    mx, my_top = 0.845, 0.845
    for r in range(3):
        for c in range(3):
            cx = mx + c * ms
            cy = my_top - (r + 1) * msy
            filled = OCCUPANCY[r][c]
            ax.add_patch(Rectangle((cx, cy), ms, msy,
                                   facecolor=ACCENT_DARK if filled else "#ffffff",
                                   edgecolor=ACCENT_DARK, linewidth=0.9))
    ax.text(mx + 1.5 * ms, my_top + 0.028, "cell matrix", fontsize=6.2, color=INK, ha="center")
    # legend under the matrix
    lg_y = my_top - 3 * msy - 0.058
    ax.add_patch(Rectangle((mx + 0.004, lg_y), 0.020, 0.020 * AR,
                           facecolor=ACCENT_DARK, edgecolor=ACCENT_DARK, linewidth=0.8))
    ax.text(mx + 0.030, lg_y + 0.020 * AR / 2, "filled", fontsize=5.8, color=INK_MUTED,
            ha="left", va="center")
    ax.add_patch(Rectangle((mx + 0.062, lg_y), 0.020, 0.020 * AR,
                           facecolor="#ffffff", edgecolor=ACCENT_DARK, linewidth=0.8))
    ax.text(mx + 0.088, lg_y + 0.020 * AR / 2, "empty", fontsize=5.8, color=INK_MUTED,
            ha="left", va="center")
    arr(ax, (0.784, 0.745), (0.841, 0.745))

    box(ax, 0.595, 0.360, 0.395, 0.135,
        "one etagere_state message per rack\nUDP + MQTT, topic {prefix}/etagere/{zone_id}, retained",
        chip=True, fs=5.9)
    arr(ax, (0.690, 0.636), (0.690, 0.500))

    box(ax, 0.595, 0.130, 0.395, 0.145,
        "consumers\nwarehouse management system (rack occupancy),\n"
        "isicomms GET /etagere, operator dashboard matrix", fs=6.0)
    arr(ax, (0.792, 0.356), (0.792, 0.281))

    ax.text(0.5, 0.018,
            "the rack path never enters the metric floor pipeline: no homography, no triangulation, one camera per rack",
            fontsize=6.3, color=ACCENT, ha="center", style="italic")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"F7_etagere.{ext}", facecolor=SURFACE, bbox_inches="tight", dpi=300)
    print(f"F7 written: {OUT}/F7_etagere.[pdf,png]")


if __name__ == "__main__":
    main()
