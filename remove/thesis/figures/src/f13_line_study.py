"""F13 - System B: choosing where to put the trigger line, from measured tracks.

Left: how many of the 35 tracks straddle each candidate line position, i.e. how
many are actually seen on both sides of the line and can therefore produce a
crossing event. Right: where the two candidate anchors sit inside the region of
interest (p10 to p90 with the median marked), which is why the deployed position
follows the leading-edge anchor rather than the box centre.

Every number is read from the study outputs `line_result.txt` and
`line_result_le.txt` (3 clips x 1000 frames, 35 tracks, 2609 detections).

Palette: the article's accent blue plus one warm hue; validated with the dataviz
validator (worst adjacent pair dE 26.2 protan, 25.8 tritan, 29.3 normal, all
checks pass on the #fcfcfb surface).

Run:  conda activate monitor3d && python thesis/figures/src/f13_line_study.py
Out:  thesis/figures/F13_line_study.{pdf,png}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parents[1]

INK = "#1a1a18"
INK_MUTED = "#6b6b66"
SURFACE = "#fcfcfb"
GRID = "#e2e1dd"
BLUE = "#2a78d6"
BLUE_DARK = "#0d366b"
WARM = "#b8621b"

# line_result_le.txt: straddle count per candidate line, leading-edge anchor
CANDIDATES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
STRADDLE = [11, 1, 2, 7, 10, 10, 14, 13, 12]
N_TRACKS = 35
BEST_POS, BEST_N = 0.71, 15

# anchor height distribution inside the ROI (p10, median, p90)
ANCHORS = [
    ("leading edge (deployed)", 0.70, 0.71, 0.72, BLUE),
    ("box centre", 0.45, 0.45, 0.46, WARM),
]


def rounded_bar(ax, x, height, width, color, radius=0.018):
    """Thin bar with rounded data-end, anchored on the baseline."""
    ax.add_patch(
        FancyBboxPatch(
            (x - width / 2, 0), width, max(height, 1e-6),
            boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor=color, edgecolor=SURFACE, linewidth=0.8,
            mutation_aspect=0.35, zorder=3,
        )
    )


def main() -> None:
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(7.1, 2.45), gridspec_kw={"width_ratios": [1.05, 1.0], "wspace": 0.22}
    )
    fig.patch.set_facecolor(SURFACE)

    # ------------------------------------------------------------------ left
    axL.set_facecolor(SURFACE)
    for x, n in zip(CANDIDATES, STRADDLE):
        rounded_bar(axL, x, n, 0.062, BLUE if x != 0.7 else BLUE_DARK)

    axL.axvline(BEST_POS, color=WARM, linewidth=1.2, linestyle=(0, (4, 2)), zorder=2)
    axL.annotate(
        f"fitted optimum {BEST_POS:.2f}\n{BEST_N} of {N_TRACKS} tracks",
        xy=(BEST_POS, BEST_N), xytext=(0.52, 15.6),
        fontsize=6.2, color=WARM, ha="center", va="bottom", linespacing=1.35,
    )
    axL.text(0.7, 14.6, "14", fontsize=6.2, color=BLUE_DARK, ha="center", va="bottom", weight="bold")

    axL.set_xlim(0.03, 0.97)
    axL.set_ylim(0, 18.5)
    axL.set_xticks(CANDIDATES)
    axL.set_xticklabels([f"{c:.1f}" for c in CANDIDATES], fontsize=6.2, color=INK_MUTED)
    axL.set_yticks([0, 5, 10, 15])
    axL.set_yticklabels(["0", "5", "10", "15"], fontsize=6.2, color=INK_MUTED)
    axL.set_xlabel("candidate line position (fraction of region height)", fontsize=6.5, color=INK)
    axL.set_ylabel(f"tracks straddling the line (of {N_TRACKS})", fontsize=6.5, color=INK)
    axL.set_title("a. how many tracks can produce a crossing", fontsize=6.9, color=INK,
                  loc="left", pad=6)
    axL.yaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
    axL.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axL.spines[side].set_visible(False)
    axL.spines["bottom"].set_color(GRID)
    axL.tick_params(length=0)

    # ----------------------------------------------------------------- right
    axR.set_facecolor(SURFACE)
    for i, (label, p10, med, p90, color) in enumerate(ANCHORS):
        y = 1 - i
        axR.plot([p10, p90], [y, y], color=color, linewidth=7.0,
                 solid_capstyle="round", alpha=0.40, zorder=2)
        axR.plot([med], [y], marker="o", markersize=7.0, color=color,
                 markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=4)
        axR.text(0.408, y + 0.26, label, fontsize=6.4, color=INK, ha="left")
        if i == 0:   # leading edge sits near the right edge, label to its left
            axR.text(p10 - 0.016, y, f"p10 {p10:.2f} to p90 {p90:.2f}, median {med:.2f}",
                     fontsize=5.9, color=INK_MUTED, ha="right", va="center")
        else:
            axR.text(p90 + 0.016, y, f"p10 {p10:.2f} to p90 {p90:.2f}, median {med:.2f}",
                     fontsize=5.9, color=INK_MUTED, ha="left", va="center")

    axR.axvline(BEST_POS, color=WARM, linewidth=1.2, linestyle=(0, (4, 2)), zorder=1)
    axR.text(BEST_POS - 0.014, -0.62, "deployed line 0.71", fontsize=6.1, color=WARM,
             ha="right", va="center")

    axR.annotate("", xy=(0.45, 0.50), xytext=(0.71, 0.50),
                 arrowprops=dict(arrowstyle="<->", color=INK_MUTED, linewidth=0.9))
    axR.text(0.58, 0.58, "the anchor choice moves the\nline by 0.26 of the region height",
             fontsize=5.9, color=INK_MUTED, ha="center", va="bottom", linespacing=1.3)

    axR.set_xlim(0.36, 0.90)
    axR.set_ylim(-0.95, 1.75)
    axR.set_yticks([])
    axR.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8])
    axR.set_xticklabels(["0.4", "0.5", "0.6", "0.7", "0.8"], fontsize=6.2, color=INK_MUTED)
    axR.set_xlabel("anchor height inside the region (fraction)", fontsize=6.5, color=INK)
    axR.set_title("b. where each candidate anchor sits", fontsize=6.9, color=INK, loc="left", pad=6)
    axR.xaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
    axR.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axR.spines[side].set_visible(False)
    axR.spines["bottom"].set_color(GRID)
    axR.tick_params(length=0)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"F13_line_study.{ext}", facecolor=SURFACE, bbox_inches="tight", dpi=300)
    print(f"F13 written: {OUT}/F13_line_study.[pdf,png]")


if __name__ == "__main__":
    main()
