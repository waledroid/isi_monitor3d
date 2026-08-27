"""F2 — the dual-method geometric core.

Top row: the always-on per-frame homography chain ending in Track2D.
Bottom row: the subscription-driven triangulation branch ending in Track3D,
inheriting the SAME track identity. Gates carry an accent outline; the two
outputs are dark chips. One calibration feeds both methods (annotation).

Run:  conda activate monitor3d && python thesis/figures/src/f2_pipeline.py
Out:  thesis/figures/F2_pipeline.{pdf,png}
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


def stage(ax, x, y, w, h, text, gate=False, chip=False):
    fc = ACCENT_DARK if chip else (ACCENT_FILL if gate else BOX_FILL)
    ec = ACCENT_DARK if chip else (ACCENT if gate else BOX_EDGE)
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.008,rounding_size=0.010",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.2 if gate or chip else 1.0,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=6.6,
        color="white" if chip else INK,
        weight="bold" if chip else "normal",
        linespacing=1.25,
    )
    return x + w  # right edge


def arr(ax, xy_a, xy_b, color=INK_MUTED, curve=0.0, ls="solid", lw=1.1):
    ax.add_patch(
        FancyArrowPatch(
            xy_a,
            xy_b,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=lw,
            color=color,
            linestyle=ls,
            shrinkA=1.5,
            shrinkB=1.5,
            connectionstyle=f"arc3,rad={curve}",
        )
    )


def main() -> None:
    fig, ax = plt.subplots(figsize=(6.9, 2.9), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ---- top row: homography chain (always on, per frame) ------------------
    Y1, H1 = 0.60, 0.17
    ax.text(0.01, 0.885, "HOMOGRAPHY: always on, every frame, every detection", fontsize=7,
            color=INK_MUTED, weight="bold")
    xs = 0.01
    w_gap = 0.018
    tops = [
        ("foot point\n(per detection)", 0.105, False),
        ("undistort\n(K, D)", 0.085, False),
        ("ground-plane\nhomography H", 0.105, False),
        ("cross-camera\nfusion", 0.095, False),
        ("disagreement\ngate", 0.095, True),
        ("ByteTrack\nin meters", 0.09, False),
        ("temporal\nstabilizer", 0.09, False),
    ]
    edges = []
    for text, w, gate in tops:
        edges.append((xs, xs + w))
        stage(ax, xs, Y1, w, H1, text, gate=gate)
        xs += w + w_gap
    chip2d_x = xs
    stage(ax, xs, Y1 + 0.02, 0.075, H1 - 0.04, "Track2D", chip=True)
    for (a0, a1), (b0, _b1) in zip(edges, edges[1:] + [(xs, xs)]):
        arr(ax, (a1 + 0.002, Y1 + H1 / 2), (b0 - 0.002, Y1 + H1 / 2))

    # meters annotation above the H -> fusion hop
    ax.text(0.30, Y1 + H1 + 0.045, "(X, Y) in meters", fontsize=6.2, color=INK_MUTED, ha="center")

    # ---- bottom row: triangulation branch (on demand) ----------------------
    Y2, H2 = 0.13, 0.17
    ax.text(0.01, 0.415, "TRIANGULATION: on demand, only for subscribed tracks (Mode 2)",
            fontsize=7, color=INK_MUTED, weight="bold")
    xs2 = 0.115
    bots = [
        ("subscription\nmatch", 0.095, False),
        ("keypoint\nassociation", 0.095, False),
        ("two-view DLT\n(P_a, P_b)", 0.10, False),
        ("reprojection\ngate ≤ 5 px", 0.10, True),
        ("3-D Kalman\nfilter", 0.09, False),
    ]
    edges2 = []
    for text, w, gate in bots:
        edges2.append((xs2, xs2 + w))
        stage(ax, xs2, Y2, w, H2, text, gate=gate)
        xs2 += w + w_gap
    stage(ax, xs2, Y2 + 0.02, 0.075, H2 - 0.04, "Track3D", chip=True)
    for (a0, a1), (b0, _b1) in zip(edges2, edges2[1:] + [(xs2, xs2)]):
        arr(ax, (a1 + 0.002, Y2 + H2 / 2), (b0 - 0.002, Y2 + H2 / 2))

    # identity inheritance: connector stubs (exit port under Track2D, entry
    # port above subscription match) — avoids sweeping an arc across the rows
    cx = chip2d_x + 0.037
    arr(ax, (cx, Y1 + 0.02), (cx, Y1 - 0.075), color=ACCENT, lw=1.3)
    ax.text(cx + 0.012, Y1 - 0.045, "track_id", fontsize=6.4, color=ACCENT, ha="left",
            style="italic")
    arr(ax, (0.163, Y2 + H2 + 0.095), (0.163, Y2 + H2), color=ACCENT, lw=1.3)
    ax.text(0.178, Y2 + H2 + 0.075, "track_id (from Track2D)", fontsize=6.4, color=ACCENT,
            ha="left", style="italic")
    ax.text(0.5, 0.025, "identity is inherited from the 2-D tracker and never re-assigned",
            fontsize=6.4, color=ACCENT, ha="center", style="italic")

    # one calibration feeds both
    ax.add_patch(
        FancyBboxPatch((0.865, 0.36), 0.115, 0.14,
                       boxstyle="round,pad=0.008,rounding_size=0.010",
                       facecolor=BOX_FILL, edgecolor=BOX_EDGE, linestyle=(0, (4, 2)), linewidth=1.0)
    )
    ax.text(0.9225, 0.43, "calibration.json\nK, D, R, t → H, P", ha="center", va="center",
            fontsize=6.2, color=INK_MUTED, linespacing=1.3)
    arr(ax, (0.90, 0.50), (0.32, Y1 - 0.005), color=INK_MUTED, ls=(0, (3, 2)), curve=-0.12)
    arr(ax, (0.90, 0.36), (0.45, Y2 + H2 + 0.005), color=INK_MUTED, ls=(0, (3, 2)), curve=0.10)
    ax.text(0.9225, 0.325, "one calibration,\ntwo queries", fontsize=6.2, color=INK_MUTED,
            ha="center", va="top", linespacing=1.3)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"F2_pipeline.{ext}", facecolor=SURFACE, bbox_inches="tight")
    print(f"F2 written: {OUT}/F2_pipeline.[pdf,png]")


if __name__ == "__main__":
    main()
