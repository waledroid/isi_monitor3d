"""Generate placeholder figure files for images the author will supply later.

Each placeholder is a light panel carrying the figure id, the intended content,
and the file the author should drop in to replace it. The LaTeX build therefore
works today and the swap later is a file copy, no source edit.

Run:  conda activate monitor3d && python thesis/figures/src/make_placeholders.py
Out:  thesis/figures/<ID>.{pdf,png}  (+ thesis/latex/figures/ copies)
      The real image later overwrites the same filename; the LaTeX never changes.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parents[1]
LATEX = OUT.parent / "latex" / "figures"

INK = "#1a1a18"
INK_MUTED = "#6b6b66"
SURFACE = "#fcfcfb"
BOX_FILL = "#f2f1ee"
BOX_EDGE = "#c3c2bd"
ACCENT = "#2a78d6"

# (id, aspect, what the author should supply)
SLOTS = [
    ("F9_conveyor_scene", 2.4,
     "Photograph of the conveyor installation: camera position over the belt,\n"
     "parcels (carton and polybag) in transit, the counting line as the operator sees it."),
    ("F10_webapp_live", 2.0,
     "Screenshot of the live web platform: annotated video with masks, the counting line,\n"
     "per-class totals, and the sorter output indicators."),
    ("F11_plc_receiver", 2.0,
     "Screenshot of the customer-side receiver application logging live datagrams\n"
     "(class, tracker id, timestamp) arriving from the site PC on the sorter network."),
    ("F12_site_network", 2.2,
     "Site network and integration path: camera subnet, the site PC, the sorter\n"
     "controller subnet, and the wired relay line, as installed."),
]


def placeholder(slot_id: str, aspect: float, description: str) -> None:
    w = 7.1
    fig, ax = plt.subplots(figsize=(w, w / aspect))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor(SURFACE)

    ax.add_patch(
        FancyBboxPatch(
            (0.015, 0.03), 0.97, 0.94,
            boxstyle="round,pad=0.004,rounding_size=0.012",
            facecolor=BOX_FILL, edgecolor=BOX_EDGE,
            linewidth=1.2, linestyle=(0, (6, 4)),
        )
    )
    ax.text(0.5, 0.74, "FIGURE PLACEHOLDER", fontsize=10, weight="bold",
            color=ACCENT, ha="center", va="center")
    ax.text(0.5, 0.58, slot_id, fontsize=12, weight="bold", color=INK,
            ha="center", va="center", family="monospace")
    ax.text(0.5, 0.40, description, fontsize=7.6, color=INK,
            ha="center", va="center", linespacing=1.6)
    ax.text(0.5, 0.145,
            f"to replace: overwrite  thesis/figures/{slot_id}.pdf\n"
            "with the real image (same filename, no LaTeX edit needed)",
            fontsize=6.6, color=INK_MUTED, ha="center", va="center", linespacing=1.5)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = OUT / f"{slot_id}.{ext}"
        fig.savefig(path, facecolor=SURFACE, bbox_inches="tight", dpi=200)
        if ext == "pdf":
            shutil.copy(path, LATEX / path.name)
    plt.close(fig)
    print(f"placeholder written: {slot_id}")


def main() -> None:
    LATEX.mkdir(parents=True, exist_ok=True)
    for slot_id, aspect, description in SLOTS:
        placeholder(slot_id, aspect, description)


if __name__ == "__main__":
    main()
