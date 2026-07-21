"""F3 + F4 — photographic montages from real project data.

F3: the deployed c1 calibration session — per camera (rows), the three capture
phases (columns): ChArUco intrinsics, multi-AprilGrid extrinsics (same
instant both cameras), ChArUco floor anchor. RMS numbers live in the caption,
not the image.

F4: the isiGen pipeline visually — source real photo + the control maps
derived from it (depth, canny, mask), then four synthetic generations.

Run:  conda activate monitor3d && python thesis/figures/src/f3_f4_montages.py
Out:  thesis/figures/F3_calibration.{pdf,png}, F4_isigen.{pdf,png}
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt

REPO = Path("/home/aatanda/isi_monitor3d")
C1 = REPO / "isical" / "data" / "c1"
GEN = REPO / "trainer" / "isiGen" / "data" / "black_polybag"
OUT = REPO / "thesis" / "figures"

INK = "#1a1a18"
INK_MUTED = "#6b6b66"
SURFACE = "#fcfcfb"


def imread_rgb(p: Path, crop_osd: bool = False):
    img = cv2.imread(str(p))
    if img is None:
        raise FileNotFoundError(p)
    if crop_osd:  # strip the camera OSD banner (stale pre-NTP timestamps)
        h = img.shape[0]
        img = img[int(h * 0.075) : int(h * 0.97), :]
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def first(d: Path, pattern: str = "*"):
    files = sorted(q for q in d.glob(pattern) if q.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not files:
        raise FileNotFoundError(f"no images under {d}")
    return files[0]


def f3() -> None:
    cells = [
        ("ChArUco — intrinsics", first(C1 / "intrinsic" / "cam_a"), first(C1 / "intrinsic" / "cam_b")),
        ("AprilGrids — extrinsics (same instant)", C1 / "extrinsic" / "cam_a" / "cam_a_000.jpg",
         C1 / "extrinsic" / "cam_b" / "cam_b_000.jpg"),
        ("ChArUco — floor anchor", first(C1 / "floor" / "cam_a"), first(C1 / "floor" / "cam_b")),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(6.9, 2.9), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    for col, (title, pa, pb) in enumerate(cells):
        for row, p in enumerate((pa, pb)):
            ax = axes[row][col]
            ax.imshow(imread_rgb(p, crop_osd=True))
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("#b9b8b3")
                s.set_linewidth(0.6)
        axes[0][col].set_title(title, fontsize=6.8, color=INK, pad=4)
    axes[0][0].set_ylabel("cam_a", fontsize=7, color=INK_MUTED)
    axes[1][0].set_ylabel("cam_b", fontsize=7, color=INK_MUTED)
    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"F3_calibration.{ext}", facecolor=SURFACE, bbox_inches="tight")
    print("F3 written")


def f4() -> None:
    raw = first(GEN / "raw" / "polybag")
    row1 = [
        ("real source photo", raw),
        ("depth control map", first(GEN / "maps" / "depth")),
        ("canny control map", first(GEN / "maps" / "canny")),
        ("instance mask", first(GEN / "maps" / "mask")),
    ]
    syn = sorted((GEN / "generated").glob("syn*.png"))[:4]
    row2 = [(f"synthetic #{i + 1}", p) for i, p in enumerate(syn)]

    fig, axes = plt.subplots(2, 4, figsize=(6.9, 3.4), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    for col, (title, p) in enumerate(row1):
        ax = axes[0][col]
        ax.imshow(imread_rgb(p))
        ax.set_title(title, fontsize=6.6, color=INK, pad=3)
    for col, (title, p) in enumerate(row2):
        ax = axes[1][col]
        ax.imshow(imread_rgb(p))
        ax.set_title(title, fontsize=6.6, color=INK_MUTED, pad=3)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#b9b8b3")
            s.set_linewidth(0.6)
    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"F4_isigen.{ext}", facecolor=SURFACE, bbox_inches="tight")
    print("F4 written")


if __name__ == "__main__":
    f3()
    f4()
