"""Crop every image in a directory to a fixed ROI.

Mirrors the site's inference-time crop so training framing == inference framing
(crop frames to the same ROI you crop to at runtime, then annotate the crops, so
labels live in ROI coordinates). Reusable for any future site footage.

ROI is `x0 x1 y0 y1` (x0/y0 inclusive, x1/y1 exclusive → width = x1-x0), matching
numpy slicing `img[y0:y1, x0:x1]`. Three ways to choose it:

  * **default** — the site ROI `x[736:2221] y[0:1620]` (1485x1620) if nothing given;
  * **explicit** — `--roi 736 2221 0 1620`;
  * **draw** — `--draw` opens a random image from the folder so you drag-draw the
    ROI; on accept (Enter) it crops the whole folder to that box. Since every image
    shares the same dimensions, one drawn box applies to all.

Output goes to a sibling `cropped_<dir>` (override with `-o`).

Usage:
    python scripts/crop_roi.py data/.../img                      # default ROI
    python scripts/crop_roi.py data/.../img --roi 736 2221 0 1620
    python scripts/crop_roi.py data/.../img --draw               # drag-draw the ROI

`--draw` needs a GUI-capable OpenCV + a display ($DISPLAY / WSLg); on a headless
box use `--roi` instead.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2

IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_ROI = (736, 2221, 0, 1620)   # site inference ROI: x[736:2221] y[0:1620]
_DRAW_MAX_W = 1280                    # scale the preview so big frames fit the screen


def list_images(in_dir: Path) -> list[Path]:
    imgs = sorted(p for p in in_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in IMG_SUFFIXES)
    if not imgs:
        sys.exit(f"❌ no images found in {in_dir}")
    return imgs


def draw_roi(sample: Path) -> tuple[int, int, int, int]:
    """Let the user drag-draw an ROI on `sample`; returns (x0, x1, y0, y1) in
    full-resolution coords. The preview is scaled to fit the screen."""
    img = cv2.imread(str(sample))
    if img is None:
        sys.exit(f"❌ could not read sample image {sample}")
    h, w = img.shape[:2]
    scale = min(1.0, _DRAW_MAX_W / w)
    disp = cv2.resize(img, None, fx=scale, fy=scale) if scale < 1.0 else img
    title = "Drag the ROI  —  Enter/Space = accept,  Esc/c = cancel"
    try:
        x, y, bw, bh = cv2.selectROI(title, disp, showCrosshair=False, fromCenter=False)
        cv2.destroyAllWindows()
    except cv2.error as exc:
        sys.exit("❌ --draw needs a GUI-capable OpenCV + a display ($DISPLAY/WSLg). "
                 f"Use --roi instead. ({exc})")
    if bw == 0 or bh == 0:
        sys.exit("ROI selection cancelled — nothing cropped.")
    # map the box back to full-res pixels and clamp to the image
    x0 = max(0, round(x / scale))
    y0 = max(0, round(y / scale))
    x1 = min(w, round((x + bw) / scale))
    y1 = min(h, round((y + bh) / scale))
    return x0, x1, y0, y1


def crop_dir(images: list[Path], roi: tuple[int, int, int, int], out_dir: Path) -> dict:
    x0, x1, y0, y1 = roi
    if not (0 <= x0 < x1 and 0 <= y0 < y1):
        sys.exit(f"❌ invalid ROI {roi}: need 0<=x0<x1 and 0<=y0<y1")
    out_dir.mkdir(parents=True, exist_ok=True)
    cropped = skipped = 0
    out_wh: tuple[int, int] | None = None
    for p in images:
        img = cv2.imread(str(p))
        if img is None:
            print(f"   ⚠️  unreadable, skipped: {p.name}")
            skipped += 1
            continue
        h, w = img.shape[:2]
        if x1 > w or y1 > h:
            print(f"   ⚠️  {p.name} is {w}x{h}, smaller than ROI; skipped")
            skipped += 1
            continue
        cv2.imwrite(str(out_dir / p.name), img[y0:y1, x0:x1])
        out_wh = (x1 - x0, y1 - y0)
        cropped += 1
    return {"cropped": cropped, "skipped": skipped, "out_wh": out_wh}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("in_dir", type=Path, help="directory of images to crop")
    parser.add_argument("--roi", type=int, nargs=4, default=None,
                        metavar=("X0", "X1", "Y0", "Y1"),
                        help=f"ROI bounds x0 x1 y0 y1 (default: {DEFAULT_ROI})")
    parser.add_argument("--draw", action="store_true",
                        help="drag-draw the ROI on a random image from the folder")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="output dir (default: sibling cropped_<dir>)")
    args = parser.parse_args(argv)

    if not args.in_dir.is_dir():
        sys.exit(f"❌ not a directory: {args.in_dir}")
    images = list_images(args.in_dir)

    if args.draw:
        roi = draw_roi(random.choice(images))
        print(f"   drawn ROI: x[{roi[0]}:{roi[1]}] y[{roi[2]}:{roi[3]}]")
    elif args.roi is not None:
        roi = tuple(args.roi)
    else:
        roi = DEFAULT_ROI
    x0, x1, y0, y1 = roi
    out_dir = args.output or args.in_dir.with_name(f"cropped_{args.in_dir.name}")

    print(f"✂️  cropping {len(images)} image(s) in {args.in_dir} → "
          f"ROI x[{x0}:{x1}] y[{y0}:{y1}] ({x1 - x0}x{y1 - y0}) → {out_dir}")
    res = crop_dir(images, roi, out_dir)
    wh = res["out_wh"]
    dims = f"  ({wh[0]}x{wh[1]} px)" if wh else ""
    skipped = f", skipped {res['skipped']}" if res["skipped"] else ""
    print(f"✅ cropped {res['cropped']} image(s){skipped} → {out_dir}{dims}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
