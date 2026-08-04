"""Photometric LIGHTING augmentation for background crops (no labels).

Copies each background crop and writes N variants composing 2-4 random
effects — gamma, contrast/brightness, color temperature, directional
gradient, soft shadow bands, sensor noise — plus horizontal flip at p=0.5.
Geometry is untouched (the platform's position is stable; the crops carry
no labels, so flips are safe). Effects deliberately cover what YOLO's
online HSV jitter does NOT: shadows, gradients, temperature, gamma crush.

Variant names keep the capture-series prefix that make_crop_dataset's
bg_split() hashes: `<series>_0007.jpg` -> `<series>_1007.jpg` (v1),
`<series>_2007.jpg` (v2)... so every variant lands in the SAME train/val
split as its source series.

Usage:
  conda activate monitor3d
  python tools/augment_backgrounds.py \
      --src trainer/isidet/data/grouped_backgrounds \
      --out trainer/isidet/data/grouped_backgrounds_aug \
      [--variants 4] [--seed 0]

Then rebuild the crop dataset pointing --backgrounds at the _aug folder.
"""
from __future__ import annotations

import argparse
import logging
import re
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("augment_backgrounds")


def variant_name(name: str, v: int) -> str:
    """`<series>_0007.jpg` -> v=1 `<series>_1007.jpg` (same bg_split group)."""
    m = re.match(r"^(.*)_(\d+)\.[A-Za-z]+$", name)
    if m:
        stem, num = m.groups()
        return f"{stem}_{v * 1000 + int(num):04d}.jpg"
    return f"{Path(name).stem}_{v * 1000:04d}.jpg"


# ---- effects: (float32 BGR image, rng) -> float32 image ----

def _gamma(img, rng):
    g = rng.uniform(0.55, 1.6)
    return 255.0 * np.power(np.clip(img, 0, 255) / 255.0, g)


def _contrast_brightness(img, rng):
    return img * rng.uniform(0.7, 1.3) + rng.uniform(-25.0, 25.0)


def _color_temperature(img, rng):
    s = rng.uniform(0.02, 0.12) * (1.0 if rng.random() < 0.5 else -1.0)
    out = img.copy()
    out[..., 0] *= 1.0 + s      # blue up = cooler (or down = warmer)
    out[..., 2] *= 1.0 - s
    return out


def _gradient(img, rng):
    h, w = img.shape[:2]
    low = rng.uniform(0.6, 1.0)
    ang = rng.uniform(0.0, 2.0 * np.pi)
    yy, xx = np.mgrid[0:h, 0:w]
    t = xx * np.cos(ang) + yy * np.sin(ang)
    t = (t - t.min()) / max(float(t.max() - t.min()), 1e-9)
    ramp = (low + (1.0 - low) * t).astype(np.float32)
    return img * ramp[..., None]


def _shadows(img, rng):
    h, w = img.shape[:2]
    mask = np.ones((h, w), np.float32)
    for _ in range(int(rng.integers(1, 3))):
        depth = rng.uniform(0.45, 0.75)
        pts = np.stack([rng.uniform(0, w, 4), rng.uniform(0, h, 4)], axis=1)
        m = np.zeros((h, w), np.float32)
        cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1.0)
        m = cv2.GaussianBlur(m, (31, 31), 0)
        mask *= 1.0 - m * (1.0 - depth)
    return img * mask[..., None]


def _noise(img, rng):
    sigma = rng.uniform(2.0, 6.0)
    return img + rng.normal(0.0, sigma, img.shape).astype(np.float32)


_EFFECTS = (_gamma, _contrast_brightness, _color_temperature,
            _gradient, _shadows, _noise)


def augment_image(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Compose 2-4 random effects + hflip p=0.5; uint8 in, uint8 out."""
    out = img.astype(np.float32)
    k = int(rng.integers(2, 5))
    for i in rng.choice(len(_EFFECTS), size=k, replace=False):
        out = _EFFECTS[int(i)](out, rng)
    if rng.random() < 0.5:
        out = out[:, ::-1, :]
    return np.ascontiguousarray(np.clip(out, 0, 255).astype(np.uint8))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Write photometric lighting variants of background "
                    "crops. See module docstring.")
    ap.add_argument("--src", required=True, help="folder of background crops")
    ap.add_argument("--out", required=True, help="output folder (must not exist)")
    ap.add_argument("--variants", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    src, out = Path(args.src), Path(args.out)
    if out.exists():
        logger.error("refusing: %s exists", out)
        return 2
    out.mkdir(parents=True)
    rng = np.random.default_rng(args.seed)

    copied = written = skipped = 0
    files = sorted(p for p in src.glob("*")
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            logger.warning("unreadable, skipped: %s", p)
            skipped += 1
            continue
        shutil.copy2(p, out / p.name)
        copied += 1
        for v in range(1, args.variants + 1):
            aug = augment_image(img, rng)
            cv2.imwrite(str(out / variant_name(p.name, v)), aug,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            written += 1
    logger.info("done: %d original(s) copied, %d variant(s) written, "
                "%d skipped -> %s", copied, written, skipped, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
