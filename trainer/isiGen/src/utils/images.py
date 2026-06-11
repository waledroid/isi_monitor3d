"""Image helpers: content hashing, EXIF-stripped resave, thumbnails."""

from __future__ import annotations

import hashlib
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resave_clean(src: Path, dst: Path, *, quality: int = 95) -> tuple[int, int]:
    """Re-save ``src`` at ``dst`` with EXIF orientation APPLIED and all metadata
    DROPPED (PIL save without exif kwarg writes none). Returns (width, height)."""
    from PIL import Image, ImageOps
    with Image.open(src) as img:
        img = ImageOps.exif_transpose(img)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        dst.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst, quality=quality)
        return img.width, img.height


def make_thumbnail(src: Path, dst: Path, max_px: int = 256) -> None:
    import cv2
    img = cv2.imread(str(src))
    if img is None:
        raise ValueError(f"unreadable image: {src}")
    h, w = img.shape[:2]
    s = max_px / float(max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))),
                         interpolation=cv2.INTER_AREA)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
