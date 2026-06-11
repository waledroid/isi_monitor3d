"""Lazy disk-cached thumbnails for the Studio galleries."""

from __future__ import annotations

from pathlib import Path

from ..utils.images import make_thumbnail


def thumb_path(project_dir: Path, record_id: str, image_rel: str,
               max_px: int = 256) -> Path:
    """Return the cached thumbnail path, building it on first request."""
    dst = Path(project_dir) / "thumbs" / f"{record_id}.jpg"
    src = Path(project_dir) / image_rel
    if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
        make_thumbnail(src, dst, max_px=max_px)
    return dst
