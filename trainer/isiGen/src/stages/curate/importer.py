"""Phase 1 — curate real images into a project.

Recursive folder ingest with sha256 content dedupe, EXIF-orientation applied +
metadata stripped, class tagging (explicit or subdir-derived). Idempotent —
re-running the same folder adds nothing. Curation *judgment* (exclude, retag)
stays human, in the Studio gallery.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2

from ...core.manifest import Manifest, ManifestRecord
from ...core.project import ProjectConfig
from ...utils.images import IMAGE_EXTS, resave_clean, sha256_file
from .mask_import import import_mask, prepare_source

logger = logging.getLogger(__name__)


def ingest(project_dir: Path, project: ProjectConfig, source: Path, *,
           class_name: str | None = None, auto_class: bool = False) -> dict:
    """Import every image under ``source``. Returns counts {added, skipped, warned}.

    ``class_name``: tag everything with one class. ``auto_class``: each image's
    immediate parent directory name must be a project class. Exactly one of the
    two must be chosen.
    """
    if bool(class_name) == bool(auto_class):
        raise ValueError("choose exactly one of class_name=... or auto_class=True")
    if class_name is not None:
        project.class_by_name(class_name)          # raises on unknown class

    cfg = project.phase("curate")
    min_side = int(cfg.get("min_side", 512))
    quality = int(cfg.get("jpeg_quality", 95))

    manifest = Manifest.load(project_dir)
    known_hashes = {r.sha256 for r in manifest.records.values()}
    added = skipped = warned = masks_imported = 0

    ctx = prepare_source(Path(source))      # load any COCO / YOLO-names once
    files = sorted(p for p in Path(source).rglob("*")
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    for path in files:
        cls = class_name if class_name is not None else path.parent.name
        try:
            project.class_by_name(cls)
        except KeyError:
            logger.warning("curate: %s — parent dir %r is not a project class; skipped",
                           path.name, cls)
            skipped += 1
            continue
        digest = sha256_file(path)
        if digest in known_hashes:
            skipped += 1
            continue
        rec_id = digest[:12]
        rel_image = f"raw/{cls}/{rec_id}.jpg"
        try:
            w, h = resave_clean(path, project_dir / rel_image, quality=quality)
        except Exception as exc:
            logger.warning("curate: %s unreadable (%s); skipped", path.name, exc)
            skipped += 1
            continue
        if min(w, h) < min_side:
            logger.warning("curate: %s is small (%dx%d < min_side %d) — kept, review it",
                           path.name, w, h, min_side)
            warned += 1
        rec = ManifestRecord(
            id=rec_id, sha256=digest, source_path=str(path),
            image=rel_image, class_name=cls, width=w, height=h,
        )
        # Import an existing mask (LabelMe / YOLO / COCO) if one is present for
        # this image — skips SAM2 for this record; promptless ones still go to
        # phase 3.
        try:
            mask_bgr = import_mask(path, project, (h, w), ctx)
        except Exception as exc:
            logger.warning("curate: mask import failed for %s (%s)", path.name, exc)
            mask_bgr = None
        if mask_bgr is not None:
            rel_mask = f"maps/mask/{rec_id}.png"
            (project_dir / rel_mask).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(project_dir / rel_mask), mask_bgr)
            rec.mask = rel_mask
            rec.mask_source = "imported"
            rec.needs_review = False
            masks_imported += 1
        manifest.upsert(rec)
        known_hashes.add(digest)
        added += 1

    manifest.save()
    logger.info("curate: %d added, %d skipped, %d size-warnings, %d masks imported "
                "(%d total records)", added, skipped, warned, masks_imported,
                len(manifest.records))
    return {"added": added, "skipped": skipped, "warned": warned,
            "masks_imported": masks_imported, "total": len(manifest.records)}
