#!/usr/bin/env python3
"""Convert one or more YOLO-SEG datasets to a single LabelMe dataset (polygons),
collapsing every source class into one target class.

Designed for the pallet seg merge: ``video/v1`` (4 noisy classes) + ``video/v2``
(1 class) → ``video/pallet_seg.v1`` with a single ``palette`` class and a fresh
80/20 train/val split (seed 42). Filenames are prefixed by source (``v1_…`` /
``v2_…``) so stems can never collide across the merged set.

LabelMe layout written (flat per split, JSON + image side by side):

    video/pallet_seg.v1/
      train/
        v1_<stem>.jpg
        v1_<stem>.json
        …
      valid/
        …
      dataset.json     ← summary (counts, sources, dropped/orphans)

Each per-image LabelMe JSON:
  - shape_type "polygon", label = the target class (default "palette")
  - points denormalised to pixels using the image's W,H
  - imagePath = the basename (image sits beside the JSON)
  - imageData = null (kept off disk; LabelMe loads the sibling image)
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

try:
    from PIL import Image
except ImportError:                                  # pragma: no cover
    Image = None
import cv2  # always available in monitor3d

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("yolo_seg_to_labelme")

IMG_EXTS = (".jpg", ".jpeg", ".png")
LABELME_VERSION = "5.4.1"


@dataclass
class Sample:
    source: str          # 'v1' | 'v2' | …
    stem: str            # original basename (no ext)
    image_path: Path
    label_path: Path
    src_class_count: Counter  # how many polygons per source class id (diagnostics)


def image_size(path: Path) -> tuple[int, int]:
    """Return (width, height) in pixels."""
    if Image is not None:
        try:
            with Image.open(path) as im:
                return im.size
        except Exception:
            pass
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"cannot read image dimensions: {path}")
    h, w = img.shape[:2]
    return (w, h)


def parse_yolo_line(line: str) -> tuple[int, list[tuple[float, float]]] | None:
    """Parse one YOLO label line (normalised) into (class_id, polygon_points).

    Handles BOTH geometries, so a mixed bbox/seg dataset becomes uniform polygons:
      • bbox  ``class cx cy w h``        → a 4-corner rectangle polygon (TL,TR,BR,BL)
      • seg   ``class x1 y1 … xn yn``    → the polygon as-is (≥3 points)
    Returns None for malformed/degenerate lines.
    """
    toks = line.strip().split()
    if len(toks) < 5:
        return None
    try:
        cls = int(float(toks[0]))
        coords = [float(t) for t in toks[1:]]
    except ValueError:
        return None
    if len(coords) == 4:                              # bbox cx,cy,w,h → rectangle poly
        cx, cy, w, h = coords
        x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        return cls, [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    if len(coords) >= 6 and len(coords) % 2 == 0:     # seg polygon
        points = list(zip(coords[0::2], coords[1::2], strict=False))
        return (cls, points) if len(points) >= 3 else None
    return None


def _under_dir(path: Path, name: str) -> bool:
    return any(p.name == name for p in path.parents)


def collect_samples(root: Path, source: str) -> tuple[list[Sample], list[Path]]:
    """Layout-agnostic scan of a YOLO dataset (any split structure). Pairs every
    ``*.txt`` under a ``labels/`` dir with the image of the same stem under an
    ``images/`` dir. Ignores ``:Zone.Identifier`` cruft, ``classes.txt`` and
    ``README*``. Returns (paired samples, unlabeled images)."""
    labels: dict[str, Path] = {}
    images: dict[str, Path] = {}
    for f in root.rglob("*"):
        if not f.is_file() or f.name.endswith("Zone.Identifier"):
            continue
        ext = f.suffix.lower()
        if ext == ".txt" and _under_dir(f, "labels") and f.name != "classes.txt":
            labels[f.stem] = f
        elif ext in IMG_EXTS and _under_dir(f, "images"):
            images[f.stem] = f

    samples: list[Sample] = []
    orphans = 0
    for stem, lbl in labels.items():
        img = images.get(stem)
        if img is None:
            orphans += 1            # label whose image was pruned away → drop
            continue
        classes_here: Counter = Counter()
        try:
            for line in lbl.read_text().splitlines():
                parsed = parse_yolo_line(line)
                if parsed is not None:
                    classes_here[parsed[0]] += 1
        except OSError:
            orphans += 1
            continue
        samples.append(Sample(source, stem, img, lbl, classes_here))
    unlabeled = [img for stem, img in images.items() if stem not in labels]
    if orphans:
        log.warning("%s: %d orphan label(s) without an image — dropped", root, orphans)
    if unlabeled:
        log.info("%s: %d image(s) without a label", root, len(unlabeled))
    return samples, unlabeled


def image_ahash(path: Path, size: int = 16) -> str | None:
    """Average-hash an image for duplicate detection: decode → gray → resize →
    bits above the mean. Catches re-encoded/resized copies of the same image
    across datasets; distinct images (different pixels) won't collide at 16x16."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    small = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    bits = (small >= small.mean()).flatten()
    return bits.tobytes().hex()


def write_labelme(sample: Sample, out_image: Path, out_json: Path, target_class: str) -> int:
    """Write one image's LabelMe JSON. Returns number of polygons written."""
    shutil.copy2(sample.image_path, out_image)
    w, h = image_size(out_image)
    shapes = []
    for line in sample.label_path.read_text().splitlines():
        parsed = parse_yolo_line(line)
        if parsed is None:
            continue
        _cls, pts_norm = parsed
        pts_px = [
            [max(0.0, min(float(x) * w, w - 1)),
             max(0.0, min(float(y) * h, h - 1))]
            for x, y in pts_norm
        ]
        shapes.append({
            "label": target_class,
            "points": pts_px,
            "group_id": None,
            "description": "",
            "shape_type": "polygon",
            "flags": {},
            "mask": None,
        })
    doc = {
        "version": LABELME_VERSION,
        "flags": {},
        "shapes": shapes,
        "imagePath": out_image.name,
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w,
    }
    out_json.write_text(json.dumps(doc, ensure_ascii=False, indent=2))
    return len(shapes)


def _slug(name: str) -> str:
    """Short filesystem-safe source tag from a folder name (for filename prefixes)."""
    keep = "".join(c if c.isalnum() else "_" for c in name).strip("_").lower()
    while "__" in keep:
        keep = keep.replace("__", "_")
    return keep[:24] or "src"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("archive/video/pallet_seg.v1"))
    ap.add_argument("--class-name", default="palette")
    ap.add_argument("--val-frac", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="remove --out if it exists")
    ap.add_argument("--flat", action="store_true",
                    help="write one flat LabelMe folder (image+json pairs) instead of "
                         "a train/valid split — the format X-AnyLabeling 'Open Dir' wants")
    ap.add_argument("--sources-dir", type=Path, default=None,
                    help="auto-discover every immediate sub-folder of DIR as a source")
    ap.add_argument("--sources", nargs="*", default=None,
                    help="explicit source specs ``tag:root`` (layout-agnostic)")
    ap.add_argument("--append", action="store_true",
                    help="append into an EXISTING flat --out dataset (don't wipe it); "
                         "new images are deduped against the ones already there")
    ap.add_argument("--no-dedup", action="store_true", help="keep duplicate images")
    ap.add_argument("--keep-unlabeled", action="store_true",
                    help="copy images that have NO label into <out>/unlabeled/ (no json) "
                         "so you can annotate them later")
    args = ap.parse_args()

    # Resolve sources: --sources-dir auto-discovers; --sources is explicit tag:root.
    source_roots: list[tuple[str, Path]] = []
    if args.sources_dir:
        for sub in sorted(p for p in args.sources_dir.iterdir() if p.is_dir()):
            source_roots.append((_slug(sub.name), sub))
    for spec in (args.sources or []):
        tag, _, root_s = spec.partition(":")
        source_roots.append((_slug(tag), Path(root_s)))
    if not source_roots:   # back-compat default
        source_roots = [("v1", Path("archive/video/v1")), ("v2", Path("archive/video/v2"))]

    out_root: Path = args.out
    if args.append:
        if not out_root.is_dir():
            print(f"error: --append needs an existing dataset at {out_root}", file=sys.stderr)
            return 2
        args.flat = True   # appending only makes sense into a flat LabelMe folder
    elif out_root.exists():
        if not args.force:
            print(f"error: {out_root} exists — pass --force to overwrite", file=sys.stderr)
            return 2
        shutil.rmtree(out_root)

    samples: list[Sample] = []
    unlabeled_all: list[tuple[str, Path]] = []
    src_class_totals: dict[str, Counter] = {}
    per_source_counts: dict[str, int] = {}
    for tag, root in source_roots:
        if not root.is_dir():
            log.warning("source %s: %s missing — skipped", tag, root)
            continue
        s, unl = collect_samples(root, tag)
        samples.extend(s)
        unlabeled_all.extend((tag, p) for p in unl)
        src_class_totals[tag] = sum((x.src_class_count for x in s), Counter())
        per_source_counts[tag] = len(s)
        log.info("source %s: %d paired samples (src-class hist: %s)",
                 tag, len(s), dict(src_class_totals[tag]))

    if not samples:
        print("no samples found", file=sys.stderr)
        return 1

    # Dedup images across all sources (keep first occurrence). Catches the same
    # image re-exported by several Roboflow versions.
    dropped_dups = 0
    if not args.no_dedup:
        seen: set[str] = set()
        if args.append:   # seed with images already in the dataset we're appending to
            for p in out_root.iterdir():
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    h = image_ahash(p)
                    if h is not None:
                        seen.add(h)
            log.info("append: %d existing image(s) seeded for dedup", len(seen))
        deduped: list[Sample] = []
        for s in samples:
            h = image_ahash(s.image_path)
            if h is not None and h in seen:
                dropped_dups += 1
                continue
            if h is not None:
                seen.add(h)
            deduped.append(s)
        log.info("dedup: dropped %d duplicate image(s) → %d unique", dropped_dups, len(deduped))
        samples = deduped

    random.Random(args.seed).shuffle(samples)

    def _emit(group: list[Sample], split_dir: Path) -> int:
        split_dir.mkdir(parents=True, exist_ok=True)
        total = 0
        for s in group:
            out_name = f"{s.source}_{s.stem}"
            out_img = split_dir / f"{out_name}{s.image_path.suffix.lower()}"
            total += write_labelme(s, out_img, split_dir / f"{out_name}.json", args.class_name)
        return total

    polys: dict[str, int] = {}
    if args.flat:
        polys["all"] = _emit(samples, out_root)
        log.info("flat: wrote %d image+json pairs → %s", len(samples), out_root)
    else:
        n_val = round(len(samples) * args.val_frac)
        polys["valid"] = _emit(samples[:n_val], out_root / "valid")
        polys["train"] = _emit(samples[n_val:], out_root / "train")

    # Unlabeled images (post-dedup) → a side folder to annotate later.
    n_unlabeled = 0
    if args.keep_unlabeled and unlabeled_all:
        unl_dir = out_root / "unlabeled"
        unl_dir.mkdir(parents=True, exist_ok=True)
        seen_u: set[str] = set()
        for tag, p in unlabeled_all:
            h = image_ahash(p)
            if (not args.no_dedup) and h is not None and h in seen_u:
                continue
            if h is not None:
                seen_u.add(h)
            shutil.copy2(p, unl_dir / f"{tag}_{p.name}")
            n_unlabeled += 1
        log.info("kept %d unlabeled image(s) → %s", n_unlabeled, unl_dir)

    summary = {
        "out": str(out_root),
        "class_name": args.class_name,
        "layout": "flat" if args.flat else "train/valid split",
        "sources": {tag: str(root) for tag, root in source_roots},
        "paired_per_source": per_source_counts,
        "total_samples_after_dedup": len(samples),
        "duplicate_images_dropped": dropped_dups,
        "unlabeled_images_kept": n_unlabeled,
        "polygons": polys,
        "source_class_histograms": {k: dict(v) for k, v in src_class_totals.items()},
        "seed": args.seed,
    }
    summary_name = "dataset.append.json" if args.append else "dataset.json"
    (out_root / summary_name).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    log.info("done — %s", json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
