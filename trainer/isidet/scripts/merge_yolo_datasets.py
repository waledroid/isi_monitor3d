#!/usr/bin/env python3
"""Merge multiple YOLO datasets into one, unifying classes **by name**.

Each source is a YOLO dataset dir with ``data.yaml`` (a ``names`` list) +
``images/{train,val}`` + ``labels/{train,val}``. Labels are detect **or** seg
("``cls x y x y ...``" per line, normalized) — only the leading class id is
touched, coords pass through unchanged.

Classes are unified by NAME against the target ``--classes`` list: each source
class index is remapped to the target index of the same name (case-insensitive).
A source class missing from ``--classes`` is an error (nothing is silently
dropped). Filenames are prefixed with a per-source tag (auto-derived from the
source path, or ``--names``) so ids from different datasets never collide.
Train/val splits are preserved; a flat source (no train/val) goes to train.

Empty label files (background negatives) are carried over with their images.

Output (Ultralytics-ready)::
    <out>/images/{train,val}/<tag>__<stem>.<ext>
    <out>/labels/{train,val}/<tag>__<stem>.txt
    <out>/data.yaml          # nc + names = --classes

Usage:
    python scripts/merge_yolo_datasets.py --out data/dataset_v2 \
        --classes carton polybag \
        ../isiGen/data/dataset_v1/export/yolo_seg \
        ../isiGen/data/black_polybag/export_noclip/yolo_seg
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

import yaml

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
_SKIP = {"export", "export_noclip", "yolo_seg", "yolo", "images", "labels", "data", "."}


def _auto_tag(src: Path) -> str:
    """A short unique tag for a source = the first path part that isn't a generic
    export/format dir (e.g. .../dataset_v1/export/yolo_seg -> 'dataset_v1')."""
    for part in reversed(src.resolve().parts):
        if part.lower() not in _SKIP:
            return part
    return src.resolve().name


def _src_names(src: Path) -> list[str]:
    data = yaml.safe_load((src / "data.yaml").read_text())
    names = data.get("names")
    if isinstance(names, dict):                       # {0: 'a', 1: 'b'}
        names = [names[k] for k in sorted(names)]
    if not names:
        sys.exit(f"❌ {src}/data.yaml has no 'names'")
    return list(names)


def _find_image(img_dir: Path, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _remap_label(text: str, idx_map: dict[int, int]) -> str:
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        src_idx = int(float(parts[0]))
        if src_idx not in idx_map:
            raise KeyError(f"label class id {src_idx} not in data.yaml names")
        out.append(" ".join([str(idx_map[src_idx]), *parts[1:]]))
    return "\n".join(out) + ("\n" if out else "")     # empty file = negative


def merge(srcs: list[Path], out: Path, classes: list[str],
          names: list[str] | None) -> dict:
    target = {c.lower(): i for i, c in enumerate(classes)}
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    summary: dict = {"per_source": {}, "instances": Counter(), "images": Counter()}
    for i, src in enumerate(srcs):
        if not (src / "data.yaml").exists():
            sys.exit(f"❌ {src} has no data.yaml — not a YOLO dataset")
        tag = (names[i] if names and i < len(names) else _auto_tag(src))
        src_names = _src_names(src)
        idx_map = {}
        for si, nm in enumerate(src_names):
            if nm.lower() not in target:
                sys.exit(f"❌ source {src} class {nm!r} is not in --classes {classes}")
            idx_map[si] = target[nm.lower()]

        src_counts: dict = {"images": Counter(), "instances": Counter()}
        # split dirs, or flat (everything -> train)
        split_dirs = [(s, src / "labels" / s, src / "images" / s)
                      for s in ("train", "val")]
        if not (src / "labels" / "train").exists():
            split_dirs = [("train", src / "labels", src / "images")]

        for split, lbl_dir, img_dir in split_dirs:
            if not lbl_dir.exists():
                continue
            for lf in sorted(lbl_dir.glob("*.txt")):
                img = _find_image(img_dir, lf.stem)
                if img is None:
                    print(f"   ⚠️  {src.name}/{split}/{lf.stem}: no image, skipped")
                    continue
                remapped = _remap_label(lf.read_text(), idx_map)
                stem = f"{tag}__{lf.stem}"
                (out / "labels" / split / f"{stem}.txt").write_text(remapped)
                shutil.copy(img, out / "images" / split / f"{stem}{img.suffix.lower()}")
                src_counts["images"][split] += 1
                summary["images"][split] += 1
                for line in remapped.splitlines():
                    if line.strip():
                        ci = int(line.split()[0])
                        src_counts["instances"][classes[ci]] += 1
                        summary["instances"][classes[ci]] += 1
        summary["per_source"][tag] = {
            "images": dict(src_counts["images"]),
            "instances": dict(src_counts["instances"]),
            "name_map": {nm: classes[idx_map[si]] for si, nm in enumerate(src_names)},
        }

    (out / "data.yaml").write_text(yaml.safe_dump(
        {"path": ".", "train": "images/train", "val": "images/val",
         "nc": len(classes), "names": list(classes)}, sort_keys=False))
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, type=Path, help="output dataset dir")
    ap.add_argument("--classes", required=True, nargs="+",
                    help="unified class names in target index order (e.g. carton polybag)")
    ap.add_argument("--names", nargs="+", default=None,
                    help="optional per-source filename tags (else auto from path)")
    ap.add_argument("srcs", nargs="+", type=Path, help="source YOLO dataset dirs")
    args = ap.parse_args(argv)

    srcs = [Path(s) for s in args.srcs]
    for s in srcs:
        if not s.is_dir():
            sys.exit(f"❌ not a directory: {s}")
    print(f"🔗 merging {len(srcs)} dataset(s) → {args.out}  (classes: {args.classes})")
    s = merge(srcs, args.out, args.classes, args.names)
    print("\n📊 per source:")
    for tag, info in s["per_source"].items():
        print(f"   {tag}: images {info['images']} · instances {info['instances']} · "
              f"remap {info['name_map']}")
    print(f"\n✅ {args.out}  |  images {dict(s['images'])} "
          f"(total {sum(s['images'].values())})  |  instances {dict(s['instances'])}")
    print(f"   data.yaml: nc={len(args.classes)} names={args.classes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
