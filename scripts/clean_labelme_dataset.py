#!/usr/bin/env python3
"""Clean a flat LabelMe dataset so it's perfectly paired.

Every image must have a sibling ``.json`` and every ``.json`` an image. This:
  • deletes orphan IMAGES (no matching .json),
  • deletes orphan JSONs (no matching image),
  • deletes Windows ``:Zone.Identifier`` cruft.

The ``dataset.json`` summary (and any extra ``--keep`` files) is preserved.
Run after a round of manual annotation/pruning in X-AnyLabeling.

    python scripts/clean_labelme_dataset.py <dir> --dry-run    # preview
    python scripts/clean_labelme_dataset.py <dir>              # apply
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path, help="flat LabelMe folder (image+json pairs)")
    ap.add_argument("--dry-run", action="store_true", help="report what would be deleted, change nothing")
    ap.add_argument("--keep", nargs="*", default=["dataset.json"],
                    help="filenames to never delete (default: dataset.json)")
    ap.add_argument("--max-shapes", type=int, default=None,
                    help="also drop image+json pairs whose JSON has MORE than N shapes "
                         "(e.g. --max-shapes 7 drops dense ≥8-instance stock-stack images)")
    args = ap.parse_args()

    d: Path = args.dataset
    if not d.is_dir():
        print(f"error: not a directory: {d}", file=sys.stderr)
        return 2
    keep = set(args.keep)

    files = [p for p in d.iterdir() if p.is_file()]
    junk = [p for p in files if p.name.endswith("Zone.Identifier")]
    imgs = {p.stem: p for p in files
            if p.suffix.lower() in IMG_EXTS and p.name not in keep
            and not p.name.endswith("Zone.Identifier")}
    jsons = {p.stem: p for p in files
             if p.suffix.lower() == ".json" and p.name not in keep
             and not p.name.endswith("Zone.Identifier")}

    orphan_imgs = [p for s, p in imgs.items() if s not in jsons]
    orphan_jsons = [p for s, p in jsons.items() if s not in imgs]
    to_delete = junk + orphan_imgs + orphan_jsons

    # Optional: drop dense pairs (json shape count > max-shapes) + their image.
    dense = 0
    if args.max_shapes is not None:
        import json as _json
        for stem in set(imgs) & set(jsons):
            try:
                n = len(_json.loads(jsons[stem].read_text()).get("shapes", []))
            except (OSError, ValueError):
                continue
            if n > args.max_shapes:
                to_delete += [imgs[stem], jsons[stem]]
                dense += 1

    paired = len(set(imgs) & set(jsons)) - dense
    print(f"dataset: {d}")
    print(f"  paired image+json (keep): {paired}")
    print(f"  orphan images  (no json):  {len(orphan_imgs)}")
    print(f"  orphan jsons   (no image): {len(orphan_jsons)}")
    if args.max_shapes is not None:
        print(f"  dense pairs (>{args.max_shapes} shapes): {dense}")
    print(f"  Zone.Identifier junk:      {len(junk)}")
    for p in (orphan_imgs + orphan_jsons)[:10]:
        print(f"    - {p.name}")
    if len(to_delete) > 10:
        print(f"    … (+{len(to_delete) - 10} more)")

    if args.dry_run:
        print(f"\nDRY RUN — would delete {len(to_delete)} file(s). Re-run without --dry-run to apply.")
        return 0
    for p in to_delete:
        try:
            p.unlink()
        except OSError as exc:
            print(f"  warn: could not delete {p.name}: {exc}", file=sys.stderr)
    print(f"\ndeleted {len(to_delete)} file(s). {paired} paired image+json remain.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
