#!/usr/bin/env python3
"""Migrate a zones.yaml to the new zone-category vocabulary.

Renames the old kinds/types to the current set:

    generic -> palette
    storage -> etagere   ("étagère")
    rack    -> palette   (the "rack" category was removed)
    danger  -> danger    (unchanged)

Both ``kind`` and ``type`` fields are remapped; unrecognized values are left
untouched. A timestamped ``.bak`` backup is written before any change.

Usage:
    python scripts/migrate_zone_kinds.py path/to/zones.yaml [more.yaml ...]
    python scripts/migrate_zone_kinds.py --dry-run path/to/zones.yaml
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import yaml

MAPPING = {"generic": "palette", "storage": "etagere", "rack": "palette"}


def migrate_file(path: Path, dry_run: bool) -> int:
    """Remap kinds/types in one zones.yaml. Returns the number of fields changed."""
    if not path.is_file():
        print(f"  ! not found: {path}")
        return 0
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError) as exc:
        print(f"  ! unreadable {path}: {exc}")
        return 0
    zones = data.get("zones") if isinstance(data, dict) else None
    if not isinstance(zones, list):
        print(f"  ! no 'zones' list in {path} — skipped")
        return 0

    changes = 0
    for z in zones:
        if not isinstance(z, dict):
            continue
        for field in ("kind", "type"):
            old = z.get(field)
            new = MAPPING.get(old)
            if new and new != old:
                print(f"    {z.get('name', '?'):<20} {field}: {old} -> {new}")
                z[field] = new
                changes += 1

    if changes == 0:
        print(f"  {path}: already up to date (0 changes)")
        return 0
    if dry_run:
        print(f"  {path}: {changes} change(s) — DRY RUN, not written")
        return changes

    bak = path.with_suffix(path.suffix + f".bak-{time.strftime('%Y%m%d_%H%M%S')}")
    shutil.copy2(path, bak)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    print(f"  {path}: {changes} change(s) written (backup: {bak.name})")
    return changes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", type=Path, nargs="+", help="zones.yaml file(s) to migrate")
    ap.add_argument("--dry-run", action="store_true", help="show changes without writing")
    args = ap.parse_args()
    total = 0
    for p in args.paths:
        print(f"Migrating {p}:")
        total += migrate_file(p, args.dry_run)
    print(f"\nDone — {total} field(s) {'would change' if args.dry_run else 'changed'}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
