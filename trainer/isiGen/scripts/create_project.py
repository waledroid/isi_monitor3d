"""Create a new isiGen project from the template.

  python scripts/create_project.py --name pallets_v1 \
      --classes palette:ISI_PLT carton:ISI_CRTN polybag:ISI_PLYBG
"""

from __future__ import annotations

import argparse

import _bootstrap
from src.core.project import ClassSpec, create_project

_PALETTE = [(220, 40, 40), (40, 200, 40), (40, 90, 230),
            (240, 180, 30), (160, 60, 220), (30, 200, 200)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    ap.add_argument("--classes", nargs="+", required=True,
                    metavar="name:TRIGGER",
                    help="e.g. palette:ISI_PLT carton:ISI_CRTN")
    args = ap.parse_args()
    classes = []
    for i, spec in enumerate(args.classes):
        if ":" not in spec:
            raise SystemExit(f"--classes entries are name:TRIGGER (got {spec!r})")
        name, trigger = spec.split(":", 1)
        classes.append(ClassSpec(name=name, trigger=trigger,
                                 color=list(_PALETTE[i % len(_PALETTE)])))
    path = create_project(_bootstrap.DATA_DIR, args.name, classes)
    print(f"created project at {path}")


if __name__ == "__main__":
    main()
