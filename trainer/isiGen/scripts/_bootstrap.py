"""Shared CLI bootstrap: put the isiGen root on sys.path and resolve projects.

Every scripts/run_*.py starts with `import _bootstrap` (same convention as
isidet's run_train.py sys.path insert) so `src.*` imports work no matter the
caller's CWD.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ISIGEN_ROOT = Path(__file__).resolve().parents[1]
if str(ISIGEN_ROOT) not in sys.path:
    sys.path.insert(0, str(ISIGEN_ROOT))

DATA_DIR = ISIGEN_ROOT / "data"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s: %(message)s")


def project_dir(name: str) -> Path:
    d = DATA_DIR / name
    if not (d / "project.yaml").exists():
        raise SystemExit(
            f"project {name!r} not found under {DATA_DIR} — create it first:\n"
            f"  python scripts/create_project.py --name {name} ..."
        )
    return d
