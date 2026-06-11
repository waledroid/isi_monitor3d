"""Launch isiGen Studio (FastAPI on :8200; override with ISIGEN_PORT etc.)."""

from __future__ import annotations

import _bootstrap  # noqa: F401
from src.studio.main import main

if __name__ == "__main__":
    main()
