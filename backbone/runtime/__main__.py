"""CLI entry point for the Backbone runtime.

Run with:

    python -m backbone.runtime --config config/backbone.yaml

Using the package's ``__main__`` (rather than ``python -m
backbone.runtime.orchestrator``) avoids the ``runpy`` double-import
RuntimeWarning — the orchestrator module is imported once by the package, never
also executed as ``__main__``.

Builds the :class:`~backbone.runtime.orchestrator.Orchestrator` from the YAML,
installs SIGINT/SIGTERM handlers, and runs until stopped (the dashboard's
START/STOP sends those signals).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .orchestrator import Orchestrator


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="backbone.runtime",
        description="Run the ISI Monitor 3D Backbone (RTSP/USB → Track2D/Track3D over UDP).",
    )
    ap.add_argument("--config", required=True, type=Path, help="path to backbone.yaml")
    ap.add_argument("--log-level", default="INFO",
                    help="logging level (DEBUG, INFO, WARNING, ERROR)")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("backbone.runtime")

    if not args.config.exists():
        log.error("config not found: %s", args.config)
        return 2

    try:
        orch = Orchestrator(str(args.config))
    except Exception:
        log.exception("failed to build the Backbone from %s", args.config)
        return 1

    log.info("Backbone built (mode=%s, sources=%s) — starting…",
             getattr(orch, "mode", "?"), list(getattr(orch, "_sources", {}).keys()))
    orch.install_signal_handlers()
    orch.run()   # blocks until SIGINT/SIGTERM sets the stop event
    log.info("Backbone stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
