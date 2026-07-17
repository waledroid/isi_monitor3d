#!/usr/bin/env bash
# ISI Monitor 3D cheat-sheet docs launcher — self-contained: creates .venv on first run.
# Copy this folder anywhere and run ./launch.sh  (docs on :8500).
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  ./.venv/bin/pip install --upgrade pip -q
  ./.venv/bin/pip install -r requirements.txt
fi
exec ./.venv/bin/mkdocs serve -a 0.0.0.0:8500 "$@"
