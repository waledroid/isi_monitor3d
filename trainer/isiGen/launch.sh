#!/usr/bin/env bash
# isiGen Studio launcher — self-contained: creates .venv on first run.
# Copy this folder anywhere and run ./launch.sh  (Studio on :8200, ISIGEN_* env).
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  ./.venv/bin/pip install --upgrade pip -q
  ./.venv/bin/pip install -r requirements.txt
fi
exec ./.venv/bin/python scripts/run_studio.py "$@"
