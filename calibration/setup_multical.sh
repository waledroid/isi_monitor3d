#!/usr/bin/env bash
# Bootstrap an isolated virtualenv for Multical.
#
# Why a separate venv? Multical 0.4.0 (last release, 2022) pins
# `opencv-contrib-python <=4.7.0`, which conflicts with the Backbone's
# runtime OpenCV (4.13+). Installing Multical into the project env would
# downgrade OpenCV and break the rest of the geometric pipeline. Putting
# Multical in `calibration/.venv-multical/` keeps the runtime env clean —
# `calibrate.py` invokes the `multical` binary from this venv by absolute
# path and never imports the package.
#
# Usage:
#     ./calibration/setup_multical.sh           # creates the venv
#     ./calibration/setup_multical.sh --force   # rebuilds it from scratch
#
# After this completes, `calibration/.venv-multical/bin/multical --help`
# should work. The Backbone env is unaffected.

set -euo pipefail

VENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.venv-multical"
MULTICAL_VERSION="${MULTICAL_VERSION:-0.4.0}"
PYTHON="${PYTHON:-python3}"

if [[ "${1:-}" == "--force" && -d "${VENV_DIR}" ]]; then
    echo "[setup_multical] removing existing venv at ${VENV_DIR}"
    rm -rf "${VENV_DIR}"
fi

if [[ -x "${VENV_DIR}/bin/multical" ]]; then
    echo "[setup_multical] Multical already installed at ${VENV_DIR}/bin/multical"
    "${VENV_DIR}/bin/multical" --help >/dev/null 2>&1 && {
        echo "[setup_multical] CLI is healthy. Skipping. (Pass --force to rebuild.)"
        exit 0
    }
    echo "[setup_multical] existing CLI is broken; rebuilding"
    rm -rf "${VENV_DIR}"
fi

echo "[setup_multical] creating venv at ${VENV_DIR}"
"${PYTHON}" -m venv "${VENV_DIR}"

echo "[setup_multical] upgrading pip"
"${VENV_DIR}/bin/pip" install --upgrade pip wheel setuptools

echo "[setup_multical] installing multical==${MULTICAL_VERSION}"
"${VENV_DIR}/bin/pip" install "multical==${MULTICAL_VERSION}"

echo
echo "[setup_multical] verifying installation"
"${VENV_DIR}/bin/multical" --help >/dev/null

echo
echo "[setup_multical] done. Multical binary: ${VENV_DIR}/bin/multical"
