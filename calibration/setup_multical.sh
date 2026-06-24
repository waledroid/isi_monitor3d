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
# Two pins are mandatory for a *working* install (Multical 0.4.0 is from 2022):
#   * Python 3.10 — Multical's dataclasses use mutable defaults that Python
#     3.11+ rejects ("mutable default ... is not allowed"). The system python3
#     here is 3.13, so PYTHON must point at a 3.10 interpreter (the `monitor3d`
#     conda env has one). This venv stays isolated regardless of which 3.10
#     interpreter seeds it — it gets its own site-packages + OpenCV 4.6.
#   * numpy < 2 — Multical's opencv-contrib-python 4.6 wheels are built against
#     the numpy 1.x ABI; numpy 2.x triggers "numpy.core.multiarray failed to
#     import". The resolver pulls numpy 2.x unless we pin it down afterwards.
#
# Usage:
#     PYTHON=/path/to/python3.10 ./calibration/setup_multical.sh
#     ./calibration/setup_multical.sh --force   # rebuilds it from scratch
#
# After this completes, `calibration/.venv-multical/bin/multical --help`
# should work. The Backbone env is unaffected.

set -euo pipefail

VENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.venv-multical"
MULTICAL_VERSION="${MULTICAL_VERSION:-0.4.0}"
# Prefer a real python3.10; the bare `python3` here is 3.13 (breaks Multical).
PYTHON="${PYTHON:-$(command -v python3.10 || command -v python3)}"

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

# Guard: Multical 0.4.0 won't import on Python 3.11+ (mutable dataclass defaults).
PY_VER="$("${PYTHON}" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
if [[ "${PY_VER}" != "3.10" ]]; then
    echo "[setup_multical] ERROR: Multical 0.4.0 needs Python 3.10, got ${PY_VER} (${PYTHON})." >&2
    echo "[setup_multical] Re-run with a 3.10 interpreter, e.g.:" >&2
    echo "    PYTHON=\$(conda run -n monitor3d which python) bash calibration/setup_multical.sh --force" >&2
    exit 1
fi

echo "[setup_multical] creating venv at ${VENV_DIR} (Python ${PY_VER})"
"${PYTHON}" -m venv "${VENV_DIR}"

echo "[setup_multical] upgrading pip"
"${VENV_DIR}/bin/pip" install --upgrade pip wheel setuptools

echo "[setup_multical] installing multical==${MULTICAL_VERSION}"
"${VENV_DIR}/bin/pip" install "multical==${MULTICAL_VERSION}"

# opencv-contrib-python 4.6 is built against the numpy 1.x ABI; force numpy<2.
echo "[setup_multical] pinning numpy<2 (opencv-contrib 4.6 ABI)"
"${VENV_DIR}/bin/pip" install "numpy<2"

# AprilGrid support (the `calibrate-2cam` extrinsic target) is OPTIONAL. It needs
# `apriltags2-ethz`, a C++/pybind11 package with NO usable PyPI wheel — it builds
# from source against SYSTEM libraries. Install these first (apt, need root):
#     sudo apt install -y cmake libopencv-dev libeigen3-dev
# (cmake must be 3.x — CMake 4 dropped the repo's `cmake_minimum_required(2.8)`;
#  the apt cmake 3.28 is fine. Do NOT pip-install cmake into the venv: that pulls
#  4.x and breaks the build.) The ChArUco path works WITHOUT any of this, so the
# block below is best-effort and never fails the bootstrap.
APRILTAGS_REPO="https://github.com/safijari/apriltags2_ethz.git"
echo "[setup_multical] (optional) building apriltags2-ethz for AprilGrid support"
if ! command -v cmake >/dev/null || ! [ -f /usr/include/eigen3/Eigen/Dense ] \
   || ! ls /usr/lib/*/cmake/opencv4/OpenCVConfig.cmake >/dev/null 2>&1; then
    echo "[setup_multical]   SKIP: missing system build deps. For AprilGrid run:"
    echo "[setup_multical]       sudo apt install -y cmake libopencv-dev libeigen3-dev"
    echo "[setup_multical]   then re-run this script. ChArUco calibration works regardless."
else
    SRC="$(mktemp -d)/apriltags2_ethz"
    # the repo vendors pybind11 as a submodule → must clone --recurse-submodules
    if git clone --recurse-submodules -q "${APRILTAGS_REPO}" "${SRC}"; then
        # setup.py hardcodes version='dev' (invalid PEP 440) but honors this env var;
        # --no-deps so it does NOT drag numpy>=2 / opencv-python over our pins;
        # --no-build-isolation so the build sees the venv's pybind11/numpy.
        if APPVEYOR_REPO_TAG_NAME="${MULTICAL_VERSION}" CMAKE_POLICY_VERSION_MINIMUM=3.5 \
            "${VENV_DIR}/bin/pip" install --no-deps --no-build-isolation "${SRC}" >/dev/null 2>&1; then
            echo "[setup_multical]   apriltags2-ethz built — AprilGrid boards/calibration available"
        else
            echo "[setup_multical]   apriltags2-ethz BUILD FAILED (see: ${VENV_DIR}/bin/pip install --no-deps --no-build-isolation ${SRC})"
        fi
    else
        echo "[setup_multical]   apriltags2-ethz clone failed (network?); skipping."
    fi
fi

# Multical's built-in 3D viewer (calibrate --vis / `vis` command). Multical declares
# the GUI deps as its `interactive` extra (matplotlib/qtpy/pyvista/pyvistaqt/colour/
# qtawesome) — let pip resolve them. qtpy is only an abstraction, so add a real Qt
# binding (PyQt5). Best-effort: calibration works without these; the viewer also
# needs a display (WSLg/X / $DISPLAY).
echo "[setup_multical] (optional) installing viewer deps via multical[interactive] + PyQt5"
if "${VENV_DIR}/bin/pip" install "multical[interactive]==${MULTICAL_VERSION}" PyQt5 >/dev/null 2>&1; then
    echo "[setup_multical]   3D viewer deps installed — use --vis / the 'vis' command (needs a display)"
else
    echo "[setup_multical]   viewer deps NOT installed; calibration still works, --vis/vis won't."
fi

# The optional installs above (apriltags / interactive) can pull numpy>=2 or a second
# OpenCV; re-assert the working pins LAST (opencv-contrib has the cv2.aruco multical
# needs; numpy<2 for its 4.6 ABI). --no-deps so it only fixes these two.
echo "[setup_multical] re-asserting numpy<2 + opencv-contrib pins (final)"
"${VENV_DIR}/bin/pip" install --no-deps "numpy<2" "opencv-contrib-python<=4.7.0" >/dev/null 2>&1 || true

# Patch a Multical 0.4.0 bug: the `multical intrinsic` subcommand
# (app/intrinsic.py) calls calibrate_cameras() WITHOUT the `intrinsic_error_limit`
# positional that camera.py:calibrate_cameras() requires, so the per-camera
# intrinsics solve crashes with
#   TypeError: calibrate_cameras() missing 1 required positional argument: 'intrinsic_error_limit'
# The parsed value (args.camera.intrinsic_error_limit, default 0.5) already exists;
# the call site just fails to pass it. The `multical calibrate` (joint) path is
# unaffected. Idempotent: re-running this script (or --force) re-applies it.
echo "[setup_multical] patching multical intrinsic subcommand (0.4.0 calibrate_cameras arg bug)"
"${VENV_DIR}/bin/python" - <<'PYPATCH'
import re
from pathlib import Path
import multical.app.intrinsic as m
p = Path(m.__file__)
src = p.read_text()
if "args.camera.intrinsic_error_limit" in src:
    print("[setup_multical]   already patched")
else:
    new = src.replace(
        "calibrate_cameras(boards, detected_points, image_sizes,",
        "calibrate_cameras(boards, detected_points, image_sizes,\n"
        "      args.camera.intrinsic_error_limit,",
        1,
    )
    if new == src:
        print("[setup_multical]   WARN: call site not found; intrinsic solve may fail")
    else:
        p.write_text(new)
        print("[setup_multical]   patched", p)
PYPATCH

echo
echo "[setup_multical] verifying installation"
"${VENV_DIR}/bin/multical" --help >/dev/null

echo
echo "[setup_multical] done. Multical binary: ${VENV_DIR}/bin/multical"
