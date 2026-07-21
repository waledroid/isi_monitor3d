#!/usr/bin/env bash
# Build the ISI Monitor 3D manuscript PDF.
# Tectonic shells out to `biber` (biblatex backend); biber 2.17 was installed
# into the tex env's bin, so that dir must be on PATH for the child process.
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/miniforge3/envs/tex/bin:$PATH"
tectonic isimonitor3d.tex
pdfinfo isimonitor3d.pdf | grep Pages
