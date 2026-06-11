# ISI Monitor 3D — Backbone

Industrial vision backbone for Isitec. Ingests RTSP from low-cost cameras and publishes metric, identity-stable tracks (`Track2D` always, `Track3D` on demand) over UDP/JSON to independent business modules.

See `docs/specs/Cahier des Charges-Système de Vision Industrielle.pdf` for the full spec and `~/.claude/plans/starry-stirring-fairy.md` for the build plan.

## Quick start (development)

```bash
conda env create -f environment.yml -n monitor3d
conda activate monitor3d
pytest
```

Refresh after `environment.yml` edits:

```bash
conda env update -f environment.yml -n monitor3d --prune
```

**Without conda** (pip-only path):

```bash
pip install -e ".[dev,geometry]"
pytest
```

Optional pip extras: `geometry` (OpenCV + FilterPy + scipy), `detection` (Ultralytics YOLO), `schemas` (pydantic). The conda env file already covers `geometry` and `dev`; don't combine the two install paths in the same environment.

**Calibration backend** (Multical) lives in its own isolated venv to avoid an OpenCV version conflict:

```bash
bash calibration/setup_multical.sh
```

## Repo layout

```
backbone/         Python package (core, shared, ingestion, detection,
                  homography, triangulation, metadata, runtime)
calibration/      Offline calibration tooling (ChArUco, Multical, floor anchor)
config/           YAML configs (backbone.yaml, subscriptions.yaml)
tools/            Operational tools (latency probe, etc.)
tests/            pytest suite
```

## Hardware

- **Dev:** NVIDIA RTX 5070 12 GB, Linux.
- **Production (later):** NVIDIA Jetson Orin NX 16 GB.
