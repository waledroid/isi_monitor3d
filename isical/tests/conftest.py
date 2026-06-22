"""isical test bootstrap — hermetic: no cameras, no Multical, no GStreamer."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(autouse=True)
def _isolated_dirs(tmp_path, monkeypatch):
    monkeypatch.setenv("ISICAL_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("ISICAL_RUNS_DIR", str(tmp_path / "runs"))
    # keep "install" + backbone-stamp writes inside the tmp tree
    monkeypatch.setenv("ISICAL_MODE2_CALIBRATION_PATH", str(tmp_path / "mode2" / "calibration.json"))
    monkeypatch.setenv("ISICAL_BACKBONE_CONFIG_PATH", str(tmp_path / "backbone.yaml"))
    yield


@pytest.fixture
def tiny_project(tmp_path):
    """A 2-camera calibration project (rtsp urls) under the isolated data dir."""
    from isical.core.project import CameraSpec, create_project, load_project
    data = tmp_path / "data"
    cams = {
        "cam_a": CameraSpec(id="cam_a", type="rtsp", url="rtsp://x/a"),
        "cam_b": CameraSpec(id="cam_b", type="rtsp", url="rtsp://x/b"),
    }
    pdir = create_project(data, "rig", cams)
    return pdir, load_project(pdir)
