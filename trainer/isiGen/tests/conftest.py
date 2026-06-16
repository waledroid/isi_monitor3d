"""isiGen test bootstrap — hermetic: no GPU, no model downloads, no heavy imports."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ISIGEN_ROOT = Path(__file__).resolve().parents[1]
if str(ISIGEN_ROOT) not in sys.path:
    sys.path.insert(0, str(ISIGEN_ROOT))


@pytest.fixture(autouse=True)
def _isolated_dirs(tmp_path, monkeypatch):
    """Every test gets a throwaway data/runs dir via the ISIGEN_ env prefix."""
    monkeypatch.setenv("ISIGEN_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("ISIGEN_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("ISIGEN_DISABLE_REAP", "1")     # never kill real procs in tests
    yield


@pytest.fixture
def tiny_project(tmp_path):
    """A real project dir with 3 synthetic images ingested (one per class)."""
    import cv2
    import numpy as np
    from src.core.project import ClassSpec, create_project, load_project
    from src.core.runners import run_curate

    data_dir = tmp_path / "data"
    pdir = create_project(data_dir, "tiny", [
        ClassSpec(name="palette", trigger="ISI_PLT", color=[220, 40, 40]),
        ClassSpec(name="carton", trigger="ISI_CRTN", color=[40, 200, 40]),
        ClassSpec(name="polybag", trigger="ISI_PLYBG", color=[40, 90, 230]),
    ])
    src = tmp_path / "incoming"
    for i, cls in enumerate(["palette", "carton", "polybag"]):
        d = src / cls
        d.mkdir(parents=True, exist_ok=True)
        img = np.full((600, 800, 3), 30 + i * 40, dtype=np.uint8)
        cv2.rectangle(img, (200, 150), (600, 450), (0, 128 + i * 40, 255), -1)
        cv2.imwrite(str(d / f"img_{i}.png"), img)
    run_curate(pdir, source=str(src), auto_class=True)
    return pdir, load_project(pdir)
