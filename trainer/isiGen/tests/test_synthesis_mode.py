"""synthesis_mode auto-routing: copy_paste+inpaint when backgrounds exist, else depth.

Hermetic — no GPU/models. Pins resolve_synthesis(), its use in run_scaffolds /
run_generation (via stubbed registries), create_project persistence + default, and
the Studio create + scaffolds routes.
"""

from __future__ import annotations

import numpy as np
from src.core.manifest import Manifest, ManifestRecord
from src.core.project import ClassSpec, create_project, load_project, resolve_synthesis


def _add_background(pdir):
    m = Manifest.load(pdir)
    m.upsert(ManifestRecord(id="bg0001", sha256="b" * 64, image="raw/__bg__/bg0001.jpg",
                            class_name="", width=800, height=600, background=True))
    m.save()


# ── resolve_synthesis ────────────────────────────────────────────────────────
def test_auto_no_background_uses_depth(tiny_project):
    pdir, project = tiny_project
    sources, generator, info = resolve_synthesis(project, pdir)
    assert sources == ["depth_remix"]            # template default (box3d dropped)
    assert generator == "sdxl_controlnet"
    assert info["bg_count"] == 0 and info["mode"] == "auto"


def test_auto_with_background_uses_copy_paste(tiny_project):
    pdir, project = tiny_project
    _add_background(pdir)
    sources, generator, info = resolve_synthesis(project, pdir)
    assert sources == ["copy_paste"]
    assert generator == "sdxl_inpaint"
    assert info["bg_count"] == 1 and "copy_paste" in info["path"]


def test_forced_depth_ignores_backgrounds(tiny_project):
    pdir, _ = tiny_project
    _add_background(pdir)
    project = load_project(pdir)
    project.synthesis_mode = "depth"
    sources, generator, _ = resolve_synthesis(project, pdir)
    assert sources == ["depth_remix"] and generator == "sdxl_controlnet"


def test_forced_copy_paste_without_backgrounds(tiny_project):
    pdir, _ = tiny_project
    project = load_project(pdir)
    project.synthesis_mode = "copy_paste"
    sources, generator, _ = resolve_synthesis(project, pdir)
    assert sources == ["copy_paste"] and generator == "sdxl_inpaint"


def test_depth_keeps_configured_box3d(tiny_project):
    """Depth mode preserves explicitly-configured depth sources (e.g. box3d)."""
    pdir, _ = tiny_project
    project = load_project(pdir)
    project.phases.setdefault("scaffolds", {})["sources"] = ["depth_remix", "box3d_procedural"]
    project.synthesis_mode = "depth"
    sources, _, _ = resolve_synthesis(project, pdir)
    assert sources == ["depth_remix", "box3d_procedural"]


# ── create_project persistence ───────────────────────────────────────────────
def test_create_project_default_and_explicit(tmp_path):
    data = tmp_path / "data"
    cls = [ClassSpec(name="polybag", trigger="ISI_PLYBG", color=[40, 90, 230])]
    p1 = create_project(data, "auto_proj", cls)
    assert load_project(p1).synthesis_mode == "auto"
    p2 = create_project(data, "paste_proj", cls, synthesis_mode="copy_paste")
    assert load_project(p2).synthesis_mode == "copy_paste"


# ── runner routing (stubbed registries) ──────────────────────────────────────
def test_run_scaffolds_routes_to_copy_paste(tiny_project, monkeypatch):
    # SCAFFOLD_SOURCES is imported inside run_scaffolds → patch the singleton itself.
    from src.core import runners
    from src.stages.scaffolds.base import SCAFFOLD_SOURCES
    pdir, _ = tiny_project
    _add_background(pdir)
    seen = []

    class _StubSource:
        def generate(self, project, n):
            return iter(())                       # yield nothing → no files needed

    monkeypatch.setattr(SCAFFOLD_SOURCES, "create",
                        lambda name, **k: (seen.append(name), _StubSource())[1])
    runners.run_scaffolds(pdir, count=2)
    assert seen == ["copy_paste"]                 # bg present → copy_paste, not depth_remix


def test_run_generation_picks_paired_generator(tiny_project, monkeypatch):
    # IMAGE_GENERATORS.create runs before the no-pending early-return, so the
    # paired generator name is observable even with zero scaffolds.
    from src.core import runners
    from src.stages.generation.base import IMAGE_GENERATORS
    pdir, _ = tiny_project
    _add_background(pdir)
    picked = {}

    class _StubGen:
        def load(self):
            pass

        def generate(self, *a, **k):
            return np.zeros((4, 4, 3), "uint8")

        def close(self):
            pass

    monkeypatch.setattr(IMAGE_GENERATORS, "create",
                        lambda name, **k: (picked.__setitem__("name", name), _StubGen())[1])
    out = runners.run_generation(pdir)            # no pending scaffolds → generated 0
    assert picked["name"] == "sdxl_inpaint"
    assert out == {"generated": 0}


def test_run_scaffolds_persists_placement(tiny_project, monkeypatch):
    from src.core import runners
    from src.core.project import load_project
    from src.stages.scaffolds.base import SCAFFOLD_SOURCES
    pdir, _ = tiny_project

    class _StubSource:
        def generate(self, project, n):
            return iter(())

    monkeypatch.setattr(SCAFFOLD_SOURCES, "create", lambda name, **k: _StubSource())
    runners.run_scaffolds(pdir, placement="original")
    assert load_project(pdir).phase("scaffolds")["copy_paste"]["placement"] == "original"


def test_run_generation_persists_strength(tiny_project, monkeypatch):
    from src.core import runners
    from src.stages.generation.base import IMAGE_GENERATORS
    pdir, _ = tiny_project

    class _StubGen:
        def load(self):
            pass

        def generate(self, *a, **k):
            return np.zeros((4, 4, 3), "uint8")

        def close(self):
            pass

    monkeypatch.setattr(IMAGE_GENERATORS, "create", lambda name, **k: _StubGen())
    runners.run_generation(pdir, strength=0.3)    # mint-page slider value
    assert load_project(pdir).phase("generation")["strength"] == 0.3


# ── Studio routes ─────────────────────────────────────────────────────────────
def test_studio_create_and_scaffolds_badge(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from src.studio.app import create_app
    from src.studio.config import Settings
    with TestClient(create_app(Settings())) as c:
        body = {"name": "synp", "classes": [
            {"name": "polybag", "trigger": "ISI_PLYBG", "color": [40, 90, 230]}],
            "synthesis_mode": "copy_paste"}
        assert c.post("/api/projects", json=body).json()["ok"] is True
        syn = c.get("/api/p/synp/scaffolds").json()["synthesis"]
        assert syn["generator"] == "sdxl_inpaint" and syn["mode"] == "copy_paste"
        # mint-page generation settings: inpaint path exposes the strength slider
        gen = c.get("/api/p/synp/generation").json()
        assert gen["is_inpaint"] is True and isinstance(gen["strength"], float)
