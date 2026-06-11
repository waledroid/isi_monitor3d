"""Phase 6 scaffolds — paired (control, mask) generation, fully local."""

import cv2
import numpy as np
from src.core.manifest import Manifest
from src.core.runners import load_scaffold_index, run_scaffolds
from src.stages.scaffolds.box3d_procedural import Box3dProceduralScaffolds
from src.stages.scaffolds.depth_remix import DepthRemixScaffolds


def test_box3d_pairs_are_valid(tiny_project):
    _pdir, project = tiny_project
    src = Box3dProceduralScaffolds(width=320, height=240, seed=7)
    pairs = list(src.generate(project, 5))
    assert len(pairs) == 5
    colors = {tuple(c.color) for c in project.classes}
    for control, mask, meta in pairs:
        assert control.shape == (240, 320) and control.dtype == np.uint8
        assert mask.shape == (240, 320, 3)
        assert meta["classes"]                       # at least one class present
        # every painted mask pixel uses an exact project class color (BGR)
        painted = mask[mask.sum(axis=2) > 0]
        for px in np.unique(painted.reshape(-1, 3), axis=0):
            b, g, r = (int(v) for v in px)
            assert (r, g, b) in colors


def test_box3d_deterministic_with_seed(tiny_project):
    _pdir, project = tiny_project
    a = list(Box3dProceduralScaffolds(width=160, height=120, seed=3).generate(project, 2))
    b = list(Box3dProceduralScaffolds(width=160, height=120, seed=3).generate(project, 2))
    assert np.array_equal(a[0][0], b[0][0]) and np.array_equal(a[1][1], b[1][1])


def _fake_phase2(pdir, project):
    """Give the tiny project fake depth+mask artifacts so depth_remix has input."""
    m = Manifest.load(pdir)
    spec = project.classes[0]
    for rec in m.active():
        depth = np.tile(np.linspace(30, 220, 120).astype(np.uint8)[:, None], (1, 160))
        mask = np.zeros((120, 160, 3), dtype=np.uint8)
        r, g, b = spec.color
        mask[40:90, 50:120] = (b, g, r)
        cv2.imwrite(str(pdir / f"maps/depth/{rec.id}.png"), depth)
        cv2.imwrite(str(pdir / f"maps/mask/{rec.id}.png"), mask)
        rec.depth_map = f"maps/depth/{rec.id}.png"
        rec.mask = f"maps/mask/{rec.id}.png"
        m.upsert(rec)
    m.save()


def test_depth_remix_pairs(tiny_project):
    pdir, project = tiny_project
    _fake_phase2(pdir, project)
    src = DepthRemixScaffolds(project_dir=str(pdir), seed=11)
    pairs = list(src.generate(project, 4))
    assert len(pairs) == 4
    for control, mask, meta in pairs:
        assert control.shape == (120, 160) and mask.shape == (120, 160, 3)
        assert meta["source"] == "depth_remix" and meta["from"]
        # mask colors stay EXACT through the affine (NEAREST)
        painted = mask[mask.sum(axis=2) > 0]
        if painted.size:
            uniq = np.unique(painted.reshape(-1, 3), axis=0)
            r, g, b = project.classes[0].color
            assert [b, g, r] in uniq.tolist()


def test_run_scaffolds_writes_index(tiny_project):
    pdir, project = tiny_project
    _fake_phase2(pdir, project)
    out = run_scaffolds(pdir, count=4)
    assert out["scaffolds"] >= 4                     # 2 sources x >=2 each
    entries = load_scaffold_index(pdir)
    assert len(entries) == out["total"]
    for e in entries:
        assert (pdir / e["control"]).exists() and (pdir / e["mask"]).exists()
        assert e["status"] == "pending"
