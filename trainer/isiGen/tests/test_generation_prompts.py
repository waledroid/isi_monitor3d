"""Phase 7 prompt builder — anti-bleed structure from scaffold classes."""

import random

from src.core.runners import _build_prompt


def test_prompt_contains_triggers_and_background(tiny_project):
    _pdir, project = tiny_project
    rng = random.Random(0)
    p = _build_prompt(project, ["palette", "carton"], rng)
    assert "ISI_PLT" in p and "ISI_CRTN" in p
    assert p.startswith("a photo of ")
    assert "," in p                                  # background clause present


def test_prompt_unknown_class_skipped(tiny_project):
    _pdir, project = tiny_project
    p = _build_prompt(project, ["not_a_class"], random.Random(0))
    assert "industrial goods" in p
