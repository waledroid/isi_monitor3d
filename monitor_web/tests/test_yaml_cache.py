"""The display hot path must never re-parse YAML per frame (it starved the loop)."""

from __future__ import annotations

import os
import time

import yaml

from monitor_web import yaml_cache
from monitor_web.yaml_cache import invalidate, load_yaml_cached


def _write(p, data, *, aged: bool = True):
    """Write, then (optionally) backdate the mtime past the cache's settle
    window — production settings files are seconds-to-days old when read."""
    p.write_text(yaml.safe_dump(data))
    if aged:
        old = time.time() - 60
        os.utime(p, (old, old))


def test_parses_once_until_mtime_changes(tmp_path, monkeypatch):
    p = tmp_path / "ui.yaml"
    _write(p, {"show_masks": True})
    invalidate()

    parses = {"n": 0}
    real = yaml.load

    def counting_load(*a, **k):
        parses["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(yaml_cache.yaml, "load", counting_load)

    for _ in range(500):
        assert load_yaml_cached(p)["show_masks"] is True
    assert parses["n"] == 1, f"hot path re-parsed YAML {parses['n']} times"

    # A write (atomic in production) must be picked up.
    _write(p, {"show_masks": False})
    assert load_yaml_cached(p)["show_masks"] is False
    assert parses["n"] == 2


def test_fresh_writes_are_never_served_stale(tmp_path):
    """An in-place rewrite of identical byte-length within one mtime tick must
    still be seen — the operator's Settings save may not be silently ignored."""
    p = tmp_path / "ui.yaml"
    invalidate()
    p.write_text("distance_line_color: local\n")
    assert load_yaml_cached(p)["distance_line_color"] == "local"
    p.write_text("distance_line_color: bogus\n")     # same length, same tick
    assert load_yaml_cached(p)["distance_line_color"] == "bogus"


def test_missing_and_bad_files_are_empty_dicts(tmp_path):
    invalidate()
    assert load_yaml_cached(tmp_path / "nope.yaml") == {}
    bad = tmp_path / "bad.yaml"
    bad.write_text("[1, 2, 3]")          # a list, not a mapping
    assert load_yaml_cached(bad) == {}


def test_ui_pref_is_cached(tmp_path, monkeypatch):
    """overlay._ui_pref is called several times per FRAME per stream."""
    from monitor_web import overlay

    p = tmp_path / "ui.yaml"
    _write(p, {"show_boxes": False})
    invalidate()
    cfg = type("C", (), {"ui_settings_path": p})()

    parses = {"n": 0}
    real = yaml.load
    monkeypatch.setattr(yaml_cache.yaml, "load",
                        lambda *a, **k: (parses.__setitem__("n", parses["n"] + 1), real(*a, **k))[1])
    for _ in range(200):
        assert overlay.boxes_enabled(cfg) is False
    assert parses["n"] == 1
