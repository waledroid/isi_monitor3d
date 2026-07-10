"""Unified dashboard config store — one sectioned ``monitor_web_ui.yaml``.

Merges the formerly-separate dashboard-only config files — ``link_lines.yaml``,
``warehouse_map.yaml``, ``zone_patches.yaml``, ``danger_zones_object.yaml`` —
into ONE file beside the existing UI preferences, so the Settings modal can load
and save everything from a single place and nothing is lost between sessions.

The Backbone-contract files (``backbone.yaml``, ``zones.yaml``,
``calibration.json``) are deliberately NOT merged — the engine reads those
directly (process-boundary rule), so they stay where they are.

``monitor_web_ui.yaml`` layout::

    # top-level: UI preferences (model_*_path, mp4 selection …) —
    # left at the top level so existing readers keep working unchanged.
    ...
    # merged sections (sibling keys):
    link_lines:          {rules: [...]}
    warehouse_map:       {elements: [...], outline: {...}}
    zone_patches:        {patches: [...]}
    danger_zones_object: {classes: {...}}

Back-compat: a read for a section that isn't in the unified file yet falls back
to the legacy standalone file and migrates it in (one-time write). After that the
unified file is authoritative. Legacy files are left in place as a backup.
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


class StoreCorrupt(RuntimeError):
    """The unified dashboard config EXISTS but cannot be read/parsed. Writers
    must refuse rather than rebuild over it — one transient read failure once
    erased every zone/calibration override/UI pref (observed live)."""

# Merged section keys (NOT the top-level UI preferences, which stay flat).
SECTIONS = ("link_lines", "warehouse_map", "zone_patches", "danger_zones_object")


def unified_path(cfg) -> Path:
    """The single dashboard config file (== the UI-settings YAML)."""
    return Path(cfg.ui_settings_path)


def _legacy_path(cfg, section: str) -> Path | None:
    """Where a section used to live as its own file (for one-time migration)."""
    if section == "link_lines":
        return Path(cfg.link_lines_path) if cfg.link_lines_path else None
    if section == "warehouse_map":
        return Path(cfg.warehouse_map_path) if cfg.warehouse_map_path else None
    if section == "danger_zones_object":
        return Path(cfg.danger_zones_object_path) if cfg.danger_zones_object_path else None
    if section == "zone_patches":
        return Path(cfg.backbone_config_path).resolve().parent / "zone_patches.yaml"
    return None


def _load_all_or_none(cfg) -> dict | None:
    """The unified file as a dict; ``{}`` when absent; ``None`` when the file
    EXISTS but is unreadable/unparseable (corruption — treat loudly, never as
    empty for anything that might write)."""
    p = unified_path(cfg)
    if not p.exists():
        return {}
    try:
        data = yaml.safe_load(p.read_text()) or {}
    except (OSError, yaml.YAMLError) as exc:
        logger.error("dashboard_config: %s exists but is unreadable (%s) — "
                     "refusing to treat as empty", p, exc)
        return None
    return data if isinstance(data, dict) else None


def load_all(cfg) -> dict:
    """The whole unified file as a dict (``{}`` if missing/unreadable) —
    read-only convenience; write paths use the strict variant."""
    data = _load_all_or_none(cfg)
    return {} if data is None else data


def write_all(cfg, data: dict) -> None:
    """Atomically write the whole unified file (tempfile + ``os.replace``).

    Keeps a one-generation ``.bak`` of the previous content — the cheap
    insurance that would have saved the operator's zones when the store was
    rebuilt over a transient read failure."""
    p = unified_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        try:
            shutil.copy2(p, p.with_suffix(p.suffix + ".bak"))
        except OSError:
            logger.warning("dashboard_config: could not write backup for %s", p)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True)
        os.replace(tmp, str(p))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _legacy_doc(cfg, section: str) -> dict | None:
    """Read a section's legacy standalone file, or None if absent/unreadable."""
    lp = _legacy_path(cfg, section)
    if lp is None or not lp.exists():
        return None
    try:
        data = yaml.safe_load(lp.read_text()) or {}
    except (OSError, yaml.YAMLError):
        return None
    return data if isinstance(data, dict) else None


def read_section(cfg, section: str) -> dict:
    """Return a section dict from the unified file.

    If the section isn't present yet but a legacy standalone file exists, migrate
    it into the unified file (one write) and return it. Missing everywhere → ``{}``.
    """
    data = _load_all_or_none(cfg)
    if data is None:
        # Existing-but-corrupt store: NEVER migrate/rebuild over it (that is
        # how every section was once wiped). Degrade to empty for this read.
        return {}
    val = data.get(section)
    if isinstance(val, dict):
        return val
    legacy = _legacy_doc(cfg, section)
    if legacy is not None:
        data[section] = legacy
        try:
            write_all(cfg, data)
            logger.info("dashboard_config: migrated %r into %s", section, unified_path(cfg).name)
        except OSError as exc:
            logger.warning("dashboard_config: migrate %r failed: %s", section, exc)
        return legacy
    return {}


def write_section(cfg, section: str, doc: dict) -> None:
    """Write one section into the unified file, preserving the others + UI prefs.

    Raises :class:`StoreCorrupt` when the store exists but can't be read —
    a blind write here would silently destroy every other section."""
    data = _load_all_or_none(cfg)
    if data is None:
        raise StoreCorrupt(
            f"{unified_path(cfg)} exists but is unreadable — fix or remove it "
            f"(a .bak of the last good write sits next to it)")
    data[section] = doc
    write_all(cfg, data)
