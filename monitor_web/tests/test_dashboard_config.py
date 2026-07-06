"""Unified dashboard store — durability guarantees.

Pins the anti-wipe behavior added after the live incident where a transient
read failure of ``monitor_web_ui.yaml`` let the legacy-migration path rebuild
the file from scratch, erasing zones, the Mode-2 calibration override, the
alignment store, and every UI preference in one atomic write.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import yaml

from monitor_web import dashboard_config as dc


@pytest.fixture()
def cfg(tmp_path):
    return SimpleNamespace(
        ui_settings_path=tmp_path / "monitor_web_ui.yaml",
        link_lines_path=tmp_path / "link_lines.yaml",
        warehouse_map_path=tmp_path / "warehouse_map.yaml",
        # zone_patches' legacy file lives beside backbone.yaml
        backbone_config_path=tmp_path / "backbone.yaml",
        danger_zones_object_path=tmp_path / "danger_zones_object.yaml",
    )


def _seed(cfg, data: dict) -> None:
    cfg.ui_settings_path.write_text(yaml.safe_dump(data))


GOOD = {
    "display_fps": 25,
    "mode2_calibration_path": "/some/calibration.json",
    "zone_patches": {"patches": [{"id": "zp_1", "camera_id": "cam_a"}]},
}


def test_write_all_keeps_bak_of_previous_content(cfg):
    _seed(cfg, GOOD)
    dc.write_all(cfg, {"display_fps": 30})
    bak = cfg.ui_settings_path.with_suffix(".yaml.bak")
    assert yaml.safe_load(bak.read_text()) == GOOD
    assert dc.load_all(cfg) == {"display_fps": 30}


def test_read_section_never_migrates_over_corrupt_store(cfg):
    cfg.ui_settings_path.write_text("zone_patches: [unclosed\n  - {broken")
    before = cfg.ui_settings_path.read_text()
    # Legacy file present — the old code would have "migrated" it in,
    # rebuilding the store and destroying every other section.
    (cfg.backbone_config_path.parent / "zone_patches.yaml").write_text(
        yaml.safe_dump({"patches": []}))
    assert dc.read_section(cfg, "zone_patches") == {}
    assert cfg.ui_settings_path.read_text() == before  # untouched


def test_write_section_refuses_to_clobber_corrupt_store(cfg):
    cfg.ui_settings_path.write_text(":\n:::bad yaml {{{")
    before = cfg.ui_settings_path.read_text()
    with pytest.raises(dc.StoreCorrupt):
        dc.write_section(cfg, "zone_patches", {"patches": []})
    assert cfg.ui_settings_path.read_text() == before


def test_write_section_preserves_siblings_on_healthy_store(cfg):
    _seed(cfg, GOOD)
    dc.write_section(cfg, "link_lines", {"rules": []})
    data = dc.load_all(cfg)
    assert data["link_lines"] == {"rules": []}
    assert data["mode2_calibration_path"] == GOOD["mode2_calibration_path"]
    assert data["zone_patches"] == GOOD["zone_patches"]


def test_migration_still_works_when_store_absent(cfg):
    (cfg.backbone_config_path.parent / "zone_patches.yaml").write_text(
        yaml.safe_dump({"patches": [{"id": "zp_legacy"}]}))
    assert dc.read_section(cfg, "zone_patches") == {
        "patches": [{"id": "zp_legacy"}]}
    # migrated into the unified store
    assert dc.load_all(cfg)["zone_patches"]["patches"][0]["id"] == "zp_legacy"


def test_load_all_degrades_to_empty_for_readers(cfg):
    cfg.ui_settings_path.write_text("{{{not yaml")
    assert dc.load_all(cfg) == {}
