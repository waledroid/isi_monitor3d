"""Shared test fixtures."""
import pytest


@pytest.fixture(autouse=True)
def _isolate_ui_settings(tmp_path_factory, monkeypatch):
    """Redirect the unified dashboard config (``ui_settings_path``) to a throwaway
    file for every test, so tests never read or write the real
    ``config/monitor_web_ui.yaml`` (the merged link_lines / warehouse_map /
    zone_patches / danger_zones sections now all funnel through it). Settings reads
    the ``MONITOR_WEB_`` env prefix, and init kwargs that omit ``ui_settings_path``
    fall through to this override.
    """
    p = tmp_path_factory.mktemp("uisettings") / "monitor_web_ui.yaml"
    monkeypatch.setenv("MONITOR_WEB_UI_SETTINGS_PATH", str(p))
    yield
