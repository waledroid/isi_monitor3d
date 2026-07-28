"""Shared test fixtures."""
import pytest


@pytest.fixture(autouse=True)
def _isolate_frame_bus(tmp_path_factory, monkeypatch):
    """Point the shared frame bus at a per-session temp dir. Without this a
    LIVE isistream on the same box leaks its /dev/shm frames into the suite —
    the camera-hub tests then stream from the real bus and never build their
    fake sources (observed: test_multiple_viewers_share_one_source failing
    only while the production stack ran)."""
    monkeypatch.setenv("ISI3D_SHM_DIR",
                       str(tmp_path_factory.mktemp("frame_bus")))


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

# ---- LIVE-FIRE GUARD (suite-wide) -------------------------------------------
# The reapers (`BackboneSupervisor._kill_backbones` and
# `IsistreamHost._reap_strays`, reached from app lifespan, START, STOP, and
# config-save restarts) SIGKILL matching processes on the HOST. Before the
# instance-identity work they matched by cmdline alone: any test that booted
# the app murdered a LIVE production Backbone running beside the suite
# (observed 2026-07-06: the operator's system "crashed at random" — the random
# was `pytest`); the producer reaper had the same hole. The identity rule now
# scopes kills to the suite's own (pid-qualified, matching nothing) instance —
# and this guard stays as BELT + BRACES on top:
#   belt   — ISI3D_DISABLE_REAP inerts every kill path, inherited by any
#            subprocess the suite spawns;
#   braces — both finders are patched to find nothing.
# The one test that needs the real scan gets it read-only via
# `real_backbone_finder` (the finder has no disable gate, by design).
from monitor_web import proc_reaper as _proc_reaper  # noqa: E402
from monitor_web.backbone_supervisor import BackboneSupervisor as _BBS  # noqa: E402

_REAL_BACKBONE_FINDER = _BBS._find_backbone_pids
_REAL_FIND_STRAYS = _proc_reaper.find_strays


@pytest.fixture(autouse=True)
def _no_host_wide_reaping(monkeypatch):
    monkeypatch.setenv(_proc_reaper.DISABLE_ENV, "1")
    monkeypatch.setattr(_proc_reaper, "find_strays",
                        lambda token, instance_id, exclude=None: [])
    monkeypatch.setattr(_BBS, "_find_backbone_pids",
                        lambda self, exclude=None: [])


@pytest.fixture
def real_backbone_finder():
    """The un-neutered /proc scan, for READ-ONLY assertions."""
    return _REAL_BACKBONE_FINDER


@pytest.fixture
def real_find_strays():
    """The un-neutered shared finder, for READ-ONLY assertions."""
    return _REAL_FIND_STRAYS
