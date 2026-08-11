"""CPU-branch port wiring — the dashboard must listen where the engine talks.

Regression: the branch offset the engine's UDP sink to 9003 (GPU-line
coexistence) but the dashboard's listener default stayed 9001 — the panels
went silently blind (no observations → no boxes/masks/tracks). Silence is
indistinguishable from 'nothing detected', so this is pinned by test."""

from pathlib import Path

import yaml

from monitor_web.config import Settings

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_dashboard_listener_matches_shipped_udp_sink() -> None:
    cfg = yaml.safe_load((_REPO_ROOT / "config" / "backbone.yaml").read_text())
    sinks = (cfg.get("metadata") or {}).get("sinks") or []
    udp = next(s for s in sinks if s.get("plugin") == "udp")
    assert Settings(backbone_config_path="x.yaml").udp_port == int(udp["port"]), (
        "dashboard bus_subscriber port must match the engine's udp sink — "
        "a mismatch makes every panel silently blind"
    )


def test_points_ingest_port_matches_shipped_config() -> None:
    cfg = yaml.safe_load((_REPO_ROOT / "config" / "backbone.yaml").read_text())
    assert cfg["ingestion"]["points"]["listen_port"] == 9012   # producer → engine
