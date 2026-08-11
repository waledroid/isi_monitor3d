"""Runtime configuration — pydantic-settings, env-var overrideable."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root: monitor_web/monitor_web/config.py -> parents[2] == isi_monitor3d/
_REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """All knobs for ``monitor_web``. Override via ``MONITOR_WEB_*`` env vars."""

    model_config = SettingsConfigDict(env_prefix="MONITOR_WEB_", env_file=None)

    # HTTP server.
    host: str = "0.0.0.0"
    port: int = 8000

    @property
    def instance_id(self) -> str:
        """Stable identity of THIS dashboard instance, stamped onto every
        child it spawns and matched by the stray reapers. Port-derived: the
        TCP bind guarantees uniqueness among live siblings (:8000 vs :8100),
        and it survives crashes so the next run on the same port adopts the
        previous run's orphans. Override with ``ISI3D_INSTANCE_ID``."""
        return os.environ.get("ISI3D_INSTANCE_ID") or f"monitor-web:{self.port}"

    # Backbone integration.
    backbone_config_path: Path = Field(
        default=_REPO_ROOT / "config" / "backbone.yaml",
        description="Path to the Backbone's backbone.yaml; START spawns it as a subprocess. "
        "Absolute (repo-root) so Save, calibration-write, and the launched subprocess "
        "all resolve to the same file regardless of the server's working directory.",
    )
    zones_path: Path | None = Field(
        default=None,
        description="Optional explicit zones.yaml. If unset, read from backbone.yaml's zones_path.",
    )
    danger_zones_object_path: Path | None = Field(
        default=_REPO_ROOT / "config" / "danger_zones_object.yaml",
        description=(
            "Per-class proximity-ring config (Type-1 danger zones — attached to "
            "tracked objects). Missing file => no per-object rings drawn."
        ),
    )
    warehouse_map_path: Path = Field(
        default=_REPO_ROOT / "config" / "warehouse_map.yaml",
        description="Static warehouse layout twin (racks/walls/obstacles) for the floor map.",
    )
    calibration_path: Path | None = Field(
        default=None,
        description=(
            "Optional explicit calibration.json (S17). If unset, read from "
            "backbone.yaml's calibration_path. Used by the projection endpoints "
            "(/api/project/*) that drive 'draw zones on CAM' + zone overlays."
        ),
    )
    link_lines_path: Path | None = Field(
        default=_REPO_ROOT / "config" / "link_lines.yaml",
        description=(
            "Per-class-pair distance-line rules (S16). The floor map draws a "
            "thin white line between matching object pairs with a live distance "
            "label. Missing file => no link lines drawn."
        ),
    )

    # UDP listener for the engine's metadata envelopes. CPU branch: 9003 —
    # offset from the GPU line's 9001 so both stacks can coexist on one dev
    # box (a shared port would REUSEPORT-steal each other's flows). Must
    # match the Backbone's metadata.sinks udp port in backbone.yaml (9003).
    # Low port on purpose: under WSL2 mirrored networking, ports in the
    # Windows dynamic/Hyper-V reserved range (~49152+) fail to bind.
    udp_host: str = "127.0.0.1"
    udp_port: int = 9003

    # Status panel — how stale before the green dot goes red.
    freshness_threshold_s: float = 2.0

    # Subprocess control.
    backbone_terminate_timeout_s: float = 2.0

    # Log ring buffer (lines).
    log_buffer_size: int = 500

    # Default UI language.
    default_lang: str = "fr"

    # Hidden dev MP4 viewer (S12.2). Unlocked by double-clicking the logo +
    # password; plays a media-folder MP4 through the detector in the big view.
    mp4_unlock_password: str = "isitec"
    media_dir: Path = Field(
        default=_REPO_ROOT,
        description="Root scanned recursively for *.mp4 in the hidden MP4 dev viewer "
        "(default: the whole project, hidden dirs pruned).",
    )
    mp4_conf_threshold: float = 0.25

    # Unified dashboard UI settings, persisted server-side (survives browser
    # sessions / machines), e.g. the chosen MP4. Cameras/zones keep their own
    # YAMLs (backbone.yaml / zones.yaml) — this is for UI-only preferences.
    ui_settings_path: Path = Field(
        default=_REPO_ROOT / "config" / "monitor_web_ui.yaml",
        description="YAML store for dashboard UI preferences (mp4 selection, etc.).",
    )

    # isicomms gateway integration — cross-warehouse node status.
    # Set MONITOR_WEB_GATEWAY_URL (e.g. http://gateway-host:8080) to enable the
    # "Warehouse nodes" sidebar panel.  When unset the panel shows a muted hint.
    gateway_url: str | None = Field(
        default=None,
        description=(
            "Base URL of the isicomms gateway (e.g. http://gateway-host:8080). "
            "When set, GET /api/gateway/nodes proxies the gateway's /nodes endpoint "
            "so the dashboard shows the cross-warehouse node list."
        ),
    )
    gateway_token: str | None = Field(
        default=None,
        description=(
            "Bearer token sent as 'Authorization: Bearer <token>' to the gateway. "
            "Leave unset when the gateway runs without auth."
        ),
    )
    gateway_timeout_s: float = Field(
        default=3.0,
        description="HTTP timeout (seconds) for the gateway /nodes request.",
    )
