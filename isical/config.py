"""isical Studio settings — pydantic-settings with the ISICAL_ env prefix
(mirrors monitor_web's MONITOR_WEB_ / isiGen's ISIGEN_ pattern)."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# isical/config.py -> parents[1] == repo root (isi_monitor3d/), so backbone.*,
# calibration.* and config/ all resolve relative to it.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_ISICAL_ROOT = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ISICAL_", env_file=None)

    host: str = "0.0.0.0"
    port: int = 8300
    data_dir: Path = Field(default=_ISICAL_ROOT / "data")
    runs_dir: Path = Field(default=_ISICAL_ROOT / "runs")
    # Where Export's "Install to live system" writes (the Mode-2 file the Backbone
    # + monitor_web dashboard load) and the backbone.yaml it stamps.
    repo_root: Path = Field(default=_REPO_ROOT)
    mode2_calibration_path: Path = Field(default=_REPO_ROOT / "config" / "mode2" / "calibration.json")
    backbone_config_path: Path = Field(default=_REPO_ROOT / "config" / "backbone.yaml")
    job_log_buffer: int = 1000
