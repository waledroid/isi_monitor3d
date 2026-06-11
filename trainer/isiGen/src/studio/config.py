"""Studio settings — pydantic-settings with the ISIGEN_ env prefix
(mirrors monitor_web's MONITOR_WEB_ pattern)."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_ISIGEN_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ISIGEN_", env_file=None)

    host: str = "0.0.0.0"
    port: int = 8200
    data_dir: Path = Field(default=_ISIGEN_ROOT / "data")
    runs_dir: Path = Field(default=_ISIGEN_ROOT / "runs")
    thumb_max_px: int = 256
    job_log_buffer: int = 1000
