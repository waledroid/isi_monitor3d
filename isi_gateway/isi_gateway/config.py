"""Runtime configuration — pydantic-settings, env-var overrideable.

All knobs are overrideable via ``ISI_GATEWAY_*`` environment variables.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All knobs for ``isi-gateway``. Override via ``ISI_GATEWAY_*`` env vars."""

    model_config = SettingsConfigDict(env_prefix="ISI_GATEWAY_", env_file=None)

    # HTTP server.
    host: str = "0.0.0.0"
    port: int = 8080

    # MQTT broker connection.
    mqtt_host: str = "127.0.0.1"
    mqtt_port: int = 1883
    mqtt_base: str = "isi"
    mqtt_tls: bool = False
    mqtt_ca_cert: str | None = None
    mqtt_tls_insecure: bool = False
    mqtt_username: str | None = None
    mqtt_password: str | None = None

    # Aggregation tunables.
    node_stale_after_s: float = 15.0
    passings_buffer: int = 200

    # Optional bearer-token auth (None = open).
    api_token: str | None = Field(
        default=None,
        description=(
            "When set, every route (except /healthz) requires "
            "'Authorization: Bearer <token>'. None means no auth."
        ),
    )
