"""Runtime configuration — pydantic-settings, env-var overrideable.

All knobs are overrideable via ``ISI_GATEWAY_*`` environment variables.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

API_VERSION = "v1"
"""REST API version prefix. All resource routers mount under ``/<API_VERSION>``
(e.g. ``/v1/nodes``); the bare paths (``/nodes``) remain as back-compat aliases.
Adding ``/v2`` later is a one-line extra include in ``app.py``."""


class Settings(BaseSettings):
    """All knobs for ``isi-gateway``. Override via ``ISI_GATEWAY_*`` env vars."""

    model_config = SettingsConfigDict(env_prefix="ISI_GATEWAY_", env_file=None)

    # HTTP server.
    host: str = "0.0.0.0"
    port: int = 8080

    # MQTT broker connection.
    mqtt_host: str = "127.0.0.1"
    mqtt_port: int = 1883
    mqtt_base: str = "isiMonitor3D"
    mqtt_tls: bool = False
    mqtt_ca_cert: str | None = None
    mqtt_tls_insecure: bool = False
    mqtt_username: str | None = None
    mqtt_password: str | None = None

    # Aggregation tunables.
    node_stale_after_s: float = 15.0
    # Nodes silent beyond this are EVICTED from the store entirely (vs merely
    # displayed stale). Long by design: a rig down for maintenance must survive
    # a workday; a decommissioned node ages out within a day. 0 disables.
    node_evict_after_s: float = 86400.0
    passings_buffer: int = 200
    # Raw-message ring buffer behind /recent and the /ui probe tail.
    recent_buffer: int = 300

    # Optional bearer-token auth (None = open).
    api_token: str | None = Field(
        default=None,
        description=(
            "When set, every route (except /healthz) requires "
            "'Authorization: Bearer <token>'. None means no auth."
        ),
    )
