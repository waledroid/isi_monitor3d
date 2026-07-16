"""FastAPI app factory — wires routes, lifespan, and shared state."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response

from .api import (
    routes_config,
    routes_diagnostics,
    routes_health,
    routes_nodes,
    routes_passings,
    routes_tracks,
    routes_zones,
)
from .config import API_VERSION, Settings
from .mqtt_subscriber import MqttSubscriber

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build a configured FastAPI app instance.

    ``settings`` may be omitted in tests — ``Settings()`` is constructed
    automatically (reads ``ISI_GATEWAY_*`` env vars).
    """
    cfg = settings or Settings()

    subscriber = MqttSubscriber(
        host=cfg.mqtt_host,
        port=cfg.mqtt_port,
        base=cfg.mqtt_base,
        tls=cfg.mqtt_tls,
        ca_cert=cfg.mqtt_ca_cert,
        tls_insecure=cfg.mqtt_tls_insecure,
        username=cfg.mqtt_username,
        password=cfg.mqtt_password,
        passings_buffer=cfg.passings_buffer,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        subscriber.start()
        try:
            yield
        finally:
            subscriber.stop()

    app = FastAPI(
        title="ISI Gateway",
        description="Central MQTT aggregator for distributed ISI Monitor 3D nodes",
        version="0.0.1",
        lifespan=lifespan,
    )

    # Shared singletons for route handlers.
    app.state.settings = cfg
    app.state.subscriber = subscriber

    # Silence favicon noise.
    @app.get("/favicon.ico", include_in_schema=False)
    async def _favicon() -> Response:
        return Response(status_code=204)

    # Resource routers are mounted twice: under the versioned prefix
    # (``/v1/nodes`` …) and bare (``/nodes`` …) as back-compat aliases so the
    # monitor_web proxy and existing consumers keep working during transition.
    # Adding ``/v2`` later is one extra include line per router.
    _resource_routers = (
        routes_nodes.router,
        routes_tracks.router,
        routes_diagnostics.router,
        routes_passings.router,
        routes_zones.router,
        routes_config.router,
    )
    version_prefix = f"/{API_VERSION}"
    for r in _resource_routers:
        app.include_router(r, prefix=version_prefix)
        app.include_router(r)  # bare alias

    # /healthz stays available un-prefixed (and also under /v1).
    app.include_router(routes_health.router)
    app.include_router(routes_health.router, prefix=version_prefix)

    return app
