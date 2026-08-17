"""FastAPI app factory — wires routes, lifespan, and shared state."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response

from .api import (
    routes_clients,
    routes_config,
    routes_diagnostics,
    routes_etagere,
    routes_health,
    routes_nodes,
    routes_passings,
    routes_tracks,
    routes_ui,
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
        recent_buffer=cfg.recent_buffer,
        node_evict_after_s=cfg.node_evict_after_s,
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

    # REST-consumer tracking (surfaced by /clients and the /ui Consumers
    # card): every API request is recorded per client, keyed by the optional
    # X-Client-Name header (AGVs are asked to send one) or client IP. Page
    # shells / docs / health are skipped — only data-endpoint traffic counts.
    # Touched only on the event loop (async middleware + async routes), so a
    # plain dict is race-free.
    app.state.api_clients = {}
    _untracked = {"/ui", "/test", "/docs", "/openapi.json",
                  "/favicon.ico", "/healthz", f"/{API_VERSION}/healthz"}
    _clients_cap = 100

    @app.middleware("http")
    async def _track_client(request: Request, call_next):
        if request.url.path not in _untracked:
            ip = request.client.host if request.client else "?"
            name = request.headers.get("x-client-name")
            key = name or ip
            store: dict[str, dict] = app.state.api_clients
            entry = store.get(key)
            if entry is None:
                if len(store) >= _clients_cap:
                    del store[min(store, key=lambda k: store[k]["last_seen"])]
                entry = store[key] = {"name": name, "ip": ip, "requests": 0}
            entry["last_seen"] = time.time()
            entry["requests"] += 1
            entry["last_path"] = request.url.path
        return await call_next(request)

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
        routes_etagere.router,
        routes_config.router,
        routes_clients.router,     # /clients — REST consumers + MQTT count
        routes_ui.router,          # /recent — the raw tail + ingest counters
    )
    version_prefix = f"/{API_VERSION}"
    for r in _resource_routers:
        app.include_router(r, prefix=version_prefix)
        app.include_router(r)  # bare alias

    # /healthz stays available un-prefixed (and also under /v1).
    app.include_router(routes_health.router)
    app.include_router(routes_health.router, prefix=version_prefix)

    # The probe page (/ui) — token-free SHELL, bare path only; its JS calls
    # the token-protected data endpoints above.
    app.include_router(routes_ui.page_router)

    return app
