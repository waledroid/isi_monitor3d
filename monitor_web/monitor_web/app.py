"""FastAPI app factory — wires routes, lifespan, and shared state."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
from backbone.shared.hardware import gpu_memory_mb
from fastapi import FastAPI, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .api import (
    routes_alignment,
    routes_calibrate,
    routes_cameras,
    routes_config,
    routes_control,
    routes_gateway,
    routes_logs,
    routes_map,
    routes_media,
    routes_pages,
    routes_projection,
    routes_status,
    routes_video,
    routes_ws,
    routes_ws_video,
    routes_zone_patches,
)
from .backbone_supervisor import BackboneSupervisor
from .bus_subscriber import BusSubscriber
from .camera_hub import get_hub
from .config import Settings
from .detection_overlay import current_model_info
from .zone_worker import ZoneWorkerManager

logger = logging.getLogger(__name__)

_BASE_DIR = Path(__file__).resolve().parent

# How often the terminal heartbeat prints the active detection model.
_HEARTBEAT_INTERVAL_S = 30


async def _heartbeat(cfg: Settings, supervisor: BackboneSupervisor) -> None:
    """Periodically log the detection model in use so the terminal shows what the
    system is actually running. Reports both the model loaded by the live preview
    overlay and the one configured in backbone.yaml — when they differ, a model
    change is pending a stream reconnect (overlay) or a Backbone STOP/START."""
    while True:
        try:
            info = current_model_info(cfg)
            loaded = info["loaded"]
            conf = info["configured"]
            # The dashboard runs no perception of its own (isistream is the single
            # source). ``loaded`` is non-None only after the MP4 dev viewer has run
            # its in-process detector; "none" is the normal live state.
            if loaded:
                loaded_str = f"{loaded['plugin']} · {loaded['label']}"
            else:
                loaded_str = "none (no preview run yet)"
            conf_str = f"{conf['plugin']} · {conf['label']}"
            if conf["path"] and not conf["resolved"]:
                conf_str += " (UNRESOLVED!)"
            # The configured path is already absolutized by current_model_info when
            # resolvable, so a direct compare against the loaded path is valid.
            pending = bool(loaded and conf["path"] and loaded["path"] != conf["path"])
            mem = gpu_memory_mb()
            gpu_str = (
                f" | gpu={mem[0]}/{mem[1]} MB ({mem[1] - mem[0]} free)" if mem else ""
            )
            logger.info(
                "heartbeat | backbone=%s | preview model=%s | configured=%s%s%s",
                supervisor.state,
                loaded_str,
                conf_str,
                "  ⟵ change pending reload" if pending else "",
                gpu_str,
            )
        except Exception as exc:  # never let the heartbeat kill the app
            logger.warning("heartbeat failed: %s", exc)
        await asyncio.sleep(_HEARTBEAT_INTERVAL_S)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build a configured FastAPI app instance."""
    cfg = settings or Settings()

    broadcast_queue: asyncio.Queue = asyncio.Queue(maxsize=512)
    bus = BusSubscriber(cfg.udp_host, cfg.udp_port, broadcast_queue=broadcast_queue)
    supervisor = BackboneSupervisor(
        config_path=cfg.backbone_config_path,
        terminate_timeout_s=cfg.backbone_terminate_timeout_s,
        log_buffer_size=cfg.log_buffer_size,
    )
    # Direction 1: when backbone.yaml says ingestion.mode: points, the
    # dashboard hosts the perception producer (hub-backed, one decode per
    # camera) for the metric engine. Started/stopped with the Backbone by
    # the control routes; a no-op in frames mode.
    from .isistream_host import IsistreamHost
    perception = IsistreamHost(cfg.backbone_config_path)
    # Background zone detection: one worker thread per camera with zones, publishing
    # one coherent snapshot per frame. Panels + cam views are pure renderers of it.
    zone_manager = ZoneWorkerManager(
        cfg, is_running=lambda: supervisor.state == "running",
        # Late-binding: app.state.bus is attached below; the workers render the
        # Backbone's per-camera observations from it — ONE perception, zero
        # dashboard inference.
        bus_getter=lambda: getattr(app.state, "bus", None))
    templates = Jinja2Templates(directory=str(_BASE_DIR / "templates"))

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # Every MJPEG stream is a SYNC generator, so Starlette pumps each one in the
        # AnyIO worker-thread pool — one thread held per open stream for its lifetime.
        # The default pool is only 40 threads, shared with every `run_in_threadpool`
        # call (/api/status, config saves, control). A handful of accumulated stream
        # connections could saturate it and freeze the whole app until the tab closed.
        # Raise the ceiling so streams can never starve the API/control endpoints.
        anyio.to_thread.current_default_thread_limiter().total_tokens = 256
        # Reap any Backbone orphaned by a previous dashboard that died without a
        # clean STOP (e.g. OOM-killed). Otherwise the stray (~1.5 GB) survives until
        # the operator next presses START and can OOM-kill THIS dashboard first.
        supervisor.reap_orphans_on_boot()
        bus.attach_loop(asyncio.get_running_loop())
        bus.start()
        zone_manager.start()      # no-op when no zones are configured
        heartbeat = asyncio.create_task(_heartbeat(cfg, supervisor))
        try:
            yield
        finally:
            heartbeat.cancel()
            bus.stop()
            zone_manager.stop()   # before hub shutdown: workers release their streams
            perception.stop()     # before hub shutdown: it holds hub readers
            supervisor.stop()
            get_hub().shutdown()      # release any open camera sessions

    app = FastAPI(
        title="ISI Monitor 3D — Operator Dashboard",
        version="0.0.1",
        lifespan=lifespan,
    )

    # Make shared singletons accessible to route handlers.
    app.state.settings = cfg
    app.state.bus = bus
    app.state.supervisor = supervisor
    app.state.isistream = perception
    app.state.zone_manager = zone_manager
    app.state.templates = templates
    app.state.broadcast_queue = broadcast_queue

    # Browsers auto-request /favicon.ico on every page load; the app ships no icon,
    # so it 404s and clutters the log. Answer 204 (No Content) to silence it cheaply.
    @app.get("/favicon.ico", include_in_schema=False)
    async def _favicon() -> Response:
        return Response(status_code=204)

    app.mount("/static", StaticFiles(directory=str(_BASE_DIR / "static")), name="static")

    # Automatic cache-busting for static assets. `no-cache` lets the browser
    # keep a copy but forces it to REVALIDATE every time (StaticFiles answers the
    # conditional request with a cheap 304 when unchanged, 200 with fresh bytes
    # when edited). This picks up any CSS/JS change — including ES-module internal
    # imports that a `?v=` query on <script> tags would miss — with no hard
    # refresh, which is exactly the "I changed it but the browser shows old" trap.
    @app.middleware("http")
    async def _revalidate_static(request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    app.include_router(routes_pages.router)
    app.include_router(routes_status.router)
    app.include_router(routes_gateway.router)
    app.include_router(routes_config.router)
    app.include_router(routes_cameras.router)
    app.include_router(routes_media.router)
    app.include_router(routes_logs.router)
    app.include_router(routes_control.router)
    app.include_router(routes_projection.router)
    app.include_router(routes_calibrate.router)
    app.include_router(routes_alignment.router)
    app.include_router(routes_video.router)
    app.include_router(routes_map.router)
    app.include_router(routes_ws.router)
    app.include_router(routes_ws_video.router)
    app.include_router(routes_zone_patches.router)

    return app
