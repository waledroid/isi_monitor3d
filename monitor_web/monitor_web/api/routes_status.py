"""Status + zones JSON endpoints."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import yaml
from backbone.shared.camera_rig import CameraRig
from backbone.shared.hardware import gpu_memory_mb, gpu_utilization_pct, host_memory_mb
from backbone.shared.zones import ZoneRegistry
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

from .. import dashboard_config
from ..detection_overlay import (
    latest_trained_onnx,
    latest_trained_openvino,
    read_backbone,
    resolve_model,
)
from ..link_lines import parse_link_lines, rules_to_dict
from .routes_calibrate import _configured_cameras, _mode_calibration_path
from .routes_video import _load_cameras_from_backbone_yaml, grab_real_frame

logger = logging.getLogger(__name__)

router = APIRouter()

# Camera-liveness probe is cached this long so a status poll doesn't re-open an
# idle source every few seconds. A probe reuses the shared hub stream (instant
# when a CAM tab is open), else briefly warms the source.
_CAM_LIVE_TTL_S = 8.0
_CAM_PROBE_TIMEOUT_S = 0.8


# ---- readiness / 3-state status light -------------------------------------
#
# RED   = blocked/broken: a precondition is missing, or the Backbone crashed.
# AMBER = ready, idle: all preconditions met + Backbone stopped → press START.
# GREEN = live: Backbone running AND ≥1 camera delivering real frames (strict).


def _config_ok(cfg) -> bool:
    data = read_backbone(cfg)
    return bool(data) and isinstance(data.get("cameras"), dict) and bool(data["cameras"])


def _model_ok(cfg) -> bool:
    det = read_backbone(cfg).get("detection") or {}
    plugin = det.get("plugin", "yolo_onnx")
    key = "model_xml" if plugin == "yolo_openvino" else "onnx_path"
    raw = det.get(key)
    if raw and resolve_model(raw, cfg) is not None:
        return True
    fallback = latest_trained_openvino() if plugin == "yolo_openvino" else latest_trained_onnx()
    return fallback is not None


def _sink_ok(cfg) -> bool:
    sinks = (read_backbone(cfg).get("metadata") or {}).get("sinks")
    return isinstance(sinks, list) and len(sinks) > 0


def _calibration_ok(cfg) -> bool:
    configured = _configured_cameras(cfg)
    if not configured:
        return False
    cal_path = _mode_calibration_path(cfg)
    if not cal_path.exists():
        return False
    try:
        rig = CameraRig.from_file(cal_path)
    except Exception:
        return False
    return set(configured) <= set(rig.camera_ids)


def _cameras_live(cfg, app) -> dict[str, bool]:
    """Per-camera liveness: is each configured camera delivering REAL frames (not
    the "connecting…" placeholder)? Cached per camera to bound probe cost — a probe
    reuses the warm hub stream (instant when a CAM/unified tab is open), else briefly
    warms the source. This is the live-feed signal (cam_b "configured" ≠ "streaming")
    that gates the unified tab + the degraded state."""
    cams = _load_cameras_from_backbone_yaml(cfg.backbone_config_path)
    cache = getattr(app.state, "cam_live_cache", None)
    if cache is None:
        cache = {}
        app.state.cam_live_cache = cache
    now = time.monotonic()
    out: dict[str, bool] = {}
    for cam_id, cam in cams.items():
        hit = cache.get(cam_id)
        if hit is not None and (now - hit[1]) < _CAM_LIVE_TTL_S:
            out[cam_id] = hit[0]
            continue
        try:
            live = grab_real_frame(cam_id, cam.get("source", {}),
                                   timeout=_CAM_PROBE_TIMEOUT_S) is not None
        except Exception:
            live = False
        cache[cam_id] = (live, now)
        out[cam_id] = live
    return out


def _compute_readiness(cfg, app, supervisor_state: str) -> dict:
    """Per-precondition booleans + the derived 3-state ``light``. Sync (does the
    camera probe); call via ``run_in_threadpool`` so it never blocks the loop."""
    cams_live = _cameras_live(cfg, app)
    any_live = any(cams_live.values())
    all_live = bool(cams_live) and all(cams_live.values())
    checks = {
        "config_ok": _config_ok(cfg),
        "camera_live": any_live,
        "model_ok": _model_ok(cfg),
        "calibration_ok": _calibration_ok(cfg),
        "sink_ok": _sink_ok(cfg),
    }
    ready = all(checks.values())
    degraded = False
    if supervisor_state == "running" and all_live:
        light = "green"               # live: process up + ALL cameras working (strict)
    elif supervisor_state == "running" and any_live:
        light = "amber"               # DEGRADED: running on the surviving camera(s)
        degraded = True               # e.g. Mode 2 with cam_b feed down → Track2D only
    elif supervisor_state == "crashed":
        light = "red"                 # crashed → fix it
    elif ready:
        light = "amber"               # all preconditions met, stopped → press START
    else:
        light = "red"                 # a precondition missing
    return {
        "light": light,
        "ready": ready,
        "degraded": degraded,
        "checks": checks,
        "cameras_live": cams_live,
    }


def _reprojection_px(cfg) -> dict[str, float]:
    """Per-camera homography reprojection RMS (px) from the current-mode calibration
    — the calibration-quality KPI (target ≤ 2 px). Static (set at calibration time);
    ``{}`` when no calibration exists."""
    cal_path = _mode_calibration_path(cfg)
    if not cal_path.exists():
        return {}
    try:
        data = json.loads(cal_path.read_text())
    except (OSError, ValueError):
        return {}
    out: dict[str, float] = {}
    for cam_id, c in (data.get("cameras") or {}).items():
        rms = c.get("reprojection_rms_px")
        if rms is not None:
            out[cam_id] = round(float(rms), 3)
    return out


def _resolve_zones_path(cfg) -> Path | None:
    """If the user pinned a zones_path explicitly, use it. Otherwise read it
    out of backbone.yaml if present. Returns None if no zones are configured."""
    if cfg.zones_path is not None:
        return cfg.zones_path
    if not cfg.backbone_config_path.exists():
        return None
    try:
        data = yaml.safe_load(cfg.backbone_config_path.read_text())
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(data, dict):
        return None
    raw = data.get("zones_path")
    return Path(raw) if raw else None


_RES_TTL_S = 1.5
_res_cache: dict = {"ts": 0.0, "data": None}


def _resources() -> dict:
    """Live host/GPU memory for the STATUS panel — VRAM + system RAM as used/total MB
    (+ GPU util %). Cached ~1.5 s so a 1 s status poll doesn't spawn nvidia-smi twice
    a second; runs in the threadpool (nvidia-smi is a subprocess)."""
    now = time.monotonic()
    if _res_cache["data"] is not None and now - _res_cache["ts"] < _RES_TTL_S:
        return _res_cache["data"]
    vram = gpu_memory_mb()        # (used, total) MB or None
    ram = host_memory_mb()        # (used, total) MB or None
    data = {
        "vram_used_mb": vram[0] if vram else None,
        "vram_total_mb": vram[1] if vram else None,
        "ram_used_mb": ram[0] if ram else None,
        "ram_total_mb": ram[1] if ram else None,
        "gpu_util_pct": gpu_utilization_pct(),
    }
    _res_cache.update(ts=now, data=data)
    return data


@router.get("/api/status")
async def status(request: Request) -> JSONResponse:
    cfg = request.app.state.settings
    bus = request.app.state.bus
    supervisor = request.app.state.supervisor
    snap = bus.snapshot()
    # 3-state status light (server-computed so the dot is the single source of truth).
    readiness = await run_in_threadpool(
        _compute_readiness, cfg, request.app, supervisor.state)
    resources = await run_in_threadpool(_resources)
    return JSONResponse(
        {
            "readiness": readiness,
            "resources": resources,
            "backbone": {
                "state": supervisor.state,
                "pid": supervisor.pid,
                "last_exit_code": supervisor.last_exit_code,
            },
            # Direction 1: the in-process perception producer (points mode).
            # {"running": False} in frames mode — harmless for old readers.
            "isistream": getattr(request.app.state, "isistream", None).status()
            if getattr(request.app.state, "isistream", None) is not None else {"running": False},
            "udp": {
                "received": snap.received,
                "dropped_malformed": snap.dropped_malformed,
                "dropped_version": snap.dropped_version,
                "last_envelope_ts": snap.last_envelope_ts,
                "fresh": bus.is_fresh(cfg.freshness_threshold_s),
                "freshness_threshold_s": cfg.freshness_threshold_s,
            },
            "tracks": {
                "active_2d": len(snap.last_track2d_by_id),
                "active_3d": len(snap.last_track3d_by_id),
            },
            # Live KPIs for the expanded STATUS panel. FPS comes from the
            # Backbone's diagnostics heartbeat (5 s interval) — hide it once
            # stale (> 2 heartbeats) so a stopped Backbone doesn't show a
            # frozen rate.
            "kpis": {
                # The KPI latency is the ENGINE's own capture→publish measure
                # (diagnostics heartbeat) — authoritative, unaffected by how
                # busy this web process is. The bus thread's processing delay
                # is reported separately as ui_lag (it's what makes panels
                # feel behind, but it is NOT pipeline latency).
                "latency_p50_ms": (
                    round((snap.engine_latency_ms or {}).get("p50"), 1)
                    if time.time() - snap.diagnostics_ts <= 12.0
                    and (snap.engine_latency_ms or {}).get("p50") is not None
                    else None),
                "latency_p95_ms": (
                    round((snap.engine_latency_ms or {}).get("p95"), 1)
                    if time.time() - snap.diagnostics_ts <= 12.0
                    and (snap.engine_latency_ms or {}).get("p95") is not None
                    else None),
                "latency_samples": snap.latency_samples,
                "latency_target_ms": 200,
                "ui_lag_p50_ms": snap.latency_p50_ms,
                # points mode: fps_by_camera is the DETECTION-SET rate, not
                # camera capture fps — the frontend relabels on this flag.
                "points_mode": bool(
                    getattr(request.app.state, "isistream", None)
                    and request.app.state.isistream.points_mode()),
                "reproj_rms_px": _reprojection_px(cfg),
                "reproj_target_px": 2.0,
                "fps_by_camera": (
                    snap.fps_by_camera
                    if time.time() - snap.diagnostics_ts <= 12.0 else {}
                ),
                "pipeline_fps": (
                    snap.pipeline_fps
                    if time.time() - snap.diagnostics_ts <= 12.0 else None
                ),
            },
        }
    )


@router.get("/api/zones")
async def zones(request: Request) -> JSONResponse:
    cfg = request.app.state.settings
    zones_path = _resolve_zones_path(cfg)
    if zones_path is None or not zones_path.exists():
        return JSONResponse({"zones": []})
    try:
        registry = ZoneRegistry.load(zones_path)
    except (ValueError, OSError) as exc:
        logger.warning("zones endpoint: %s", exc)
        return JSONResponse({"zones": [], "error": str(exc)}, status_code=500)
    return JSONResponse(
        {
            "zones": [
                {
                    "id": registry[n].id,   # stable identity (id != name)
                    "name": registry[n].name,
                    "type": registry[n].type,
                    "kind": registry[n].kind,
                    "severity": registry[n].severity,
                    "polygon": registry[n].polygon.tolist(),
                }
                for n in registry.names
            ]
        }
    )


@router.get("/api/zones/state")
async def zones_state(request: Request) -> JSONResponse:
    """Latest per-zone contents from the LOCAL UDP bus (``ZoneStateMessage``).

    Feeds the COMMUNICATION panel's zone cards when no gateway is configured:
    ``states`` maps zone name → the zone-state payload (objects with cls /
    confidence / occupancy, plus count and capture ts). ``fresh`` mirrors the
    bus-freshness gate the STATUS panel uses — stale bus ⇒ the cards dim.
    """
    bus = getattr(request.app.state, "bus", None)
    if bus is None:
        return JSONResponse({"fresh": False, "states": {}})
    snap = bus.snapshot()
    return JSONResponse({
        "fresh": bus.is_fresh(2.0),
        # The bus now keys its store by the STABLE zone_id, but this endpoint
        # keeps its name-keyed contract (the COMMUNICATION cards match floor
        # zones by name) — so key the output off each message's own ``zone``.
        "states": {
            msg.zone: {
                "objects": [o.model_dump(mode="json") for o in msg.objects],
                "count": msg.count,
                "ts": msg.ts,
            }
            for msg in snap.zone_state_by_zone.values()
        },
    })


@router.get("/api/danger-zones-object")
async def danger_zones_object(request: Request) -> JSONResponse:
    """Per-class proximity-ring radii (Type-1 danger zones — attached to objects).

    Schema:
        classes:
          custom-robot: { green_m: 3.0, yellow_m: 1.5, red_m: 0.5, alpha: 0.18 }
          forklift:     { green_m: 4.0, yellow_m: 2.0, red_m: 0.8, alpha: 0.18 }

    Missing file -> empty config; the dashboard then draws no per-object rings.
    """
    cfg = request.app.state.settings
    data = dashboard_config.read_section(cfg, "danger_zones_object")
    classes = data.get("classes", {})
    if not isinstance(classes, dict):
        return JSONResponse(
            {"classes": {}, "error": "top-level 'classes' must be a mapping"},
            status_code=500,
        )
    return JSONResponse({"classes": classes})


@router.get("/api/link-lines")
async def link_lines(request: Request) -> JSONResponse:
    """Per-class-pair distance-line rules (S16).

    Schema:
        rules:
          - from: person
            to: ["forklift", "palette", "robot"]   # or '*'
            max_distance_m: 5.0                    # optional
            color: "#ffffff"                       # optional

    Missing file -> empty rules; the floor map then draws no link lines.
    """
    cfg = request.app.state.settings
    try:
        rules = parse_link_lines(dashboard_config.read_section(cfg, "link_lines"))
    except ValueError as exc:
        logger.warning("link_lines endpoint: %s", exc)
        return JSONResponse({"rules": [], "error": str(exc)}, status_code=500)
    return JSONResponse({"rules": rules_to_dict(rules)})


# Re-exported so callers don't need to import Jinja2Templates.
__all__ = ["Jinja2Templates", "router"]
