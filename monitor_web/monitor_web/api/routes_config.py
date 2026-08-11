"""Operator-facing config endpoints — cameras + zones, atomic-write to YAML.

The zone manager overlay (``static/js/zone_manager.js``) reads the current
camera URLs + zones via ``GET /api/config`` and posts edits via ``POST
/api/config``. Writes are atomic (tempfile + ``os.replace``) and only target
the two paths configured in ``Settings``: ``backbone_config_path`` and the
``zones_path`` resolved out of it.

The Backbone reads ``backbone.yaml`` once at orchestrator boot — camera URL
changes require a STOP/START cycle to apply. ``zones.yaml`` is hot-loaded by
the dashboard's ``/api/zones`` route on every request, so zone edits surface
instantly on the floor map without restarting anything.
"""

from __future__ import annotations

import ast
import functools
import logging
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Literal

import yaml
from backbone.shared.hardware import (
    gpu_available,  # consumer-side helper (like backbone.shared.zones)
)
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from .. import dashboard_config
from ..detection_overlay import (
    latest_pose_onnx,
    latest_trained_onnx,
    latest_trained_openvino,
    list_pose_onnx,
    list_trained_onnx,
    select_plugin,
)
from ..link_lines import LinkLineRule, parse_link_lines, rules_to_dict
from ..overlay import CLASS_COLORS_HEX, DEFAULT_CLASS_COLOR_HEX

logger = logging.getLogger(__name__)

router = APIRouter()


def _detect_backend() -> str:
    """Server-decided detector backend: GPU host → ONNX (CUDA); CPU-only → OpenVINO."""
    return "yolo_onnx" if gpu_available() else "yolo_openvino"


def _ensure_launchable(backbone_data: dict, cfg) -> None:
    """Fill in the required-but-not-UI-edited keys so the written backbone.yaml is
    actually launchable by the orchestrator. Without this, a config saved purely
    from the Settings modal is missing ``metadata.sinks`` (the orchestrator hard-
    refuses to start) and ``calibration_path`` — the two reasons START used to die
    instantly. Only fills gaps; never overwrites operator-set values.
    """
    # metadata.sinks — at least one sink is mandatory. Default to a UDP sink
    # pointing at the dashboard's own listener so tracks reach the floor map.
    meta = backbone_data.get("metadata")
    if not isinstance(meta, dict):
        meta = {}
    sinks = meta.get("sinks")
    if not isinstance(sinks, list) or not sinks:
        meta["sinks"] = [{"plugin": "udp", "host": cfg.udp_host, "port": cfg.udp_port}]
        backbone_data["metadata"] = meta

    # calibration_path — point at the CURRENT mode's calibration file (Mode 1 vs
    # Mode 2 keep separate files, re-applied per camera count). Mode is derived
    # from the cameras being written NOW (in-memory), not the stale on-disk file,
    # so a mode switch repoints correctly in the same save. The file may not exist
    # yet (calibration still to be run); caught with a clear message at START.
    # Mode 2 honours the operator's path override (``mode2_calibration_path`` UI
    # setting, set via the calibration picker) — e.g. an isical-Studio artefact —
    # so a Settings save must not stomp it back to the managed default.
    cams = backbone_data.get("cameras", {})
    n_cams = len(cams) if isinstance(cams, dict) else 0
    mode = 1 if n_cams <= 1 else 2
    cal = Path(cfg.backbone_config_path).resolve().parent / f"mode{mode}" / "calibration.json"
    if mode == 2:
        override = str(_read_ui_settings(cfg).get("mode2_calibration_path") or "").strip()
        if override:
            cal = Path(override)
    backbone_data["calibration_path"] = str(cal)


MAX_ZONES = 6
# Zone categories: palette (neutral), etagere = "étagère" (light green),
# danger (light red). Stored ascii; the UI shows the accented French labels.
ALLOWED_KINDS = ("palette", "etagere", "danger")
ALLOWED_SEVERITIES = ("info", "warning", "critical")
# Camera slots the dashboard owns (fixed Cam 1 / Cam 2). A POST that omits one
# of these means "remove it"; cameras outside this set (hand-added to
# backbone.yaml) are never touched by the dashboard writer.
MANAGED_CAMERA_SLOTS = ("cam_a", "cam_b")

# ---- 3D localization (triangulation subscriptions) ----
# Fixed rule shape written per checked class: both cameras must see the object
# (Mode 2) at a modest rate — the 3D-localized classes move slowly (pallets etc.).
LOC3D_DEFAULT_RATE_HZ = 5.0
LOC3D_CAMERAS_SEEING_MIN = 2
# person never reaches triangulation (the orchestrator excludes person
# detections from detections_by_camera) — a person rule would be dead code.
LOC3D_UNSUPPORTED = {"person"}
MAX_LOC3D_CLASSES = 32
# Class names feed rule names ("<cls>_3d") and come from model metadata —
# keep them to a safe identifier charset.
_LOC3D_CLS_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class CameraConfig(BaseModel):
    """One camera's source. Exactly one of ``url`` (RTSP) or ``device`` (V4L2)
    must be set. To remove a camera, omit it from the ``cameras`` mapping."""

    url: str | None = None
    device: str | None = None

    @model_validator(mode="after")
    def exactly_one_source(self) -> CameraConfig:
        has_url = bool(self.url and self.url.strip())
        has_device = bool(self.device and self.device.strip())
        if has_url == has_device:   # both set OR neither set
            raise ValueError("camera needs exactly one of 'url' (RTSP) or 'device' (V4L2)")
        return self


class ZoneConfig(BaseModel):
    # Immutable identity, generated once on the client at creation. External
    # systems key on it, so it must survive renames + never renumber on delete.
    # None ⇒ legacy zone; the Backbone's loader derives an id from the name.
    id: str | None = None
    name: str = Field(..., min_length=1)
    type: str = "palette"
    kind: Literal["palette", "etagere", "danger"] = "palette"
    severity: Literal["info", "warning", "critical"] = "info"
    polygon: list[tuple[float, float]] = Field(..., min_length=3)
    # Per-zone detection config (Phase 3). The Backbone ignores these (its
    # ZoneRegistry only reads name/type/polygon/kind/severity), so they're safe to
    # store in zones.yaml; the dashboard uses them. None = use the global model/conf.
    model: str | None = None
    confidence_threshold: float | None = None
    # Height of the zone's plane above the floor, metres (platform/shelf zones).
    # Absent = 0.0, matching the Backbone loader's default (Zone.z_base_m) —
    # an older UI that omits the field keeps writing floor zones.
    z_base_m: float = 0.0

    @field_validator("z_base_m")
    @classmethod
    def _z_base_range(cls, v: float) -> float:
        if not (0.0 <= float(v) <= 5.0):   # NaN also fails this → rejected
            raise ValueError("z_base_m must be between 0 and 5 metres")
        return float(v)

    @field_validator("polygon")
    @classmethod
    def finite_coords(cls, v: list[tuple[float, float]]) -> list[tuple[float, float]]:
        for x, y in v:
            if not (math.isfinite(x) and math.isfinite(y)):
                raise ValueError("polygon coords must be finite")
        return v

    @field_validator("confidence_threshold")
    @classmethod
    def _conf_range(cls, v: float | None) -> float | None:
        return None if v is None else max(0.0, min(1.0, float(v)))


class DetectionConfig(BaseModel):
    """The Backbone's detection model — written to backbone.yaml's `detection`
    block (drives both the live Backbone and the MP4 viewer). The active backend
    is decided by the server from hardware (GPU → yolo_onnx, CPU-only →
    yolo_openvino), NOT by the client; the inactive path is remembered in the
    UI-settings store so the modal can repopulate it."""

    onnx_path: str | None = None
    model_xml: str | None = None
    # Optional person-POSE ONNX (handled by a separate pose model in the design;
    # gives ankle foot nodes). Written to backbone.yaml's detection block as
    # `pose_onnx_path` and remembered in UI-settings. Empty/None = no pose model.
    pose_enabled: bool = True
    pose_onnx_path: str | None = None
    # Person-pose detection confidence (the separate pose engine's `conf`).
    pose_confidence_threshold: float = 0.3
    class_names: list[str] = Field(..., min_length=1)
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    # Backbone-side: decode instance masks so the wire observations carry
    # mask POLYGONS (boxes are always published). Costs CPU per detection in
    # the Backbone; applies at its next START.
    decode_masks: bool = False
    # Model input size (the Settings slider). Only effective on a DYNAMIC-exported
    # ONNX; a fixed export ignores it (the detector adopts the model's own size).
    # Smaller = faster, less accurate on far/small objects.
    inference_imgsz: int = 1024
    # Dashboard-only: draw a white circle at each bbox's bottom-center (foot
    # node) on the live preview + MP4 viewer. NOT written to backbone.yaml —
    # stored in the UI-settings YAML (it doesn't affect the live Backbone).
    show_nodes: bool = True
    # Dashboard-only: blend the segmentation mask overlay (seg detectors only;
    # detect detectors have no mask). Also stored in UI-settings YAML.
    show_masks: bool = True
    # Dashboard-only: draw the detection bounding box. Off ⇒ mask + class label
    # only (cleaner with a seg model). Stored in the UI-settings YAML.
    show_boxes: bool = True
    # Dashboard-only: cap the per-frame inference/compositing rate on display
    # streams (CAM detect, zone patches, unified). Stored in the UI-settings YAML.
    # Dashboard-only: person↔pallet distance-line look (UI-settings YAML).
    distance_line_opacity: float = 0.25
    distance_line_color: str = "#ffffff"
    distance_line_thickness: int = 2


    @field_validator("distance_line_opacity")
    @classmethod
    def _opacity_range(cls, v: float) -> float:
        return max(0.05, min(1.0, float(v)))

    @field_validator("distance_line_thickness")
    @classmethod
    def _thickness_range(cls, v: int) -> int:
        return max(1, min(8, int(v)))

    @field_validator("inference_imgsz")
    @classmethod
    def _imgsz_allowed(cls, v: int) -> int:
        allowed = {320, 448, 512, 640, 1024}
        if v not in allowed:
            raise ValueError(f"inference_imgsz must be one of {sorted(allowed)}")
        return v

    @model_validator(mode="after")
    def at_least_one_path(self) -> DetectionConfig:
        if not ((self.onnx_path and self.onnx_path.strip())
                or (self.model_xml and self.model_xml.strip())):
            raise ValueError("a model path (onnx_path or model_xml) is required")
        return self


class PosePayload(BaseModel):
    """The Settings ▸ Isistream section.

    isistream is the single perception: ONE object model, zone-scoped and
    batched, serves every zone of every camera — so the object model, its
    inference size, confidence, SAHI and enhancement are GLOBAL here (there
    are no per-zone models any more). All of these are spliced into
    backbone.yaml's ``detection`` block and hot-applied by restarting the
    producer. Empty path = clear.
    """

    pose_enabled: bool = True
    pose_onnx_path: str = ""
    pose_confidence_threshold: float = 0.3
    # Global object model (all zones). Empty = leave the configured one.
    onnx_path: str = ""
    zone_imgsz: int | None = None
    confidence_threshold: float | None = None
    sahi_enabled: bool | None = None
    sahi_tile: int | None = None
    sahi_overlap: float | None = None
    enhance_enabled: bool | None = None
    enhance_gamma: float | None = None


class MqttSinkConfig(BaseModel):
    """MQTT sink configuration — spliced into backbone.yaml's metadata.sinks."""

    host: str = ""
    port: int = 1883
    tls: bool = False
    ca_cert: str = ""
    username: str = ""
    password: str = ""
    prefix: str = ""


class ConfigPayload(BaseModel):
    cameras: dict[str, CameraConfig]
    # Metric floor-zones (zones.yaml). Now OPTIONAL: the Settings metric-zone editor
    # was retired (operator zones are drawn on the cam → zone_patches). Omitted ==
    # "leave zones.yaml untouched"; an explicit list still rewrites it.
    zones: list[ZoneConfig] | None = Field(default=None, max_length=MAX_ZONES)
    detection: DetectionConfig | None = None
    # Pose-only update (the current Settings modal). Splices ONLY the pose keys
    # into backbone.yaml's detection block, never the object-model keys.
    pose: PosePayload | None = None
    # Backbone mask decode (the "Segmentation masks" switch): observations carry
    # mask polygons when on. None = leave untouched; applies at backbone START.
    decode_masks: bool | None = None
    # isistream perf toggles (both default ON in the producer). None = leave
    # untouched; a save hot-restarts a running producer so they apply live.
    motion_gate: bool | None = None
    detect_substream: bool | None = None
    # Distance-line rules (S16). Omitted == "no change"; an empty list explicitly
    # clears all rules on disk (the dashboard sends this when the operator
    # deletes every rule).
    link_lines: list[LinkLineRule] | None = None
    # Communication — node identity + MQTT broker. Both None = leave untouched.
    node_id: str | None = None
    mqtt_sink: MqttSinkConfig | None = None
    # Camera FPS — written to every camera's source.capture_fps in backbone.yaml.
    # Controls the shared camera-hub rate, which sets the cam-view VIDEO frame rate.
    # None = leave the existing per-camera values untouched.
    camera_fps: int | None = None
    # 3D localization — class names to triangulate (regenerates subscriptions.yaml
    # wholesale). None = leave the file untouched; [] explicitly clears every rule.
    localization_3d: list[str] | None = None

    @field_validator("camera_fps")
    @classmethod
    def _camera_fps_range(cls, v: int | None) -> int | None:
        if v is None:
            return None
        return max(1, min(30, int(v)))

    @field_validator("localization_3d")
    @classmethod
    def _loc3d_classes(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return None
        out: list[str] = []
        for name in v:
            name = (name or "").strip()
            if not name or name in LOC3D_UNSUPPORTED:
                continue   # empties + person silently dropped, not an error
            if not _LOC3D_CLS_RE.match(name):
                raise ValueError(f"invalid class name: {name!r}")
            if name not in out:   # dedupe, order preserved
                out.append(name)
        return out[:MAX_LOC3D_CLASSES]


# ---- path resolution helpers ----


def _resolve_zones_path(cfg, backbone_data: dict[str, Any] | None) -> Path | None:
    """Mirror of routes_status._resolve_zones_path but accepts a pre-parsed
    backbone.yaml body (we already read it for the GET handler)."""
    if cfg.zones_path is not None:
        return Path(cfg.zones_path)
    if backbone_data is None:
        return None
    raw = backbone_data.get("zones_path")
    return Path(raw) if raw else None


def _resolve_subscriptions_path(backbone_data: dict[str, Any], backbone_path: Path) -> Path:
    """subscriptions.yaml location — ``subscriptions_path`` from backbone.yaml when
    set, else the default beside it (``mode2/subscriptions.yaml``). The default is
    RECORDED into ``backbone_data`` (mirrors the zones_path self-heal in
    ``post_config``) so the Backbone and the UI agree on the same file."""
    raw = backbone_data.get("subscriptions_path")
    if raw:
        return Path(raw)
    path = backbone_path.parent / "mode2" / "subscriptions.yaml"
    backbone_data["subscriptions_path"] = str(path)
    return path


def _read_localization_3d(subs_path: Path) -> list[str]:
    """Class names currently subscribed for 3D: rules with ``request: xyz`` and a
    ``match.cls``, file order, deduped. The dead person path is hidden. A
    missing/unreadable file is an empty selection, never an error."""
    if not subs_path.exists():
        return []
    try:
        data = yaml.safe_load(subs_path.read_text()) or []
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("localization_3d: %s unreadable: %s", subs_path, exc)
        return []
    if not isinstance(data, list):
        logger.warning("localization_3d: %s: top-level must be a list", subs_path)
        return []
    out: list[str] = []
    for entry in data:
        if not isinstance(entry, dict) or entry.get("request") != "xyz":
            continue
        match = entry.get("match")
        cls_name = match.get("cls") if isinstance(match, dict) else None
        if not cls_name or cls_name in LOC3D_UNSUPPORTED:
            continue
        if cls_name not in out:
            out.append(cls_name)
    return out


def _localization_3d_doc(classes: list[str]) -> list[dict[str, Any]]:
    """Deterministic subscriptions.yaml body for the operator's class selection.

    The file becomes fully UI-OWNED: every save regenerates it wholesale, so
    hand edits (``in_zone`` predicates, ``rate_hz`` tweaks, person rules) are
    dropped. person is filtered because it never reaches triangulation — the
    orchestrator excludes person detections before the subscription filter.
    """
    return [
        {
            "name": f"{cls_name}_3d",
            "module": "palettes",
            "match": {"cls": cls_name, "cameras_seeing_min": LOC3D_CAMERAS_SEEING_MIN},
            "request": "xyz",
            "rate_hz": LOC3D_DEFAULT_RATE_HZ,
        }
        for cls_name in classes
        if cls_name not in LOC3D_UNSUPPORTED
    ]


def _read_backbone(cfg) -> dict[str, Any]:
    path = Path(cfg.backbone_config_path)
    if not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise HTTPException(status_code=500, detail=f"backbone.yaml unreadable: {exc}") from exc


def _read_zones(zones_path: Path | None) -> list[dict[str, Any]]:
    if zones_path is None or not zones_path.exists():
        return []
    try:
        data = yaml.safe_load(zones_path.read_text()) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise HTTPException(status_code=500, detail=f"zones.yaml unreadable: {exc}") from exc
    raw = data.get("zones", []) if isinstance(data, dict) else []
    return raw if isinstance(raw, list) else []


def _write_yaml_atomic(path: Path, data: dict[str, Any] | list[Any]) -> None:
    """Write ``data`` to ``path`` atomically (tmp in same dir + os.replace).

    A crash between the write and the rename leaves the previous file intact;
    a crash after the rename has the new file fully on disk. The only failure
    mode that loses data is a hardware failure between the rename and the next
    fsync — outside our envelope.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # delete=False so we keep the temp file across the close; we replace it ourselves.
    with tempfile.NamedTemporaryFile(
        "w",
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        yaml.safe_dump(data, tmp, sort_keys=False, allow_unicode=True)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    os.replace(tmp_name, path)


def _read_ui_settings(cfg) -> dict[str, Any]:
    path = Path(cfg.ui_settings_path)
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}


def _merge_ui_settings(cfg, patch: dict[str, Any]) -> dict[str, Any]:
    # Strict read-modify-write via the unified store: an existing-but-unreadable
    # file must raise (→ HTTP 500), never merge into {} and overwrite — that
    # exact path once wiped every zone/calibration override/preference.
    current = dashboard_config._load_all_or_none(cfg)
    if current is None:
        raise dashboard_config.StoreCorrupt(
            f"{cfg.ui_settings_path} exists but is unreadable — fix or remove it "
            f"(a .bak of the last good write sits next to it)")
    current.update(patch or {})
    dashboard_config.write_all(cfg, current)   # atomic + keeps a .bak
    return current


# ---- handlers ----


@router.get("/api/config")
# Sync `def` ON PURPOSE (here and for the other read handlers below): these do
# filesystem walks / YAML reads / ONNX metadata loads. As `async def` they ran
# that blocking I/O ON the event loop — one cold model-list walk froze every
# request for its whole duration (measured 20+ s Settings opens). Sync handlers
# run in the threadpool; the loop stays responsive.
def get_config(request: Request) -> JSONResponse:
    """Return the editable subset of the Backbone config + the current zones."""
    cfg = request.app.state.settings
    backbone_data = _read_backbone(cfg)
    cameras_raw = backbone_data.get("cameras", {}) if isinstance(backbone_data, dict) else {}
    cameras_out: dict[str, dict[str, str | None]] = {}
    _camera_fps_found: int | None = None
    if isinstance(cameras_raw, dict):
        for cam_id, body in cameras_raw.items():
            name, url, device = "rtsp", "", None
            if isinstance(body, dict):
                src = body.get("source", {})
                if isinstance(src, dict):
                    name = src.get("name", "rtsp") or "rtsp"
                    url = src.get("url", "") or ""
                    device = src.get("device")
                    # Read camera_fps from any camera's source (all should be equal).
                    if _camera_fps_found is None and src.get("capture_fps") is not None:
                        try:
                            _camera_fps_found = int(src["capture_fps"])
                        except (TypeError, ValueError):
                            pass
            cameras_out[cam_id] = {"name": name, "url": url, "device": device}
    camera_fps_out = _camera_fps_found if _camera_fps_found is not None else 20

    zones_path = _resolve_zones_path(cfg, backbone_data)
    zones_raw = _read_zones(zones_path)
    zones_out: list[dict[str, Any]] = []
    for z in zones_raw:
        if not isinstance(z, dict):
            continue
        try:
            z_base_m = float(z.get("z_base_m") or 0.0)
        except (TypeError, ValueError):
            # A hand-edited zones.yaml can carry a non-numeric z_base_m —
            # default to the floor rather than 500ing the whole config read.
            z_base_m = 0.0
        zones_out.append(
            {
                "id": z.get("id"),
                "name": z.get("name", ""),
                "type": z.get("type", "palette"),
                "kind": z.get("kind", "palette"),
                "severity": z.get("severity", "info"),
                "polygon": z.get("polygon", []),
                "model": z.get("model"),
                "confidence_threshold": z.get("confidence_threshold"),
                "z_base_m": z_base_m,
            }
        )

    # detection: active model from backbone.yaml, with the inactive path filled
    # from the UI-settings memory so the modal can show both.
    det_raw = backbone_data.get("detection", {}) if isinstance(backbone_data, dict) else {}
    if not isinstance(det_raw, dict):
        det_raw = {}
    ui = _read_ui_settings(cfg)
    # Default the model paths to the latest trained export so the modal pre-fills
    # a real, resolvable path (operator just clicks Save).
    detection_out = {
        # backend is hardware-decided (not the saved plugin) so the modal shows
        # the right field for THIS host.
        "backend": _detect_backend(),
        "onnx_path": (det_raw.get("onnx_path") or ui.get("model_onnx_path")
                      or latest_trained_onnx() or ""),
        "model_xml": (det_raw.get("model_xml") or ui.get("model_xml_path")
                      or latest_trained_openvino() or ""),
        # Person-pose model (separate ONNX). Pre-fill from the saved config /
        # UI memory / newest pose export so the dropdown shows a real choice.
        "decode_masks": bool(det_raw.get("decode_masks", False)),
        "zone_imgsz": int(det_raw.get("zone_imgsz", 384) or 384),
        "sahi_enabled": bool((det_raw.get("sahi") or {}).get("enabled", False)),
        "sahi_tile": int((det_raw.get("sahi") or {}).get("tile", 0) or 0),
        "sahi_overlap": float((det_raw.get("sahi") or {}).get("overlap", 0.2)),
        "enhance_enabled": bool((det_raw.get("enhance") or {}).get("enabled", False)),
        "enhance_gamma": float((det_raw.get("enhance") or {}).get("gamma", 1.0)),
        "pose_enabled": bool(det_raw.get("pose_enabled", True)),
        "pose_onnx_path": (det_raw.get("pose_onnx_path") or ui.get("pose_onnx_path")
                           or latest_pose_onnx() or ""),
        "pose_confidence_threshold": det_raw.get("pose_confidence_threshold", 0.3),
        "class_names": det_raw.get("class_names") or ["palette_vide"],
        "confidence_threshold": det_raw.get("confidence_threshold", 0.25),
        "iou_threshold": det_raw.get("iou_threshold", 0.45),
        "inference_imgsz": det_raw.get("inference_imgsz", 1024),
        "show_nodes": bool(ui.get("show_nodes", True)),
        "show_masks": bool(ui.get("show_masks", True)),
        "show_floor_zones": bool(ui.get("show_floor_zones", False)),
        "show_zone_fill": bool(ui.get("show_zone_fill", False)),
        "show_boxes": bool(ui.get("show_boxes", True)),
        "distance_line_opacity": float(ui.get("distance_line_opacity", 0.25)),
        "distance_line_color": str(ui.get("distance_line_color", "#ffffff")),
        "distance_line_thickness": int(ui.get("distance_line_thickness", 2)),
    }

    try:
        link_lines_rules = parse_link_lines(dashboard_config.read_section(cfg, "link_lines"))
    except ValueError as exc:
        # Bad on-disk file shouldn't break the modal — surface as empty + log.
        logger.warning("link_lines: %s", exc)
        link_lines_rules = []

    # ---- communications: node_id + mqtt_sink ----
    node_id_out = backbone_data.get("node_id", "") if isinstance(backbone_data, dict) else ""
    sinks_raw = []
    if isinstance(backbone_data, dict):
        meta_raw = backbone_data.get("metadata")
        if isinstance(meta_raw, dict):
            sinks_raw = meta_raw.get("sinks") or []
    mqtt_raw = next((s for s in sinks_raw if isinstance(s, dict) and s.get("plugin") == "mqtt"), None)
    if mqtt_raw:
        mqtt_sink_out = {
            "host": mqtt_raw.get("host", ""),
            "port": int(mqtt_raw.get("port", 1883)),
            "tls": bool(mqtt_raw.get("tls", False)),
            "ca_cert": mqtt_raw.get("ca_cert", ""),
            "username": mqtt_raw.get("username", ""),
            "password": mqtt_raw.get("password", ""),
            "prefix": mqtt_raw.get("prefix", ""),
        }
    else:
        mqtt_sink_out = {
            "host": "", "port": 1883, "tls": False,
            "ca_cert": "", "username": "", "password": "", "prefix": "",
        }

    return JSONResponse(
        {
            "cameras": cameras_out,
            "zones": zones_out,
            "detection": detection_out,
            "isistream": {
                "motion_gate": bool((backbone_data.get("isistream") or {}).get("motion_gate", True)),
                "detect_substream": bool(
                    (backbone_data.get("isistream") or {}).get("detect_substream", True)),
                # Substream URLs are configured per camera (detect_source); the
                # toggle is inert until at least one camera has one.
                "has_detect_source": any(
                    isinstance(c, dict) and c.get("detect_source")
                    for c in (backbone_data.get("cameras") or {}).values()),
            },
            "link_lines": rules_to_dict(link_lines_rules),
            "localization_3d": _read_localization_3d(
                _resolve_subscriptions_path(
                    backbone_data if isinstance(backbone_data, dict) else {},
                    Path(cfg.backbone_config_path))),
            "max_zones": MAX_ZONES,
            "allowed_kinds": list(ALLOWED_KINDS),
            "allowed_severities": list(ALLOWED_SEVERITIES),
            "node_id": node_id_out,
            "mqtt_sink": mqtt_sink_out,
            "camera_fps": camera_fps_out,
        }
    )


@router.get("/api/detection/onnx-files")
def detection_onnx_files(request: Request) -> JSONResponse:
    """List the trained ``best.onnx`` exports under ``trainer/isidet/runs/``.

    Mirrors ``/api/cameras/available``: scan the filesystem, return a JSON list,
    and the Settings modal populates a ``<datalist>`` tied to the ONNX path field.
    Newest first; the operator can still type/paste any path manually.
    """
    return JSONResponse({"files": list_trained_onnx()})


@router.get("/api/detection/pose-onnx-files")
def detection_pose_onnx_files(request: Request) -> JSONResponse:
    """List person-pose ``*.onnx`` exports (path contains "pose") under the trainer
    runs and ``models/`` — populates the Settings pose-model dropdown."""
    return JSONResponse({"files": list_pose_onnx()})


@functools.lru_cache(maxsize=16)
def _onnx_class_names_cached(path_str: str, _mtime_ns: int) -> tuple[str, ...]:
    """Class names embedded in an Ultralytics ONNX export (model metadata ``names``).
    Cached by (path, mtime) so repeated modal opens are instant. Empty if unreadable
    or the model carries none."""
    try:
        import onnx
        model = onnx.load(path_str, load_external_data=False)
        for prop in model.metadata_props:
            if prop.key == "names":
                parsed = ast.literal_eval(prop.value)
                if isinstance(parsed, dict):
                    return tuple(str(parsed[k]) for k in sorted(parsed))
                if isinstance(parsed, (list, tuple)):
                    return tuple(str(n) for n in parsed)
        # RF-DETR exports carry NO embedded names — infer the trained classes from
        # the RF-DETR output signature (dets/labels) so the Settings form isn't empty
        # (an empty class_names list fails the save's class_names>=1 validation).
        out_names = {o.name for o in model.graph.output}
        if {"dets", "labels"}.issubset(out_names):
            return ("palette", "carton", "polybag")
    except Exception as exc:  # any read/parse failure → no names (non-fatal)
        logger.warning("detection/classes: reading %s failed: %s", path_str, exc)
    return ()


def _model_class_names(onnx_path: str | None) -> list[str]:
    if not onnx_path:
        return []
    p = Path(onnx_path)
    if not p.exists():
        return []
    return list(_onnx_class_names_cached(str(p), p.stat().st_mtime_ns))


def _onnx_output_names(onnx_path: str) -> list[str]:
    """The model's output names (cheap) — used to pick the right task plugin
    (RF-DETR vs YOLO) on save, the same way the overlay does. For a native
    ``.engine`` the names come from its conversion sidecar (an engine can't be
    introspected without deserializing on the GPU)."""
    try:
        if str(onnx_path).endswith(".engine"):
            from backbone.shared.trt_session import read_sidecar
            return list((read_sidecar(onnx_path) or {}).get("outputs") or [])
        import onnx
        model = onnx.load(onnx_path, load_external_data=False)
        return [o.name for o in model.graph.output]
    except Exception as exc:
        logger.warning("detection: reading outputs of %s failed: %s", onnx_path, exc)
        return []


def _model_info(onnx_path: str | None) -> dict:
    """``{classes, input_wh, fixed_input, family}`` for the Settings modal.

    ``fixed_input`` is True when the ONNX has a static HxW (e.g. RF-DETR @432, or a
    YOLO exported with ``dynamic=False``) — the imgsz slider can't change it then, so
    the modal disables it and shows ``input_wh``. Dynamic models keep the slider."""
    info = {"classes": _model_class_names(onnx_path), "input_wh": None,
            "fixed_input": False, "family": "yolo"}
    if not onnx_path or not Path(onnx_path).exists():
        return info
    try:
        import onnx
        model = onnx.load(onnx_path, load_external_data=False)
        out_names = {o.name for o in model.graph.output}
        if {"dets", "labels"}.issubset(out_names):
            info["family"] = "rfdetr"
        dims = model.graph.input[0].type.tensor_type.shape.dim
        if len(dims) == 4:
            h, w = dims[2], dims[3]
            dynamic = bool(h.dim_param) or bool(w.dim_param) or h.dim_value <= 0 or w.dim_value <= 0
            if not dynamic:
                info["input_wh"] = [int(w.dim_value), int(h.dim_value)]
                info["fixed_input"] = True
    except Exception as exc:
        logger.warning("detection: model-info for %s failed: %s", onnx_path, exc)
    return info


@router.get("/api/detection/classes")
def detection_classes(request: Request, path: str | None = None) -> JSONResponse:
    """Class names embedded in the selected (``?path=``) or configured detection
    ONNX. The Settings modal DISPLAYS these read-only so the operator never keeps a
    class-name list in sync with the model — the detector self-configures from the
    model's metadata, and this surfaces the same names in the UI."""
    cfg = request.app.state.settings
    if not path:
        det = (_read_backbone(cfg) or {}).get("detection") or {}
        path = det.get("onnx_path")
    # Returns {classes, input_wh, fixed_input, family} — additive, so the modal's
    # existing `.classes` read keeps working while it can now also gate the slider.
    return JSONResponse(_model_info(path))


@router.post("/api/config")
def post_config(payload: ConfigPayload, request: Request) -> JSONResponse:
    """Atomically persist cameras + zones. Returns the number of zones written.

    Deliberately a SYNC handler: FastAPI runs it in the worker threadpool, so its
    blocking tail — onnx introspection, fsync'd YAML writes, reset_detector()'s
    gc.collect(), zone-worker reload — never stalls the event loop (which would
    freeze every /ws/video frame and API call for the duration)."""
    cfg = request.app.state.settings
    backbone_path = Path(cfg.backbone_config_path)
    backbone_data = _read_backbone(cfg) or {}

    # ---- splice camera URLs into the existing backbone.yaml structure ----
    cameras_block = backbone_data.get("cameras", {})
    if not isinstance(cameras_block, dict):
        cameras_block = {}
    had_cameras_before = bool(cameras_block)
    for cam_id, cam_cfg in payload.cameras.items():
        existing = cameras_block.get(cam_id, {})
        if not isinstance(existing, dict):
            existing = {}
        source = existing.get("source", {})
        if not isinstance(source, dict):
            source = {}
        if cam_cfg.device and cam_cfg.device.strip():
            # USB / V4L2 device — switch the plugin and drop RTSP-only keys.
            source["name"] = "v4l2"
            source["device"] = cam_cfg.device.strip()
            source.pop("url", None)
        else:
            # RTSP URL — switch the plugin and drop V4L2-only keys.
            source["name"] = "rtsp"
            source["url"] = cam_cfg.url.strip()
            source.pop("device", None)
        # Other source-level keys (latency_ms, width, height …) are preserved.
        existing["source"] = source
        cameras_block[cam_id] = existing

    # Remove dashboard-managed camera slots the operator cleared. The dashboard
    # sends the COMPLETE intended set of its two fixed slots (cam_a / cam_b), so
    # a slot absent from the payload means "delete it" — e.g. clearing Cam 2 to
    # drop from Mode 2 back to Mode 1. We only touch the managed slots, never a
    # camera a human added to backbone.yaml by hand.
    for slot in MANAGED_CAMERA_SLOTS:
        if slot in cameras_block and slot not in payload.cameras:
            cameras_block.pop(slot, None)

    # Fail-fast guard: refuse to wipe EVERY camera from a config that had at
    # least one. A camera-less backbone.yaml cannot boot (FrameSynchronizer
    # requires >=1 camera), so a save from a badly-prefilled Settings modal
    # must never take down a working system — reject it here, before any
    # write reaches disk (2026-07-22 incident: an empty modal wrote
    # `cameras: {}` and the backbone crash-looped at build).
    if had_cameras_before and not cameras_block:
        raise HTTPException(
            status_code=422,
            detail="refusing to remove every camera: a camera-less config "
                   "cannot run. Fill Cam 1 (and optionally Cam 2) in the "
                   "Settings modal, or reload the page if the fields came "
                   "up empty.",
        )

    # Camera FPS — write to every configured camera's source.capture_fps so the
    # camera-hub rate (and thus the cam-view video frame rate) matches the setting.
    # Only touches cameras currently in cameras_block (managed + hand-added).
    if payload.camera_fps is not None:
        fps_val = payload.camera_fps
        for cam_body in cameras_block.values():
            if isinstance(cam_body, dict):
                src = cam_body.get("source", {})
                if not isinstance(src, dict):
                    src = {}
                src["capture_fps"] = fps_val
                cam_body["source"] = src

    backbone_data["cameras"] = cameras_block

    # ---- splice the detection model into backbone.yaml (drives Backbone + MP4 viewer) ----
    # Backend is decided by the host's hardware, not the client.
    if payload.detection is not None:
        det = payload.detection
        backend = _detect_backend()
        block = backbone_data.get("detection", {})
        if not isinstance(block, dict):
            block = {}
        block["plugin"] = backend
        block["class_names"] = list(det.class_names)
        block["confidence_threshold"] = det.confidence_threshold
        block["iou_threshold"] = det.iou_threshold
        block["inference_imgsz"] = det.inference_imgsz
        if backend == "yolo_openvino":
            if not (det.model_xml and det.model_xml.strip()):
                raise HTTPException(400, "CPU-only host: an OpenVINO .xml model path is required")
            block["model_xml"] = det.model_xml.strip()
            block["device"] = "AUTO"
            block.pop("onnx_path", None)
            block.pop("providers", None)
        else:  # ONNX host
            if not (det.onnx_path and det.onnx_path.strip()):
                raise HTTPException(
                    400, "GPU host: a model path (.onnx or .engine) is required")
            onnx_path = det.onnx_path.strip()
            block["onnx_path"] = onnx_path
            block.pop("model_xml", None)
            block.pop("device", None)
            # Refine the plugin from the model's output signature (RF-DETR vs YOLO),
            # the SAME rule the overlay uses — so backbone.yaml's plugin matches the
            # model and the Backbone (on START) loads the right decoder, not just the
            # live preview. RF-DETR is NMS-free + fixed-input, so its YOLO-only knobs
            # are dropped.
            plugin = select_plugin(backend, _onnx_output_names(onnx_path))
            block["plugin"] = plugin
            if plugin == "rfdetr_onnx_seg":
                for k in ("iou_threshold", "inference_imgsz", "keep_classes"):
                    block.pop(k, None)
        # Person-pose model path — independent of the detection backend; persisted
        # so the operator's choice survives restarts. Empty clears it.
        if det.pose_onnx_path and det.pose_onnx_path.strip():
            block["pose_onnx_path"] = det.pose_onnx_path.strip()
        else:
            block.pop("pose_onnx_path", None)
        block["pose_enabled"] = bool(det.pose_enabled)
        block["pose_confidence_threshold"] = det.pose_confidence_threshold
        block["decode_masks"] = bool(det.decode_masks)
        backbone_data["detection"] = block
        # Remember whichever paths were provided so the modal can repopulate the
        # inactive one (only overwrite a path when non-empty).
        patch: dict[str, Any] = {
            "show_nodes": bool(det.show_nodes),
            "show_masks": bool(det.show_masks),
            "show_boxes": bool(det.show_boxes),
            "distance_line_opacity": float(det.distance_line_opacity),
            "distance_line_color": str(det.distance_line_color),
            "distance_line_thickness": int(det.distance_line_thickness),
        }
        if det.onnx_path and det.onnx_path.strip():
            patch["model_onnx_path"] = det.onnx_path.strip()
        if det.model_xml and det.model_xml.strip():
            patch["model_xml_path"] = det.model_xml.strip()
        if det.pose_onnx_path and det.pose_onnx_path.strip():
            patch["pose_onnx_path"] = det.pose_onnx_path.strip()
        _merge_ui_settings(cfg, patch)

    # ---- pose-only update (the current Settings modal): splice JUST the pose
    #      keys into the detection block — onnx_path/plugin/class_names/etc. are
    #      managed per zone and stay exactly as they are on disk. ----
    if payload.pose is not None:
        block = backbone_data.get("detection", {})
        if not isinstance(block, dict):
            block = {}
        pose_path = payload.pose.pose_onnx_path.strip()
        if pose_path:
            block["pose_onnx_path"] = pose_path
        else:
            block.pop("pose_onnx_path", None)   # empty = clear
        block["pose_enabled"] = bool(payload.pose.pose_enabled)
        block["pose_confidence_threshold"] = payload.pose.pose_confidence_threshold

        # ---- global isistream object-model knobs (one model, all zones) ----
        p = payload.pose
        if p.onnx_path and p.onnx_path.strip():
            onnx_path = p.onnx_path.strip()
            block["onnx_path"] = onnx_path
            # Pick the TASK plugin from the model's own output names (detect /
            # seg / RF-DETR), exactly like the camera-save path does. The base
            # backend stays hardware-decided.
            block["plugin"] = select_plugin(_detect_backend(),
                                            _onnx_output_names(onnx_path))
            if block["plugin"] == "rfdetr_onnx_seg":
                # switching from a YOLO model must not leave YOLO-only knobs
                # behind — isistream forwards the block to the constructor
                for yolo_key in ("iou_threshold", "keep_classes", "decode_masks"):
                    block.pop(yolo_key, None)
        if p.zone_imgsz is not None:
            block["zone_imgsz"] = max(128, min(1280, int(p.zone_imgsz)))
        if p.confidence_threshold is not None:
            block["confidence_threshold"] = max(0.0, min(1.0, float(p.confidence_threshold)))
        if p.sahi_enabled is not None or p.sahi_tile is not None or p.sahi_overlap is not None:
            sahi = dict(block.get("sahi") or {})
            if p.sahi_enabled is not None:
                sahi["enabled"] = bool(p.sahi_enabled)
            if p.sahi_tile is not None:
                sahi["tile"] = max(0, min(1280, int(p.sahi_tile)))
            if p.sahi_overlap is not None:
                sahi["overlap"] = max(0.0, min(0.9, float(p.sahi_overlap)))
            block["sahi"] = sahi
        if p.enhance_enabled is not None or p.enhance_gamma is not None:
            enh = dict(block.get("enhance") or {})
            if p.enhance_enabled is not None:
                enh["enabled"] = bool(p.enhance_enabled)
            if p.enhance_gamma is not None:
                enh["gamma"] = max(0.2, min(3.0, float(p.enhance_gamma)))
            block["enhance"] = enh
        backbone_data["detection"] = block

    if payload.decode_masks is not None:
        block = backbone_data.get("detection", {})
        if isinstance(block, dict):
            block["decode_masks"] = bool(payload.decode_masks)
            backbone_data["detection"] = block

    # ---- isistream toggles (motion gate / substream detection) ----
    if payload.motion_gate is not None or payload.detect_substream is not None:
        isis = backbone_data.get("isistream")
        if not isinstance(isis, dict):
            isis = {}
        if payload.motion_gate is not None:
            isis["motion_gate"] = bool(payload.motion_gate)
        if payload.detect_substream is not None:
            isis["detect_substream"] = bool(payload.detect_substream)
        backbone_data["isistream"] = isis

    # ---- communications: node_id + mqtt_sink ----
    # Both fields are optional: None = leave untouched. Handled BEFORE
    # _ensure_launchable so the sink splice isn't clobbered by the default-fill.
    if payload.node_id is not None:
        backbone_data["node_id"] = payload.node_id

    if payload.mqtt_sink is not None:
        ms = payload.mqtt_sink
        sink_dict: dict[str, Any] = {"plugin": "mqtt", "host": ms.host, "port": ms.port}
        # Splice the prefix (use provided value; JS defaults it to isiMonitor3D/v1/<node_id>).
        if ms.prefix:
            sink_dict["prefix"] = ms.prefix
        # Only write optional auth/tls fields when truthy — keep the YAML clean.
        if ms.tls:
            sink_dict["tls"] = True
        if ms.ca_cert:
            sink_dict["ca_cert"] = ms.ca_cert
        if ms.username:
            sink_dict["username"] = ms.username
        if ms.password:
            sink_dict["password"] = ms.password

        meta_block = backbone_data.get("metadata")
        if not isinstance(meta_block, dict):
            meta_block = {}
        sinks_list = meta_block.get("sinks")
        if not isinstance(sinks_list, list):
            sinks_list = []
        # Replace an existing mqtt entry; preserve all non-mqtt sinks (udp etc.).
        sinks_list = [s for s in sinks_list if not (isinstance(s, dict) and s.get("plugin") == "mqtt")]
        sinks_list.append(sink_dict)
        meta_block["sinks"] = sinks_list
        backbone_data["metadata"] = meta_block

    # ---- resolve zones_path; if backbone.yaml has none, default beside it ----
    zones_path = _resolve_zones_path(cfg, backbone_data)
    if zones_path is None:
        zones_path = backbone_path.parent / "zones.yaml"
        # Record it in backbone.yaml so subsequent reads find it.
        backbone_data["zones_path"] = str(zones_path)

    # ---- shape zones for YAML, preserving the original key order. Only when the
    #      payload actually carries zones — the metric-zone editor is gone, so the
    #      dashboard omits `zones` and we leave zones.yaml exactly as it is. ----
    if payload.zones is not None:
        zones_doc = {
            "zones": [
                {
                    # Persist the immutable id when the client supplies one so
                    # identity survives renames/reorders; omit for legacy zones
                    # (the loader then slugs one from the name).
                    **({"id": z.id} if z.id else {}),
                    "name": z.name,
                    "type": z.type,
                    "kind": z.kind,
                    "severity": z.severity,
                    "polygon": [list(pt) for pt in z.polygon],
                    # Per-zone detection config — omit when unset so zones.yaml stays
                    # clean (and a kind-less/legacy file round-trips unchanged).
                    **({"model": z.model} if z.model else {}),
                    **({"confidence_threshold": z.confidence_threshold}
                       if z.confidence_threshold is not None else {}),
                    # z_base_m omitted at 0.0 (the loader's default) so floor
                    # zones / legacy files round-trip without the key.
                    **({"z_base_m": z.z_base_m} if z.z_base_m else {}),
                }
                for z in payload.zones
            ]
        }

    # ---- 3D localization: regenerate subscriptions.yaml from the class list.
    #      Resolving the path may record a defaulted subscriptions_path into
    #      backbone_data — done BEFORE the backbone.yaml write below so both land
    #      in the same save. No hot-apply: subscriptions load at orchestrator
    #      boot, so the change takes effect at the next Backbone START. ----
    if payload.localization_3d is not None:
        subs_path = _resolve_subscriptions_path(backbone_data, backbone_path)
        try:
            _write_yaml_atomic(subs_path, _localization_3d_doc(payload.localization_3d))
        except OSError as exc:
            logger.exception("post_config: subscriptions write failed")
            raise HTTPException(
                status_code=500, detail=f"subscriptions write failed: {exc}"
            ) from exc

    # ---- fill required keys (metadata.sinks, calibration_path) so the written
    #      config can actually be launched, then write the file(s) atomically ----
    _ensure_launchable(backbone_data, cfg)
    try:
        if payload.zones is not None:
            _write_yaml_atomic(zones_path, zones_doc)
        _write_yaml_atomic(backbone_path, backbone_data)
    except OSError as exc:
        logger.exception("post_config: write failed")
        raise HTTPException(status_code=500, detail=f"write failed: {exc}") from exc

    # ---- write the link_lines section into the unified dashboard config ----
    if payload.link_lines is not None:
        ll_doc = {"rules": rules_to_dict(payload.link_lines)}
        try:
            dashboard_config.write_section(cfg, "link_lines", ll_doc)
        except OSError as exc:
            logger.exception("post_config: link_lines write failed")
            raise HTTPException(
                status_code=500, detail=f"link_lines write failed: {exc}"
            ) from exc

    # A model change must invalidate the dashboard's cached detector so the live
    # CAM preview + MP4 viewer pick up the new model without a restart.
    if payload.detection is not None:
        from ..detection_overlay import reset_detector

        reset_detector()

    # Camera sources may have changed — resync the background zone-detection
    # workers (a worker holds its camera's hub stream; a source change must
    # restart it against the new config).
    mgr = getattr(request.app.state, "zone_manager", None)
    if mgr is not None:
        mgr.reload()

    # Direction 1 hot-apply: a RUNNING isistream producer reads backbone.yaml
    # only at spawn, so model/camera/fps changes take a producer restart —
    # done here automatically (a few seconds; the metric engine keeps running
    # and coasts, the panels ride the RTSP fallback until the bus returns).
    host = getattr(request.app.state, "isistream", None)
    if (host is not None and host.points_mode()
            and host.status().get("running")):
        # In the background: the restart takes ~4 s (SIGTERM grace + model
        # load) and the Save button must not hang on it. The engine keeps
        # running; panels ride the RTSP fallback until the bus returns.
        logger.info("config: restarting isistream to apply the new settings")
        import threading as _threading

        def _restart() -> None:
            try:
                host.stop()
                host.start()
            except Exception:
                logger.warning("config: isistream restart failed", exc_info=True)

        _threading.Thread(target=_restart, daemon=True,
                          name="isistream-restart").start()

    return JSONResponse(
        {
            "ok": True,
            "zones_written": len(payload.zones) if payload.zones is not None else 0,
            "cameras_written": len(payload.cameras),
            "backbone_path": str(backbone_path),
            "zones_path": str(zones_path),
        }
    )


# ---- unified dashboard UI settings (server-side, survives sessions) ----


@router.get("/api/ui-settings")
def get_ui_settings(request: Request) -> JSONResponse:
    """Return the dashboard UI-preferences dict (e.g. {'mp4_selected': ...}).

    Also carries ``class_colors`` — the SERVER's canonical per-class overlay
    palette. The client-side cam-view renderer must use it so a pallet is the
    same colour there as on the server-rendered zone panels and their twins.

    Notable keys (the store is a generic merge — any JS can add its own):
    - ``video_passthrough`` (default true when absent): the big CAM views use
      the compressed-video passthrough (``camh264:`` over /ws/video + WebCodecs
      decode + /ws/overlays client-side drawing). ``false`` forces the classic
      server-drawn JPEG path (passthrough_player.js never subscribes camh264).
    """
    data = dict(_read_ui_settings(request.app.state.settings))
    # Server-owned, not an operator preference: the canonical overlay palette.
    data["class_colors"] = dict(CLASS_COLORS_HEX)
    data["class_color_default"] = DEFAULT_CLASS_COLOR_HEX
    return JSONResponse(data)


@router.post("/api/ui-settings")
def post_ui_settings(payload: dict, request: Request) -> JSONResponse:
    """Merge UI preferences into the YAML store (atomic write). Sync handler on
    purpose — the fsync'd write runs in the threadpool, off the event loop."""
    try:
        settings = _merge_ui_settings(request.app.state.settings, payload or {})
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"write failed: {exc}") from exc
    return JSONResponse({"ok": True, "settings": settings})
