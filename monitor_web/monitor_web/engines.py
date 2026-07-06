"""Inference-session lifecycle — the ENGINES stage: lazy detector / pose /
per-zone CUDA sessions, the shared GPU-memory guard, and reset_detector()
(the STOP/model-change teardown). No file discovery (model_store.py), no
drawing (overlay.py). Split out of detection_overlay.py.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from fastapi import HTTPException

from .model_store import (
    _DEFAULT_CLASS_NAMES,
    _RFDETR_DEFAULT_CLASS_NAMES,
    latest_trained_onnx,
    latest_trained_openvino,
    model_label,
    read_backbone,
    resolve_model,
    select_plugin,
)

logger = logging.getLogger(__name__)


# Lazy detector singleton (loading an ONNX/OpenVINO session is expensive; reuse it).
_DETECTOR = None


# What the live singleton actually loaded — {"plugin", "path", "class_names"} —
# so the heartbeat / status can report the model in use, not just the configured
# one. Set in get_detector(), cleared in reset_detector().
_LOADED: dict | None = None


# Lazy person-pose engine + the path/conf it loaded (None = no pose model configured).
_POSE = None


_POSE_PATH: str | None = None


_POSE_CONF: float | None = None
# Runtime pose input size (detection.pose_imgsz) the engine was built with.
_POSE_IMGSZ: int | None = None


# Per-zone detector sessions, keyed by resolved onnx path — a zone patch can run a
# DIFFERENT (usually lighter) model than the global one. Each distinct model is its
# own CUDA session, so prefer small/nano models per zone.
_ZONE_DETECTORS: dict[str, object] = {}


# Serializes zone-detector BUILDS so concurrent first-access (e.g. both zone panels
# + cam reuse opening at once) builds one CUDA session per key, not N racing ones.
_ZONE_BUILD_LOCK = threading.Lock()


# --- GPU memory guard --------------------------------------------------------
# The dashboard preview loads its own CUDA session(s) ON TOP of the live Backbone
# (and pose, and per-zone models). On a 12 GB card these can collectively exhaust
# VRAM; the failing allocation throws `CUDA 700: illegal memory access`, which
# CORRUPTS the CUDA context so every subsequent inference — Backbone included —
# fails and the app goes unresponsive. There is no recovery once 700 fires, so we
# must PREVENT it: when free VRAM is below a safety margin, the preview yields
# (skips this frame's inference) and lets the resident sessions breathe. The probe
# is the existing nvidia-smi query, cached so it's not run per frame.
_GPU_MIN_FREE_MB = 900          # skip preview inference when free VRAM < this


_GPU_PROBE_TTL_S = 1.5          # reuse the (subprocess) VRAM reading this long


# Building a NEW zone session needs the margin PLUS room for the session itself
# (weights + arena; ~300 MB for a nano @320, ~1.5 GB for RF-DETR @840 — this is a
# refuse-early floor, not a sizing model). Below it, get_zone_detector raises
# ZoneModelUnavailable instead of letting ORT OOM and corrupt the CUDA context.
_ZONE_SESSION_HEADROOM_MB = 600


_gpu_probe = {"ts": 0.0, "free_mb": None}


class ZoneModelUnavailable(RuntimeError):
    """A zone's detector session cannot be (safely) built right now — e.g. not
    enough free VRAM. The zone worker disables just that zone for a cooldown and
    keeps the others running; nothing else in the process is affected."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(detail or reason)
        self.reason = reason


def _gpu_free_mb():
    """Free VRAM in MB (cached ~1.5 s), or None when there's no GPU / probe fails."""
    now = time.time()
    if now - _gpu_probe["ts"] < _GPU_PROBE_TTL_S:
        return _gpu_probe["free_mb"]
    try:
        from backbone.shared.hardware import gpu_memory_mb
        mem = gpu_memory_mb()                       # (used_mb, total_mb) or None
        free = (mem[1] - mem[0]) if mem else None
    except Exception:
        free = None
    _gpu_probe["ts"] = now
    _gpu_probe["free_mb"] = free
    return free


def gpu_inference_safe() -> bool:
    """True if a preview inference is safe to run. No GPU / unknown → True (the
    CPU path can't trigger the CUDA OOM). False only when free VRAM is below the
    margin, so the preview backs off and yields the card to the live Backbone."""
    free = _gpu_free_mb()
    return free is None or free >= _GPU_MIN_FREE_MB


def reset_detector() -> None:
    """Drop the cached detector + pose engine so the next stream reloads them
    (call on model change).

    Forces a GC pass so the previous ONNX Runtime CUDA session releases its GPU
    memory *before* the next session is built. Without this, swapping to a larger
    model (e.g. 840x840 RF-DETR, whose attention buffers are ~0.5 GB each) leaves
    the old and new sessions briefly coexistent on the 12 GB card and the swap
    OOMs (CUBLAS_STATUS_ALLOC_FAILED). The stream loop also drops its local
    detector ref before re-fetching (see _detect_iter) so nothing else pins it."""
    global _DETECTOR, _LOADED, _POSE, _POSE_PATH, _POSE_CONF, _POSE_IMGSZ
    _DETECTOR = None
    _LOADED = None
    _POSE = None
    _POSE_PATH = None
    _POSE_CONF = None
    _POSE_IMGSZ = None
    _ZONE_DETECTORS.clear()   # drop per-zone sessions too
    # Drain the per-camera async pose runners: each pins its OWN PoseEngine
    # CUDA session (independent of _POSE) plus a worker thread — without this,
    # pose VRAM outlived STOP indefinitely.
    for runner in list(_ASYNC_POSE.values()):
        try:
            runner.stop()
        except Exception:
            logger.warning("reset_detector: async pose runner stop failed",
                           exc_info=True)
    _ASYNC_POSE.clear()
    import gc
    gc.collect()


def get_pose_detector(cfg):
    """Lazy person-pose engine from ``detection.pose_onnx_path`` in backbone.yaml,
    or None if none is configured/resolvable. Reloads when the configured path
    changes (so the Settings pose dropdown applies live, like the detector)."""
    global _POSE, _POSE_PATH, _POSE_CONF, _POSE_IMGSZ
    det_cfg = read_backbone(cfg).get("detection") or {}
    raw = det_cfg.get("pose_onnx_path")
    resolved = resolve_model(raw, cfg) if raw else None
    path = str(resolved) if resolved else None
    try:
        conf = float(det_cfg.get("pose_confidence_threshold", 0.3))
    except (TypeError, ValueError):
        conf = 0.3
    try:
        imgsz = int(det_cfg.get("pose_imgsz") or 0) or None
    except (TypeError, ValueError):
        imgsz = None
    if path != _POSE_PATH or conf != _POSE_CONF or imgsz != _POSE_IMGSZ:
        # config changed → drop stale engine
        _POSE, _POSE_PATH, _POSE_CONF, _POSE_IMGSZ = None, path, conf, imgsz
    if path is None:
        return None
    if _POSE is None:
        try:
            from .pose_overlay import PoseEngine

            _POSE = PoseEngine(path, conf=conf, imgsz=_POSE_IMGSZ)
            logger.info("pose overlay: using %s (conf=%.2f, imgsz=%s)",
                        path, conf, _POSE_IMGSZ or "model default")
        except Exception as exc:
            logger.warning("pose overlay: failed to load %s: %s", path, exc)
            _POSE = None
            _POSE_PATH = None
    return _POSE


# One async pose runner per camera view — created on demand, rebuilt when the
# underlying engine changes (Settings pose model change / reset_detector()).
_ASYNC_POSE: dict[str, object] = {}


def get_async_pose(cfg, camera_id: str):
    """Per-camera :class:`~monitor_web.pose_overlay.AsyncPoseRunner` around the
    configured pose engine, or ``None``. Same ``predict``/``draw`` interface as
    the engine — but inference runs in a background worker so the cam-view
    video rate is never chained to the pose model's latency."""
    from .pose_overlay import AsyncPoseRunner

    engine = get_pose_detector(cfg)
    if engine is None:
        _ASYNC_POSE.pop(camera_id, None)
        return None
    runner = _ASYNC_POSE.get(camera_id)
    if runner is None or runner.engine is not engine:
        runner = AsyncPoseRunner(engine)
        _ASYNC_POSE[camera_id] = runner
    return runner


def get_detector(cfg):
    """Load (once) the configured detector from backbone.yaml — whichever backend
    (`yolo_onnx` / `yolo_openvino`) is set. Raises HTTPException(503) if no usable
    model is configured. No calibration needed."""
    global _DETECTOR, _LOADED
    if _DETECTOR is not None:
        return _DETECTOR
    det_cfg = dict(read_backbone(cfg).get("detection") or {})
    plugin = det_cfg.pop("plugin", "yolo_onnx")
    # pose_onnx_path belongs to the separate pose engine (get_pose_detector), not
    # the object detector — drop it so it isn't passed to the detector constructor.
    det_cfg.pop("pose_onnx_path", None)
    det_cfg.pop("pose_confidence_threshold", None)   # belongs to the pose engine, not the object detector
    # inference_imgsz (the Settings slider) → the detector's square input_size.
    # Only effective on a DYNAMIC model; a fixed-size export ignores it (the
    # detector adopts the model's own size). Drop the raw key so it isn't passed
    # as an unknown kwarg to the constructor.
    imgsz = det_cfg.pop("inference_imgsz", None)
    if imgsz:
        det_cfg["input_size"] = (int(imgsz), int(imgsz))
    path_key = "model_xml" if plugin == "yolo_openvino" else "onnx_path"
    raw_path = det_cfg.get(path_key)
    resolved = resolve_model(raw_path, cfg) if raw_path else None
    if resolved is None:
        # Nothing configured / unresolved → fall back to the latest trained model
        # so the preview "just works" without the operator hunting for the path.
        fallback = latest_trained_openvino() if plugin == "yolo_openvino" else latest_trained_onnx()
        if not fallback:
            raise HTTPException(
                503, f"no detection.{path_key} configured and no trained model found"
            )
        logger.info("detection overlay: %s unset/unresolved — using latest trained %s",
                    path_key, fallback)
        resolved = Path(fallback)
    det_cfg[path_key] = str(resolved)

    # Auto-pick the task plugin from the model's output names/arity. RF-DETR is
    # detected by its named outputs (dets/labels/masks → rfdetr_onnx_seg); 2 outputs
    # (head + mask protos) ⇒ {base}_seg; 1 output ⇒ detect. Same rule for ONNX and
    # OpenVINO IR. The operator just drops the file in Settings — no manual "task"
    # picker. select_plugin() is pure; here we only do the introspection I/O.
    if plugin == "yolo_onnx":
        try:
            import onnx as _onnx
            names = [o.name for o in _onnx.load(str(resolved)).graph.output]
            chosen = select_plugin(plugin, names)
            if chosen != plugin:
                logger.info("detection overlay: %s outputs %s → using %s",
                            resolved.name, names, chosen)
                plugin = chosen
        except Exception as exc:
            logger.warning("detection overlay: could not introspect %s outputs: %s",
                           resolved.name, exc)
    elif plugin == "yolo_openvino":
        try:
            import openvino as _ov
            model = _ov.Core().read_model(str(resolved))
            names = [next(iter(o.names), "") for o in model.outputs]
            chosen = select_plugin(plugin, names)
            if chosen != plugin:
                logger.info("detection overlay: %s outputs %s → using %s",
                            resolved.name, names, chosen)
                plugin = chosen
        except Exception as exc:
            logger.warning("detection overlay: could not introspect %s outputs: %s",
                           resolved.name, exc)

    class_names = det_cfg.get("class_names") or det_cfg.get("keep_classes")
    if not class_names:
        class_names = list(_DEFAULT_CLASS_NAMES)   # single-class pallet default
        det_cfg["class_names"] = class_names

    if plugin == "rfdetr_onnx_seg":
        # RF-DETR reads its own fixed square input from the ONNX and is NMS-free —
        # drop the YOLO-only kwargs (iou/keep_classes/input_size) so they aren't
        # passed to its constructor. Default class_names to the trained palette/
        # carton/polybag triplet (not the single-class pallet default) unless the
        # operator configured names explicitly.
        if not det_cfg.get("class_names"):
            class_names = list(_RFDETR_DEFAULT_CLASS_NAMES)
        det_cfg = {
            "onnx_path": det_cfg["onnx_path"],
            "class_names": class_names,
            "confidence_threshold": float(det_cfg.get("confidence_threshold", 0.3)),
        }
        if "mask_threshold" in (read_backbone(cfg).get("detection") or {}):
            det_cfg["mask_threshold"] = float(
                (read_backbone(cfg).get("detection") or {})["mask_threshold"]
            )

    try:
        import backbone.detection  # noqa: F401 — registers yolo_onnx{,_seg}, yolo_openvino{,_seg}, rfdetr_onnx_seg
        from backbone.core.interfaces import detector_registry

        _DETECTOR = detector_registry.create(plugin, **det_cfg)
    except Exception as exc:
        raise HTTPException(503, f"failed to build detector '{plugin}': {exc}") from exc
    _LOADED = {"plugin": plugin, "path": str(resolved), "class_names": list(class_names)}
    logger.info("detection overlay: loaded %s (%s, %d classes)", plugin, resolved, len(class_names))
    return _DETECTOR


def get_zone_detector(model_path, cfg, input_size: int = 320):
    """Detector for a zone patch, built at ``input_size`` (small = fast/light), cached
    per ``(resolved model, input_size)``. Uses the zone's own model when set, else the
    globally-configured one (or the latest trained fallback). Zones sharing a
    (model, size) share one CUDA session; a 320 session is far lighter than the main
    640 preview one, which is the point — run zone detection cheaply.

    NOTE: confidence is deliberately NOT part of the cache key. The session is built
    at a low floor and each zone's threshold is applied as a cheap POST-FILTER on the
    returned detections (see ``_zone_patch_iter``). Keying by confidence would rebuild
    a whole CUDA session on every conf change — multi-second stalls + leaked sessions.

    The task plugin is auto-picked from the ONNX outputs (same rule as the main
    detector). Class names come from config when using the global model, else default
    to the RF-DETR / pallet sets."""
    det_cfg = read_backbone(cfg).get("detection") or {}
    # Resolve the model: per-zone override, else the global configured model, else
    # the latest trained export so a fresh setup still previews.
    resolved = resolve_model(model_path, cfg) if model_path else None
    using_global = resolved is None
    if resolved is None:
        raw = det_cfg.get("onnx_path")
        resolved = resolve_model(raw, cfg) if raw else None
    if resolved is None:
        fb = latest_trained_onnx()
        resolved = Path(fb) if fb else None
    if resolved is None:
        return get_detector(cfg)        # nothing resolvable → shared global (may 503)
    # Build at a LOW floor so per-zone post-filtering is authoritative down to it
    # (never above the configured global, so a low global still wins).
    conf = min(float(det_cfg.get("confidence_threshold", 0.3)), 0.05)
    key = (str(resolved), int(input_size))
    det = _ZONE_DETECTORS.get(key)
    if det is not None:
        return det
    # Double-checked locking: serialize the (slow, multi-second) CUDA-session build
    # so concurrent first-access for the same key builds ONCE — not N racing sessions
    # that each grab VRAM before one wins the cache slot.
    with _ZONE_BUILD_LOCK:
        det = _ZONE_DETECTORS.get(key)
        if det is not None:
            return det
        # VRAM admission: refuse to even START building when the card is nearly
        # full — an ORT CUDA OOM mid-build throws error 700, which corrupts the
        # context for EVERY resident session (Backbone + pose + other zones).
        # Raising here keeps the failure scoped to this one zone. No GPU /
        # unknown probe → no check (CPU execution can't trigger it).
        free = _gpu_free_mb()
        if free is not None and free < _GPU_MIN_FREE_MB + _ZONE_SESSION_HEADROOM_MB:
            raise ZoneModelUnavailable(
                "no_vram",
                f"refusing to build zone session for {resolved.name}: "
                f"{free:.0f} MB free < {_GPU_MIN_FREE_MB + _ZONE_SESSION_HEADROOM_MB} MB required",
            )
        plugin = "yolo_onnx"
        try:
            import onnx as _onnx
            out_names = [o.name for o in _onnx.load(str(resolved)).graph.output]
            plugin = select_plugin(plugin, out_names)
        except Exception as exc:
            logger.warning("zone detector: could not introspect %s: %s", resolved, exc)
        cfg_names = det_cfg.get("class_names")
        if using_global and isinstance(cfg_names, list) and cfg_names:
            names = list(cfg_names)
        else:
            names = list(_RFDETR_DEFAULT_CLASS_NAMES if plugin == "rfdetr_onnx_seg" else _DEFAULT_CLASS_NAMES)
        try:
            import backbone.detection  # noqa: F401 — registers the detector plugins
            from backbone.core.interfaces import detector_registry
            kwargs = dict(onnx_path=str(resolved), class_names=names, confidence_threshold=conf)
            if plugin != "rfdetr_onnx_seg":   # RF-DETR reads its own fixed square input
                kwargs["input_size"] = (int(input_size), int(input_size))
            det = detector_registry.create(plugin, **kwargs)
        except Exception as exc:
            logger.warning("zone detector: build failed for %s (%s) — using global: %s",
                           resolved, plugin, exc)
            return get_detector(cfg)
        _ZONE_DETECTORS[key] = det
    logger.info("zone detector: loaded %s @ %dpx floor-conf=%.2f (%s)", plugin, input_size, conf, resolved)
    return det


def current_model_info(cfg) -> dict:
    """Snapshot of the detection model for the heartbeat / status line.

    ``loaded`` is what the live overlay singleton actually built (None until a
    ``?detect=1`` stream has run since the last reset). ``configured`` is what
    ``backbone.yaml`` points at right now — what the next stream and the next
    Backbone boot will use. They differ exactly while a change is pending a
    reload, which is what makes this worth logging.
    """
    det_cfg = read_backbone(cfg).get("detection") or {}
    plugin = det_cfg.get("plugin", "yolo_onnx")
    path_key = "model_xml" if plugin == "yolo_openvino" else "onnx_path"
    raw = det_cfg.get(path_key)
    resolved = resolve_model(raw, cfg) if raw else None
    conf_path = str(resolved) if resolved else (raw or None)
    loaded_path = _LOADED["path"] if _LOADED else None
    # Per-zone detector sessions actually doing the cam-view detection (the
    # full-frame `_LOADED` preview is NOT built when zones exist). Keyed by
    # (resolved path, input_size) — surface a short summary of each.
    zones = [
        {"label": model_label(k[0]), "size": k[1]}
        for k in _ZONE_DETECTORS
    ]
    return {
        "configured": {
            "plugin": plugin,
            "path": conf_path,
            "label": model_label(conf_path),
            "resolved": resolved is not None,
        },
        "loaded": (
            {**_LOADED, "label": model_label(loaded_path)} if _LOADED else None
        ),
        "zones": zones,
    }
