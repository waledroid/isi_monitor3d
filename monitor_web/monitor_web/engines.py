"""Inference-session lifecycle — the ENGINES stage: the lazy OpenVINO
detector singleton and reset_detector() (the STOP/model-change teardown).
No file discovery (model_store.py), no drawing (overlay.py). CPU branch:
dashboard-side pose engines were removed — skeletons ride the wire.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from fastapi import HTTPException

from .model_store import (
    _DEFAULT_CLASS_NAMES,
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


_gpu_probe = {"ts": 0.0, "free_mb": None}


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
    global _DETECTOR, _LOADED
    _DETECTOR = None
    _LOADED = None
    import gc
    gc.collect()


def get_detector(cfg):
    """Load (once) the configured OpenVINO detector from backbone.yaml.
    Raises HTTPException(503) if no usable IR is configured. No calibration
    needed. CPU branch: the backend is always OpenVINO — `model_xml` is the
    only model path key."""
    global _DETECTOR, _LOADED
    if _DETECTOR is not None:
        return _DETECTOR
    det_cfg = dict(read_backbone(cfg).get("detection") or {})
    plugin = det_cfg.pop("plugin", "yolo_openvino")
    if not plugin.startswith("yolo_openvino"):
        plugin = "yolo_openvino"                     # legacy configs: force IR
    # Pose keys belong to the producer's pose model, not the object detector.
    for k in ("pose_model_xml", "pose_onnx_path", "pose_confidence_threshold",
              "pose_enabled", "pose_imgsz", "pose_every_n", "onnx_path"):
        det_cfg.pop(k, None)
    # inference_imgsz (the Settings slider) → the detector's square input_size.
    imgsz = det_cfg.pop("inference_imgsz", None)
    if imgsz:
        det_cfg["input_size"] = (int(imgsz), int(imgsz))
    raw_path = det_cfg.get("model_xml")
    resolved = resolve_model(raw_path, cfg) if raw_path else None
    if resolved is None:
        # Nothing configured / unresolved → fall back to the latest IR under
        # models/ so the preview "just works".
        fallback = latest_trained_openvino()
        if not fallback:
            raise HTTPException(
                503, "no detection.model_xml configured and no IR found under models/"
            )
        logger.info("detection overlay: model_xml unset/unresolved — using %s", fallback)
        resolved = Path(fallback)
    det_cfg["model_xml"] = str(resolved)

    # Auto-pick the task plugin from the IR's output arity/names (2 outputs =
    # head + mask protos ⇒ seg; 1 output ⇒ detect). The operator just drops
    # the .xml in Settings — no manual "task" picker.
    try:
        import openvino as _ov
        model = _ov.Core().read_model(str(resolved))
        names = [next(iter(o.names), "") for o in model.outputs]
        chosen = select_plugin("yolo_openvino", names)
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

    if not plugin.endswith("_seg"):
        # Seg-only kwargs would be a constructor TypeError on the detect plugin.
        det_cfg.pop("decode_masks", None)
        det_cfg.pop("mask_threshold", None)

    try:
        import backbone.detection  # noqa: F401 — registers yolo_openvino{,_seg,_pose}
        from backbone.core.interfaces import detector_registry

        _DETECTOR = detector_registry.create(plugin, **det_cfg)
    except Exception as exc:
        raise HTTPException(503, f"failed to build detector '{plugin}': {exc}") from exc
    _LOADED = {"plugin": plugin, "path": str(resolved), "class_names": list(class_names)}
    logger.info("detection overlay: loaded %s (%s, %d classes)", plugin, resolved, len(class_names))
    return _DETECTOR


def current_model_info(cfg) -> dict:
    """Snapshot of the detection model for the heartbeat / status line.

    ``loaded`` is what the live overlay singleton actually built (None until a
    ``?detect=1`` stream — the MP4 dev viewer — has run since the last reset).
    ``configured`` is what ``backbone.yaml`` points at right now — what the next
    stream and the next Backbone boot will use. They differ exactly while a change
    is pending a reload, which is what makes this worth logging.
    """
    det_cfg = read_backbone(cfg).get("detection") or {}
    plugin = det_cfg.get("plugin", "yolo_openvino")
    raw = det_cfg.get("model_xml")
    resolved = resolve_model(raw, cfg) if raw else None
    conf_path = str(resolved) if resolved else (raw or None)
    loaded_path = _LOADED["path"] if _LOADED else None
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
    }
