"""Shared in-process detection overlay for the dashboard's annotated streams.

Used by the hidden MP4 viewer (`routes_media`) and the live CAM detect-preview
(`routes_video`). Runs the Backbone's detector in-process and draws boxes — the
documented exception to monitor_web's "no detector import" rule, for
preview/validation. It needs **no calibration**: it only loads the detection
model from `backbone.yaml`'s `detection` block and draws pixel boxes (the full
Backbone, by contrast, requires calibration because its output is metric).

Backend follows `detection.plugin` (yolo_onnx / yolo_openvino) via the registry,
so the preview matches whatever the live pipeline would use.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from fastapi import HTTPException

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
# Per-zone detector sessions, keyed by resolved onnx path — a zone patch can run a
# DIFFERENT (usually lighter) model than the global one. Each distinct model is its
# own CUDA session, so prefer small/nano models per zone.
_ZONE_DETECTORS: dict[str, object] = {}
# Serializes zone-detector BUILDS so concurrent first-access (e.g. both zone panels
# + cam reuse opening at once) builds one CUDA session per key, not N racing ones.
_ZONE_BUILD_LOCK = threading.Lock()
_BOX_COLOR = (80, 220, 80)  # BGR — default / unknown class

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
_gpu_skip_log_ts = 0.0


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

# Per-class overlay colours (BGR). The mask, box, and label all share the class
# colour so each object class is instantly distinguishable.
_CLASS_COLORS = {
    "palette": (80, 220, 80),    # green
    "carton": (120, 120, 255),   # light red
    "polybag": (255, 180, 120),  # light blue
}


def _color_for(cls) -> tuple[int, int, int]:
    return _CLASS_COLORS.get(str(cls).lower(), _BOX_COLOR)

# Detection classes treated as a "pallet" for person↔pallet distance lines.
_PALLET_CLASSES = {"palette", "pallet", "palette_vide"}

# Repo root (monitor_web/monitor_web/detection_overlay.py -> parents[2]).
_REPO_ROOT = Path(__file__).resolve().parents[2]
# Match any task subdir (detect, segment, …) so a seg run under runs/segment/
# is found by the latest-trained fallback, not just detect runs.
_RUNS_GLOB = "trainer/isidet/runs/*/models/yolo/*/weights"
# Default class for the single-class pallet model when none is configured.
_DEFAULT_CLASS_NAMES = ["palette_vide"]
# RF-DETR's trained classes map to logits columns 1/2/3 (column 0 = background).
_RFDETR_DEFAULT_CLASS_NAMES = ["palette", "carton", "polybag"]


def reset_detector() -> None:
    """Drop the cached detector + pose engine so the next stream reloads them
    (call on model change).

    Forces a GC pass so the previous ONNX Runtime CUDA session releases its GPU
    memory *before* the next session is built. Without this, swapping to a larger
    model (e.g. 840x840 RF-DETR, whose attention buffers are ~0.5 GB each) leaves
    the old and new sessions briefly coexistent on the 12 GB card and the swap
    OOMs (CUBLAS_STATUS_ALLOC_FAILED). The stream loop also drops its local
    detector ref before re-fetching (see _detect_iter) so nothing else pins it."""
    global _DETECTOR, _LOADED, _POSE, _POSE_PATH, _POSE_CONF
    _DETECTOR = None
    _LOADED = None
    _POSE = None
    _POSE_PATH = None
    _POSE_CONF = None
    _ZONE_DETECTORS.clear()   # drop per-zone sessions too
    import gc
    gc.collect()


def get_pose_detector(cfg):
    """Lazy person-pose engine from ``detection.pose_onnx_path`` in backbone.yaml,
    or None if none is configured/resolvable. Reloads when the configured path
    changes (so the Settings pose dropdown applies live, like the detector)."""
    global _POSE, _POSE_PATH, _POSE_CONF
    det_cfg = read_backbone(cfg).get("detection") or {}
    raw = det_cfg.get("pose_onnx_path")
    resolved = resolve_model(raw, cfg) if raw else None
    path = str(resolved) if resolved else None
    try:
        conf = float(det_cfg.get("pose_confidence_threshold", 0.3))
    except (TypeError, ValueError):
        conf = 0.3
    if path != _POSE_PATH or conf != _POSE_CONF:
        _POSE, _POSE_PATH, _POSE_CONF = None, path, conf   # config changed → drop stale engine
    if path is None:
        return None
    if _POSE is None:
        try:
            from .pose_overlay import PoseEngine

            _POSE = PoseEngine(path, conf=conf)
            logger.info("pose overlay: using %s (conf=%.2f)", path, conf)
        except Exception as exc:
            logger.warning("pose overlay: failed to load %s: %s", path, exc)
            _POSE = None
            _POSE_PATH = None
    return _POSE


def latest_trained_onnx() -> str | None:
    """Newest exported ``best.onnx`` under the isidet trainer runs, or None."""
    files = [p for p in _REPO_ROOT.glob(f"{_RUNS_GLOB}/best.onnx") if p.is_file()]
    return str(max(files, key=lambda p: p.stat().st_mtime)) if files else None


# Root of the isidet trainer runs — every *.onnx below here is offered in the
# Settings model dropdown.
_RUNS_ROOT = _REPO_ROOT / "trainer/isidet/runs"
# RF-DETR exports don't land under runs/ — they sit in models/rfdetr/<ts>/. Scan
# it too so the RF-DETR inference_model is selectable in the Settings dropdown
# (mirrors how pose models are discovered via several _POSE_ROOTS).
_MODEL_ROOTS = (_RUNS_ROOT, _REPO_ROOT / "trainer/isidet/models/rfdetr")


def list_trained_onnx() -> list[dict[str, object]]:
    """Every ``*.onnx`` under ``trainer/isidet/runs/`` and ``models/rfdetr/``,
    newest first.

    Returns ``[{"path", "label", "mtime"}, …]`` where ``path`` is the ABSOLUTE
    file path (what gets written to ``backbone.yaml``'s ``detection.onnx_path`` —
    absolute paths load verbatim in both the live overlay and the Backbone) and
    ``label`` is the path relative to ``trainer/isidet/`` (what the operator reads
    in the dropdown, e.g. ``runs/detect/models/yolo/…/weights/best.onnx`` or
    ``models/rfdetr/rfdetr-medium-seg_e41_432px/inference_model.sim.onnx``).
    """
    trainer_root = _REPO_ROOT / "trainer/isidet"
    seen: set[str] = set()
    files: list[Path] = []
    for root in _MODEL_ROOTS:
        if not root.exists():
            continue
        for p in root.glob("**/*.onnx"):
            rp = str(p.resolve())
            if p.is_file() and rp not in seen:
                seen.add(rp)
                files.append(p)
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    out: list[dict[str, object]] = []
    for p in files:
        try:
            label = str(p.relative_to(trainer_root))
        except ValueError:
            label = p.name
        out.append({"path": str(p), "label": label, "mtime": p.stat().st_mtime})
    return out


# Where pose ONNX exports live: pose training runs land under runs/.../pose/...,
# and hand-dropped/exported pose models commonly sit in models/. We treat any
# *.onnx whose path mentions "pose" as a person-pose model.
_POSE_ROOTS = (_RUNS_ROOT, _REPO_ROOT / "models", _REPO_ROOT / "trainer/isidet")


def list_pose_onnx() -> list[dict[str, object]]:
    """Every pose ``*.onnx`` (path contains "pose") under the trainer runs and
    ``models/``, newest first. Same shape as :func:`list_trained_onnx`; ``label``
    is the path relative to the repo root (pose models can live in several places,
    so an absolute-relative label is the least ambiguous)."""
    seen: set[str] = set()
    files: list[Path] = []
    for root in _POSE_ROOTS:
        if not root.exists():
            continue
        for p in root.glob("**/*.onnx"):
            rp = str(p.resolve())
            if p.is_file() and "pose" in rp.lower() and rp not in seen:
                seen.add(rp)
                files.append(p)
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    out: list[dict[str, object]] = []
    for p in files:
        try:
            label = str(p.relative_to(_REPO_ROOT))
        except ValueError:
            label = p.name
        out.append({"path": str(p), "label": label, "mtime": p.stat().st_mtime})
    return out


def latest_pose_onnx() -> str | None:
    """Newest pose ``*.onnx``, or None."""
    files = list_pose_onnx()
    return str(files[0]["path"]) if files else None


def latest_trained_openvino() -> str | None:
    """Newest exported OpenVINO ``*.xml`` under the isidet trainer runs, or None."""
    files = [p for p in _REPO_ROOT.glob(f"{_RUNS_GLOB}/**/*.xml") if p.is_file()]
    return str(max(files, key=lambda p: p.stat().st_mtime)) if files else None


def read_backbone(cfg) -> dict:
    path = Path(cfg.backbone_config_path)
    if not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError):
        return {}


def resolve_model(model_path: str, cfg) -> Path | None:
    """Resolve a possibly-relative model path (onnx_path / model_xml) against
    likely roots (CWD, next to backbone.yaml, repo root)."""
    p = Path(model_path)
    if p.is_absolute():
        return p if p.exists() else None
    bb = Path(cfg.backbone_config_path).resolve()
    candidates = [Path.cwd() / p, bb.parent / p, bb.parent.parent / p]
    return next((c for c in candidates if c.exists()), None)


# RF-DETR ONNX exports name their outputs dets / labels / masks — this is how an
# RF-DETR model is told apart from YOLO (whose seg/detect outputs aren't so named).
# `dets` + `labels` are mandatory (matching RfdetrOnnxSegDetector's own check);
# `masks` is the seg head and may be absent on a detect-only RF-DETR export.
_RFDETR_REQUIRED_NAMES = frozenset({"dets", "labels"})


def select_plugin(base_plugin: str, output_names: list[str]) -> str:
    """Pure plugin-selection from a model's ONNX/IR output names.

    ``base_plugin`` is the configured backend (``yolo_onnx`` / ``yolo_openvino``);
    the model's output *names* refine it to the right task plugin:

    * RF-DETR (outputs named ``dets`` AND ``labels``, ``masks`` optional) →
      ``rfdetr_onnx_seg`` — takes priority, independent of the base backend
      (the registered plugin is ONNX-only). Matches the backbone plugin's own
      output-name check, not a fragile arity guess.
    * 2 outputs (YOLO head + mask protos) → ``{base}_seg``.
    * otherwise the base plugin is kept unchanged.

    Kept pure (no I/O) so the selection rule is unit-testable without a real ONNX.
    """
    names = list(output_names or [])
    name_set = {str(n) for n in names}
    if _RFDETR_REQUIRED_NAMES.issubset(name_set):
        return "rfdetr_onnx_seg"
    if len(names) == 2:
        if base_plugin == "yolo_onnx":
            return "yolo_onnx_seg"
        if base_plugin == "yolo_openvino":
            return "yolo_openvino_seg"
    return base_plugin


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


def model_label(path: str | None) -> str:
    """Short, distinguishable label for a model path. Every export is named
    ``best.onnx``, so the bare filename is useless — show the path relative to
    ``runs/`` (same as the Settings dropdown) when it lives there, else the last
    two path components."""
    if not path:
        return "unset"
    p = Path(path)
    try:
        return str(p.relative_to(_RUNS_ROOT))
    except ValueError:
        parts = p.parts
        return str(Path(*parts[-2:])) if len(parts) >= 2 else p.name


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


def draw(image, det, show_nodes: bool = True, show_masks: bool = True,
         show_boxes: bool = True) -> None:
    color = _color_for(det.cls)   # per-class: palette=green, carton=light-red, polybag=light-blue
    # Segmentation mask underlay — only set by `yolo_onnx_seg` (or a future seg
    # plugin); detect detectors leave `det.mask = None` and this branch is skipped.
    if show_masks and getattr(det, "mask", None) is not None:
        m = det.mask
        # Blend the class colour only inside the mask. `addWeighted` is the
        # cheapest cv2 path; the boolean mask keeps the blend localised.
        if m.shape == image.shape[:2]:
            overlay = image.copy()
            overlay[m] = color
            cv2.addWeighted(overlay, 0.35, image, 0.65, 0, dst=image)
    x1, y1, x2, y2 = (int(v) for v in det.bbox_xyxy)
    # The bounding box is optional (Settings toggle) — with a seg model the mask +
    # label alone often reads cleaner. The class-name label is always drawn,
    # anchored at the box's top-left even when the box itself is hidden.
    if show_boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    label = f"{det.cls} {det.confidence:.2f}"
    cv2.putText(image, label, (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    # Object foot/edge nodes are intentionally NOT drawn — the only node point
    # shown is the PERSON foot node (drawn by the pose overlay). The pallet's
    # bbox-edge nodes are used only to anchor the nearest distance line, not shown
    # as points. (`show_nodes` is kept for signature compatibility / future use.)
    _ = show_nodes


def _project_to_floor(feet_uv, K, D, H):
    """Project pixel foot points to floor metres via undistort → homography.
    Reuses the SAME backbone geometry the metric pipeline uses, so the metres
    match what the Backbone would compute."""
    from backbone.shared.geometry import pixel_to_floor, undistort_points

    pts = np.asarray(feet_uv, dtype=np.float64).reshape(-1, 2)
    return pixel_to_floor(undistort_points(pts, K, D), H)


def compute_person_pallet_distances(person_feet, pallet_feet, view, frame_wh, *,
                                    max_m: float = 6.0):
    """Return ``[(person_uv, pallet_uv, d_m), ...]`` for every person↔pallet pair
    within ``max_m`` metres, using the floor homography. Pure (no drawing) so it's
    unit-testable. ``view`` carries the calibrated ``K, D, H, image_size_wh``.

    Frame-size guard: when the live frame size differs from the calibration size,
    ``H`` is rescaled (``H @ diag(cal_w/iw, cal_h/ih, 1)``) so actual-frame pixels
    map to the right metres — the same guard the MAP warp uses."""
    if not person_feet or not pallet_feet:
        return []
    H = np.asarray(view.H, dtype=np.float64)
    iw, ih = int(frame_wh[0]), int(frame_wh[1])
    cal_w, cal_h = int(view.image_size_wh[0]), int(view.image_size_wh[1])
    if (iw, ih) != (cal_w, cal_h):
        H = H @ np.diag([cal_w / iw, cal_h / ih, 1.0])
    K = np.asarray(view.K, dtype=np.float64)
    D = np.asarray(view.D, dtype=np.float64)
    persons_m = _project_to_floor(person_feet, K, D, H)
    pallets_m = _project_to_floor(pallet_feet, K, D, H)
    out = []
    for i, puv in enumerate(person_feet):
        for j, quv in enumerate(pallet_feet):
            d_m = float(np.hypot(persons_m[i, 0] - pallets_m[j, 0],
                                 persons_m[i, 1] - pallets_m[j, 1]))
            if d_m <= max_m:
                out.append((puv, quv, d_m))
    return out


def _filled_rounded_rect(img, x1, y1, x2, y2, r, color) -> None:
    """Filled rounded rectangle (cv2 has no native one): two rects + 4 corner discs."""
    r = max(0, min(r, (x2 - x1) // 2, (y2 - y1) // 2))
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
    for ccx, ccy in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
        cv2.circle(img, (ccx, ccy), r, color, -1, cv2.LINE_AA)


def _draw_distance(image, p1, p2, d_m: float, *, style=None) -> None:
    """Elastic line ``p1→p2`` + a white rounded centre badge with black ``'X.X m'``
    text. Line look (opacity / colour / thickness) comes from ``style`` (UI-settings
    via :func:`distance_line_style`); defaults to faint white 2 px."""
    opacity = float((style or {}).get("opacity", 0.25))
    color = (style or {}).get("color", (255, 255, 255))
    thickness = int((style or {}).get("thickness", 2))
    h, w = image.shape[:2]
    # Blend the line over the frame at `opacity` (line drawn on a copy → only its
    # pixels are blended).
    overlay = image.copy()
    cv2.line(overlay, p1, p2, color, thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, opacity, image, 1.0 - opacity, 0, dst=image)
    fs = max(0.4, min(h, w) / 1400.0)            # font scales with frame
    th = max(1, round(min(h, w) / 720.0))
    text = f"{d_m:.1f} m"
    (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
    cx, cy = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
    pad = max(3, round(min(h, w) / 360.0))
    rad = max(4, round(min(h, w) / 220.0))
    _filled_rounded_rect(image, cx - tw // 2 - pad, cy - tht // 2 - pad,
                         cx + tw // 2 + pad, cy + tht // 2 + pad, rad, (255, 255, 255))
    cv2.putText(image, text, (cx - tw // 2, cy + tht // 2),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th, cv2.LINE_AA)


def _bbox_edge_nodes(bbox) -> list[tuple[float, float]]:
    """The 4 edge-midpoint nodes of a bbox (pixel uv): top-mid, bottom-mid, left-mid,
    right-mid. Used so the distance line attaches to the pallet edge NEAREST the
    person, not always the bottom-centre."""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return [(cx, y1), (cx, y2), (x1, cy), (x2, cy)]


def draw_person_pallet_distances(image, detections, poses, view, *, max_m: float = 6.0,
                                 style=None) -> None:
    """Draw ONE white line + metre badge per pallet: from the nearest person foot node
    to that pallet's NEAREST bbox edge-midpoint node (of its 4 edge mids). For each
    pallet we measure the metric distance from every person to each of its 4 edge nodes
    and keep only the single shortest — so there's exactly one line per pallet, to the
    closest edge. Distances are metric via the floor homography in ``view``; no-op if
    either list is empty / projection fails."""
    pallets = [d for d in detections
               if str(getattr(d, "cls", "")).lower() in _PALLET_CLASSES
               and getattr(d, "bbox_xyxy", None) is not None]
    person_feet = [p.foot_uv for p in poses if getattr(p, "foot_uv", None) is not None]
    if not pallets or not person_feet:
        return
    frame_wh = (image.shape[1], image.shape[0])
    for pal in pallets:
        nodes = _bbox_edge_nodes(pal.bbox_xyxy)        # 4 edge midpoints
        try:
            # distances from every person to each of this pallet's 4 nodes (within max_m)
            pairs = compute_person_pallet_distances(person_feet, nodes, view, frame_wh, max_m=max_m)
        except Exception:
            logger.warning("distance overlay: projection failed", exc_info=True)
            continue
        if not pairs:
            continue
        # retain only the lowest-distance line for this bbox → one line per pallet
        puv, quv, d_m = min(pairs, key=lambda t: t[2])
        _draw_distance(image, (int(puv[0]), int(puv[1])),
                       (int(quv[0]), int(quv[1])), d_m, style=style)


# Load objects + per-state badge colours (BGR). Matches the 2D-map cue:
# empty=green · carton=amber · polybag=blue.
_OBJECT_CLASSES = {"carton", "polybag"}
_OCC_COLORS = {"empty": (80, 220, 80), "loaded": (53, 171, 245)}   # BGR: green / amber
# Canonical content order so a multi-load label is stable frame-to-frame (never
# "palette_polybag_carton" one frame and "palette_carton_polybag" the next).
_CONTENT_ORDER = ("carton", "polybag")


def _occupancy_label(contents) -> str:
    """Pallet label depicting presence + load: ``palette_vide`` when empty, else
    ``palette_<loads>`` in canonical order — e.g. ``palette_carton``,
    ``palette_polybag``, ``palette_carton_polybag``."""
    if not contents:
        return "palette_vide"
    ordered = [c for c in _CONTENT_ORDER if c in contents]
    ordered += sorted(c for c in contents if c not in _CONTENT_ORDER)   # unknowns last
    return "palette_" + "_".join(ordered)


def image_occupancy(detections, *, k: float = 1.5, a_min: float = 0.2):
    """Per-pallet empty/full from IMAGE OVERLAP (the A estimator) on one frame's
    detections — self-contained, no calibration. An object is "on" a pallet if its
    base sits in the pallet's box extended upward by ``k*`` its height and they
    overlap horizontally (fraction ≥ ``a_min``). Returns ``[(pallet_det, label), ...]``
    where label depicts pallet presence + its full load set — ``palette_vide`` when
    empty, else ``palette_<loads>`` (e.g. ``palette_carton``, ``palette_carton_polybag``).

    This mirrors the Backbone's A-association so the CAM preview agrees with the
    metric pipeline, without importing the homography layer (process boundary)."""
    pallets = [d for d in detections if str(getattr(d, "cls", "")).lower() in _PALLET_CLASSES]
    objects = [d for d in detections if str(getattr(d, "cls", "")).lower() in _OBJECT_CLASSES]
    loads: dict[int, list] = {i: [] for i in range(len(pallets))}
    for obj in objects:
        ox1, _oy1, ox2, oy2 = obj.bbox_xyxy
        best_i, best_s = None, a_min
        for i, pal in enumerate(pallets):
            px1, py1, px2, py2 = pal.bbox_xyxy
            h = max(1e-6, py2 - py1)
            if not (py1 - k * h <= oy2 <= py2 + 0.5 * h):
                continue
            overlap = max(0.0, min(ox2, px2) - max(ox1, px1)) / max(1e-6, ox2 - ox1)
            if overlap >= best_s:
                best_s, best_i = overlap, i
        if best_i is not None:
            loads[best_i].append(obj)

    out = []
    for i, pal in enumerate(pallets):
        contents = {str(o.cls).lower() for o in loads[i]}
        out.append((pal, _occupancy_label(contents)))
    return out


def _draw_occupancy_badge(image, pallet_det, label) -> None:
    """A small filled colour badge ('palette_vide' / 'palette_carton' / …) above a
    pallet box — green when empty, amber when loaded."""
    color = _OCC_COLORS["empty"] if label == "palette_vide" else _OCC_COLORS["loaded"]
    x1, y1, _x2, _y2 = (int(v) for v in pallet_det.bbox_xyxy)
    h, w = image.shape[:2]
    fs = max(0.45, min(h, w) / 1300.0)
    th = max(1, round(min(h, w) / 800.0))
    (tw, tht), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
    pad = max(3, round(min(h, w) / 360.0))
    by2 = max(tht + 2 * pad, y1)                 # badge sits just above the box
    cv2.rectangle(image, (x1, by2 - (tht + 2 * pad)), (x1 + tw + 2 * pad, by2), color, -1)
    cv2.putText(image, label, (x1 + pad, by2 - pad),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (20, 20, 20), th, cv2.LINE_AA)


def annotate_frame(image, detector, cam_id: str = "cam",
                   show_nodes: bool = True, show_masks: bool = True,
                   show_boxes: bool = True, pose_detector=None,
                   dist_view=None, dist_max_m: float = 6.0,
                   show_occupancy: bool = False, detections=None, dist_style=None):
    """Run detection on one BGR frame and draw masks (seg only) + boxes + foot
    nodes in place. When ``pose_detector`` is given, also run person-pose and draw
    skeletons + foot nodes. When ``dist_view`` (a calibrated camera) is given, draw
    a white line + metre badge from each person to each pallet. When
    ``show_occupancy``, draw a pallet empty/full badge. Never raises on a bad frame
    — returns the (possibly un-annotated) image so a stream won't break."""
    from backbone.core.types import Frame, FramePair

    # GPU-pressure guard: if the card is nearly full, skip this frame's inference
    # (return the raw image) rather than risk the OOM that corrupts the CUDA
    # context. The preview yields to the live Backbone; boxes simply don't refresh
    # for the throttled frames.
    if (detector is not None or pose_detector is not None) and not gpu_inference_safe():
        global _gpu_skip_log_ts
        now = time.time()
        if now - _gpu_skip_log_ts > 10.0:
            logger.warning(
                "GPU low on VRAM (<%d MB free) — preview skipping inference to "
                "protect the CUDA context", _GPU_MIN_FREE_MB,
            )
            _gpu_skip_log_ts = now
        return image

    ts = time.time()
    pair = FramePair(
        capture_ts=ts, frame_idx=0,
        frames={cam_id: Frame(camera_id=cam_id, capture_ts=ts, frame_idx=0, image=image)},
    )
    # `detections` may be PRE-COMPUTED (e.g. zone-based detection mapped back to the
    # full frame — cam1 then runs no heavy full-frame detector, only pose). When it
    # is None, detect on the full frame as usual.
    incoming = detections
    detections = []
    if incoming is not None:
        detections = list(incoming)
    elif detector is not None:
        try:
            detections = list(detector.detect(pair).get(cam_id, []))
        except Exception:
            logger.warning("detection overlay: detect failed", exc_info=True)
            detections = []
    for det in detections:
        draw(image, det, show_nodes=show_nodes, show_masks=show_masks, show_boxes=show_boxes)
    poses = []
    if pose_detector is not None:
        try:
            poses = pose_detector.predict(image)
            pose_detector.draw(image, poses)
        except Exception:
            logger.warning("pose overlay: draw failed", exc_info=True)
            poses = []
    # Person↔pallet distance lines need both lists + a calibrated view.
    if dist_view is not None and detections and poses:
        try:
            draw_person_pallet_distances(image, detections, poses, dist_view,
                                         max_m=dist_max_m, style=dist_style)
        except Exception:
            logger.warning("distance overlay: draw failed", exc_info=True)
    # Pallet empty/full badge (image-space association — no calibration needed).
    if show_occupancy and detections:
        try:
            for pal, label in image_occupancy(detections):
                _draw_occupancy_badge(image, pal, label)
        except Exception:
            logger.warning("occupancy overlay: draw failed", exc_info=True)
    return image


def _ui_pref(cfg, key: str, default: bool = True) -> bool:
    """Read one boolean preference from the UI-settings YAML; default if missing."""
    path = Path(cfg.ui_settings_path)
    if not path.exists():
        return default
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError):
        return default
    if not isinstance(data, dict):
        return default
    return bool(data.get(key, default))


def nodes_enabled(cfg) -> bool:
    """Dashboard preference: draw the white foot-node disc on each detection."""
    return _ui_pref(cfg, "show_nodes", True)


def masks_enabled(cfg) -> bool:
    """Dashboard preference: draw the seg mask overlay (only meaningful for
    seg detectors — detect detectors have no mask)."""
    return _ui_pref(cfg, "show_masks", True)


def occupancy_enabled(cfg) -> bool:
    """Dashboard preference: draw the pallet empty/full badge on the CAM overlay."""
    return _ui_pref(cfg, "show_occupancy", True)


def boxes_enabled(cfg) -> bool:
    """Dashboard preference: draw the detection bounding box. Off ⇒ mask + class
    label only (cleaner with a seg model)."""
    return _ui_pref(cfg, "show_boxes", True)


def distances_enabled(cfg) -> bool:
    """Dashboard preference: draw the person↔pallet distance lines (needs a pose
    model + calibration; degrades to no lines otherwise)."""
    return _ui_pref(cfg, "show_distances", True)


def display_fps(cfg) -> float:
    """Dashboard preference: cap the per-frame inference/compositing rate on display
    streams (CAM detect, zone patches, unified). Read from the UI-settings YAML;
    clamped to [1, 30]; default 10."""
    path = Path(cfg.ui_settings_path)
    val = 10.0
    if path.exists():
        try:
            data = yaml.safe_load(path.read_text()) or {}
            if isinstance(data, dict) and data.get("display_fps") is not None:
                val = float(data["display_fps"])
        except (OSError, yaml.YAMLError, TypeError, ValueError):
            val = 10.0
    return max(1.0, min(30.0, val))


def person_pallet_max_m(cfg) -> float:
    """Max person↔pallet distance (m) to draw a line for — caps clutter. Reads
    ``detection.person_pallet_max_distance_m`` from backbone.yaml (default 6.0)."""
    det = read_backbone(cfg).get("detection") or {}
    try:
        return float(det.get("person_pallet_max_distance_m", 6.0))
    except (TypeError, ValueError):
        return 6.0


def _hex_to_bgr(s, default=(255, 255, 255)) -> tuple[int, int, int]:
    """'#rrggbb' (or '#rgb') → (B, G, R) for cv2. Bad input → default."""
    try:
        h = str(s).strip().lstrip("#")
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))   # B, G, R
    except (ValueError, TypeError, IndexError):
        return default


def distance_line_style(cfg) -> dict:
    """Person↔pallet distance-line look from the UI-settings YAML: ``opacity`` [0..1],
    ``color`` (``#rrggbb`` → BGR tuple), ``thickness`` px. Defaults: 0.25 / white / 2."""
    path = Path(cfg.ui_settings_path)
    opacity, color, thickness = 0.25, (255, 255, 255), 2
    if path.exists():
        try:
            data = yaml.safe_load(path.read_text()) or {}
            if isinstance(data, dict):
                if data.get("distance_line_opacity") is not None:
                    opacity = float(data["distance_line_opacity"])
                if data.get("distance_line_color"):
                    color = _hex_to_bgr(data["distance_line_color"])
                if data.get("distance_line_thickness") is not None:
                    thickness = int(data["distance_line_thickness"])
        except (OSError, yaml.YAMLError, TypeError, ValueError):
            pass
    return {"opacity": max(0.05, min(1.0, opacity)), "color": color,
            "thickness": max(1, min(8, thickness))}
