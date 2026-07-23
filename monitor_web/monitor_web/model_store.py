"""Model discovery + resolution — which .onnx/.xml artifacts exist and which
one the config points at. One pipeline concern: the MODEL STORE. No sessions,
no drawing (see engines.py / overlay.py). Split out of detection_overlay.py.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .yaml_cache import load_yaml_cached

logger = logging.getLogger(__name__)


# Repo root (monitor_web/monitor_web/detection_overlay.py -> parents[2]).
_REPO_ROOT = Path(__file__).resolve().parents[2]


# Match any task subdir (detect, segment, …) so a seg run under runs/segment/
# is found by the latest-trained fallback, not just detect runs.
_RUNS_GLOB = "trainer/isidet/runs/*/models/yolo/*/weights"


# Default class for the single-class pallet model when none is configured.
_DEFAULT_CLASS_NAMES = ["palette_vide"]


# RF-DETR's trained classes map to logits columns 1/2/3 (column 0 = background).
_RFDETR_DEFAULT_CLASS_NAMES = ["palette", "carton", "polybag"]


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
        for p in (*root.glob("**/*.onnx"), *root.glob("**/*.engine")):
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
# NOTE: deliberately NOT the whole trainer tree — trainer/isidet/data holds
# ~34k dataset files and a recursive glob over it froze the Settings modal
# for 20+ s per open (measured live). Model artifacts only.
_POSE_ROOTS = (_RUNS_ROOT, _REPO_ROOT / "models",
               _REPO_ROOT / "trainer/isidet/models")


# Hand-dropped pose exports at the trainer top level (non-recursive).
_POSE_FLAT_DIRS = (_REPO_ROOT / "trainer/isidet",
                   _REPO_ROOT / "trainer/isidet/tests/.cache")


def list_pose_onnx() -> list[dict[str, object]]:
    """Every pose ``*.onnx`` (path contains "pose") under the trainer runs and
    ``models/``, newest first. Same shape as :func:`list_trained_onnx`; ``label``
    is the path relative to the repo root (pose models can live in several places,
    so an absolute-relative label is the least ambiguous)."""
    seen: set[str] = set()
    files: list[Path] = []

    def _consider(p: Path) -> None:
        rp = str(p.resolve())
        if p.is_file() and "pose" in rp.lower() and rp not in seen:
            seen.add(rp)
            files.append(p)

    for root in _POSE_ROOTS:
        if not root.exists():
            continue
        for p in (*root.glob("**/*.onnx"), *root.glob("**/*.engine")):
            _consider(p)
    for d in _POSE_FLAT_DIRS:
        if d.exists():
            for p in (*d.glob("*.onnx"), *d.glob("*.engine")):
                _consider(p)
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
    """backbone.yaml as a dict — mtime-cached (read from hot paths)."""
    return load_yaml_cached(cfg.backbone_config_path)


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
