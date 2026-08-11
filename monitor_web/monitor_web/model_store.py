"""Model discovery + resolution — which OpenVINO ``.xml`` IRs exist and which
one the config points at. One pipeline concern: the MODEL STORE. No sessions,
no drawing (see engines.py / overlay.py).

CPU deployment branch: the only model format is the OpenVINO IR (``model.xml``
+ ``model.bin`` beside it), and the only model root is the repo's ``models/``
directory — the trainer trees of the GPU line are gone.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .yaml_cache import load_yaml_cached

logger = logging.getLogger(__name__)


# Repo root (monitor_web/monitor_web/model_store.py -> parents[2]).
_REPO_ROOT = Path(__file__).resolve().parents[2]


# Default class for the single-class pallet model when none is configured.
_DEFAULT_CLASS_NAMES = ["palette_vide"]


# The ONE model root of this branch: ship IRs under <repo>/models/<name>/model.xml.
_MODELS_ROOT = _REPO_ROOT / "models"


def _list_xml(pose: bool) -> list[dict[str, object]]:
    """Every ``*.xml`` under ``models/``, newest first, filtered by whether the
    path mentions "pose". Returns ``[{"path", "label", "mtime"}, …]`` where
    ``path`` is ABSOLUTE (what gets written to backbone.yaml) and ``label`` is
    relative to ``models/`` (what the operator reads in the dropdown)."""
    if not _MODELS_ROOT.exists():
        return []
    files = [p for p in _MODELS_ROOT.glob("**/*.xml")
             if p.is_file() and ("pose" in str(p).lower()) == pose]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    out: list[dict[str, object]] = []
    for p in files:
        try:
            label = str(p.relative_to(_MODELS_ROOT))
        except ValueError:
            label = p.name
        out.append({"path": str(p), "label": label, "mtime": p.stat().st_mtime})
    return out


def list_trained_onnx() -> list[dict[str, object]]:
    """Every OBJECT-detection IR (``*.xml`` not mentioning "pose") under
    ``models/``, newest first. (Name kept from the GPU line so the API route
    and templates keep working; the listed artifacts are IRs.)"""
    return _list_xml(pose=False)


def list_pose_onnx() -> list[dict[str, object]]:
    """Every pose IR (``*.xml`` whose path mentions "pose") under ``models/``,
    newest first. Same shape as :func:`list_trained_onnx`."""
    return _list_xml(pose=True)


def latest_pose_onnx() -> str | None:
    """Newest pose IR, or None."""
    files = list_pose_onnx()
    return str(files[0]["path"]) if files else None


def latest_trained_openvino() -> str | None:
    """Newest object-detection IR under ``models/``, or None."""
    files = list_trained_onnx()
    return str(files[0]["path"]) if files else None


def read_backbone(cfg) -> dict:
    """backbone.yaml as a dict — mtime-cached (read from hot paths)."""
    return load_yaml_cached(cfg.backbone_config_path)


def resolve_model(model_path: str, cfg) -> Path | None:
    """Resolve a possibly-relative ``model_xml`` path against likely roots
    (CWD, next to backbone.yaml, repo root)."""
    p = Path(model_path)
    if p.is_absolute():
        return p if p.exists() else None
    bb = Path(cfg.backbone_config_path).resolve()
    candidates = [Path.cwd() / p, bb.parent / p, bb.parent.parent / p]
    return next((c for c in candidates if c.exists()), None)


def select_plugin(base_plugin: str, output_names: list[str]) -> str:
    """Pure plugin-selection from an IR's output names/arity.

    * 2 outputs (YOLO head + mask protos) → ``yolo_openvino_seg``.
    * otherwise → ``yolo_openvino`` (detect).

    Kept pure (no I/O) so the selection rule is unit-testable without a real IR.
    """
    if len(list(output_names or [])) == 2:
        return "yolo_openvino_seg"
    return "yolo_openvino"


def model_label(path: str | None) -> str:
    """Short, distinguishable label for a model path — relative to ``models/``
    when it lives there, else the last two path components."""
    if not path:
        return "unset"
    p = Path(path)
    try:
        return str(p.relative_to(_MODELS_ROOT))
    except ValueError:
        parts = p.parts
        return str(Path(*parts[-2:])) if len(parts) >= 2 else p.name
