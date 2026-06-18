"""Discover trained detector ONNX files for the masks-phase auto-prompt dropdown.

Scans the sibling **isidet** trainer's ``models/`` and ``runs/`` trees for
``*.onnx`` (mirrors monitor_web's ``list_trained_onnx`` / ``_MODEL_ROOTS``), so a
model trained for the Backbone can auto-prompt SAM2 here. Labels are relative to
the isidet root for a short, recognisable dropdown entry.
"""

from __future__ import annotations

from pathlib import Path

# trainer/isiGen/src/core/models.py → parents[3] == trainer/
_TRAINER_ROOT = Path(__file__).resolve().parents[3]
_ISIDET_ROOT = _TRAINER_ROOT / "isidet"
_MODEL_ROOTS = (_ISIDET_ROOT / "models", _ISIDET_ROOT / "runs")


def list_detector_onnx() -> list[dict]:
    """Return ``[{path, label}]`` for every ``*.onnx`` under isidet's model trees.

    ``path`` is absolute; ``label`` is relative to the isidet root (falling back
    to the file name if it sits outside). Sorted by label, deduped by path.
    """
    seen: set[Path] = set()
    out: list[dict] = []
    for root in _MODEL_ROOTS:
        if not root.is_dir():
            continue
        for onnx in sorted(root.rglob("*.onnx")):
            resolved = onnx.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                label = str(onnx.relative_to(_ISIDET_ROOT))
            except ValueError:
                label = onnx.name
            out.append({"path": str(onnx), "label": label})
    out.sort(key=lambda m: m["label"])
    return out
