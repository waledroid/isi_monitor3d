"""Read metadata embedded in a YOLO ONNX export.

Ultralytics stores useful self-description in the ONNX model metadata — most
importantly the class ``names``. Reading these lets a detector configure itself
from the model instead of a separate config list, so class-name drift can never
cause an ``nc`` mismatch when the model changes.
"""
from __future__ import annotations

import ast

import onnxruntime as ort


def read_embedded_class_names(session: ort.InferenceSession) -> list[str] | None:
    """Class names embedded by the Ultralytics export, index-ordered — or
    ``None`` if the model carries none. Stored under the ``names`` metadata key as
    a stringified ``{idx: name}`` dict, e.g. ``"{0: 'palette', 1: 'carton'}"``."""
    try:
        raw = session.get_modelmeta().custom_metadata_map.get("names")
        if not raw:
            return None
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, dict):
            return [str(parsed[k]) for k in sorted(parsed)]
        if isinstance(parsed, (list, tuple)):
            return [str(n) for n in parsed]
    except (ValueError, SyntaxError, AttributeError, TypeError):
        pass
    return None
