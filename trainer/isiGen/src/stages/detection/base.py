"""Prompt-detector helper — Phase-3 (masks) auto-prompting for SAM2.

A *prompt detector* runs an object detector on a curated image and emits its
boxes as :class:`MaskPrompt` (``kind="box"``) so SAM2 can segment tight masks
fully automatically (detect → box → SAM2). This is the validated remedy for
promptless SAM2 grabbing the belt/rails instead of the package on busy ROI
scenes.

It is **not** a new phase seam — the 8 phase seams are unchanged. ``run_masks``
orchestrates detector → box prompts → the existing SAM2 masker. Default is none:
no detector ⇒ SAM2 behaves exactly as before (hand prompts / promptless fallback).

The two decoders (RF-DETR, YOLO) are **ported** from the Backbone
(``backbone/detection/...``), not imported — isiGen and the Backbone are separate
projects and must stay self-contained. ONNX runs via ``onnxruntime`` (already in
the ``isi-train`` env, CUDA 1.19.x).

:func:`build_prompt_detector` introspects the ONNX **output names** to pick the
decoder — the same rule monitor_web's ``detection_overlay.select_plugin`` uses:
``dets``/``labels`` ⇒ RF-DETR; 2 outputs ⇒ YOLO-seg (boxes from the det head,
protos ignored); 1 output ⇒ YOLO-detect.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from ...core.manifest import MaskPrompt

logger = logging.getLogger(__name__)

# CUDA first, CPU fallback — onnxruntime-gpu in isi-train, CPU on a laptop.
DEFAULT_PROVIDERS: tuple[str, ...] = ("CUDAExecutionProvider", "CPUExecutionProvider")


class PromptDetector(ABC):
    """Detect objects in a BGR image and return them as box ``MaskPrompt``s.

    The ORT session is built eagerly by :func:`build_prompt_detector` (so it can
    read the output names to choose the decoder) and injected here; ``load()`` is
    therefore usually a no-op but stays callable for symmetry with the maskers.
    """

    def __init__(
        self,
        onnx_path: str | Path,
        *,
        session=None,
        output_names: list[str] | None = None,
        class_names: list[str] | None = None,
        confidence_threshold: float = 0.35,
        class_map: dict[str, str] | None = None,
    ) -> None:
        self.onnx_path = Path(onnx_path)
        self._session = session
        self._output_names = output_names
        self._class_names = list(class_names) if class_names else None
        self.confidence_threshold = float(confidence_threshold)
        # lower-cased detector-class → lower-cased override target
        self._class_map = {k.lower(): v for k, v in (class_map or {}).items()}

    def load(self) -> None:
        """Build the ORT session if one wasn't injected."""
        if self._session is None:
            self._session, self._output_names = _build_session(self.onnx_path)

    @abstractmethod
    def detect(self, image_bgr: np.ndarray,
               project_class_names: list[str]) -> list[MaskPrompt]:
        """Run the detector and return box prompts mapped to project classes."""

    def close(self) -> None:
        self._session = None

    # -- shared class mapping --------------------------------------------------
    def _map_class(self, det_cls: str,
                   project_class_names: list[str]) -> str | None:
        """Map a detector class name to a project class (case-insensitive).

        - explicit ``class_map`` override wins;
        - else a case-insensitive name match against project classes;
        - single-class projects accept **any** detection as that one class
          (the detector is just a box source then);
        - no match ⇒ ``None`` (detection dropped).
        """
        target = self._class_map.get(det_cls.lower())
        lut = {c.lower(): c for c in project_class_names}
        if target is not None:
            return lut.get(target.lower(), target)
        if det_cls.lower() in lut:
            return lut[det_cls.lower()]
        if len(project_class_names) == 1:
            return project_class_names[0]
        return None


def _build_session(onnx_path: Path):
    """Create an ORT session (CUDA→CPU) and return ``(session, output_names)``."""
    import onnxruntime as ort

    if not Path(onnx_path).exists():
        raise FileNotFoundError(f"prompt detector ONNX not found: {onnx_path}")
    avail = ort.get_available_providers()
    providers = [p for p in DEFAULT_PROVIDERS if p in avail] or ["CPUExecutionProvider"]
    try:
        sess = ort.InferenceSession(str(onnx_path), providers=providers)
    except Exception:  # CUDA libs missing / OOM → CPU
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out_names = [o.name for o in sess.get_outputs()]
    return sess, out_names


def build_prompt_detector(onnx_path: str | Path, **cfg) -> PromptDetector:
    """Build the right :class:`PromptDetector` for ``onnx_path`` by output names.

    Selection mirrors monitor_web's ``select_plugin``:
      * outputs include ``dets`` and ``labels``  → RF-DETR (``RfdetrPromptDetector``)
      * 2 outputs                                 → YOLO-seg (boxes from det head)
      * 1 output                                  → YOLO-detect
    """
    from .rfdetr_onnx import RfdetrPromptDetector
    from .yolo_onnx import YoloPromptDetector

    session, out_names = _build_session(Path(onnx_path))
    lower = {n.lower() for n in out_names}
    if "dets" in lower and "labels" in lower:
        cls: type[PromptDetector] = RfdetrPromptDetector
    else:
        cls = YoloPromptDetector  # 1 output (detect) or 2 outputs (seg head + protos)
    logger.info("prompt detector: %s → %s (outputs=%s)",
                Path(onnx_path).name, cls.__name__, out_names)
    return cls(onnx_path, session=session, output_names=out_names, **cfg)
