"""Optional auto-prompt detector for the masks phase (detect → box → SAM2).

Not a phase seam — a masking helper. ``build_prompt_detector(onnx_path, **cfg)``
picks RF-DETR or YOLO by the ONNX output names and returns a loaded
:class:`PromptDetector`. See ``base.py``.
"""

from .base import PromptDetector, build_prompt_detector  # noqa: F401
