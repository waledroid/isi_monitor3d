"""All eight seams registered with light imports (no torch/diffusers at import
time), and every plugin constructor parses its config without loading models."""

import src.stages  # noqa: F401 — fires every @register decorator
from src.stages.captioning.base import CAPTIONERS
from src.stages.control_maps.base import CONTROL_MAP_EXTRACTORS
from src.stages.exporting.base import DATASET_EXPORTERS
from src.stages.filtering.base import QUALITY_FILTERS
from src.stages.generation.base import IMAGE_GENERATORS
from src.stages.lora.base import LORA_TRAINERS
from src.stages.masking.base import MASKERS
from src.stages.scaffolds.base import SCAFFOLD_SOURCES


def test_all_seams_registered():
    assert CONTROL_MAP_EXTRACTORS.names() == ["canny", "depth_anything_v2"]
    assert MASKERS.names() == ["sam2"]
    assert CAPTIONERS.names() == ["template"]
    assert LORA_TRAINERS.names() == ["diffusers_sdxl"]
    assert SCAFFOLD_SOURCES.names() == ["box3d_procedural", "depth_remix"]
    assert IMAGE_GENERATORS.names() == ["sdxl_controlnet"]
    assert QUALITY_FILTERS.names() == ["clip_score"]
    assert DATASET_EXPORTERS.names() == ["labelme", "yolo_seg"]


def test_heavy_plugins_construct_without_loading_models():
    """Constructors only parse config — model load is deferred to load()/train(),
    so the whole registry is usable in a deps-free process."""
    gen = IMAGE_GENERATORS.create("sdxl_controlnet",
                                  steps=12, guidance=5.0, width=512, height=512)
    assert gen.steps == 12 and gen._pipe is None
    tr = LORA_TRAINERS.create("diffusers_sdxl", rank=8, max_steps=10)
    assert tr.rank == 8
    qf = QUALITY_FILTERS.create("clip_score")
    assert qf._model is None
    mk = MASKERS.create("sam2")
    assert mk._predictor is None


def test_sdxl_control_map_preprocessing():
    """2D depth maps become RGB PIL at the target size — hermetic, no GPU."""
    import numpy as np
    from src.stages.generation.sdxl_controlnet import SdxlControlNetGenerator
    depth = (np.arange(64 * 48, dtype=np.uint8) % 255).reshape(48, 64)
    pil = SdxlControlNetGenerator._prep_control(depth, 512, 512)
    assert pil.mode == "RGB" and pil.size == (512, 512)
    rgb = np.zeros((48, 64, 3), dtype=np.uint8)
    pil2 = SdxlControlNetGenerator._prep_control(rgb, 64, 48)
    assert pil2.mode == "RGB" and pil2.size == (64, 48)
