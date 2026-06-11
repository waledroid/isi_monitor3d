"""isiGen stages — importing this package fires every @register decorator.

All eight seams' registries become populated; heavy deps (torch, transformers,
sam2, diffusers) are imported lazily inside load()/method bodies, so this
package imports cleanly in a deps-free test environment.
"""
from . import (  # noqa: F401
    captioning,
    control_maps,
    curate,
    exporting,
    filtering,
    generation,
    lora,
    masking,
    scaffolds,
)
