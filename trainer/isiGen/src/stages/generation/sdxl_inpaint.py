"""SDXL depth-ControlNet **inpaint** generator — the "harmonize" half of
paste-then-harmonize (Phases 5+7 for the copy_paste path).

Edits only the masked (pasted) region of a real composite image: the background
stays pixel-exact, the depth ControlNet keeps the object's geometry, and the
project LoRA + prompt drive its appearance. ``strength`` controls how much the
pasted pixels are changed (low = blend/keep, high = regenerate).

Pairs with the ``copy_paste`` scaffold source, which provides the base composite
RGB (``base``) and the inpaint mask (``inpaint``); the control map is the
composite depth.
"""

from __future__ import annotations

import logging

import numpy as np

from .base import IMAGE_GENERATORS, ImageGenerator

logger = logging.getLogger(__name__)


@IMAGE_GENERATORS.register("sdxl_inpaint")
class SdxlInpaintGenerator(ImageGenerator):
    def __init__(self,
                 base_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
                 controlnet: str = "diffusers/controlnet-depth-sdxl-1.0",
                 vae: str = "madebyollin/sdxl-vae-fp16-fix",
                 cpu_offload: bool = True,
                 lora_weights: str | None = None,
                 steps: int = 30,
                 guidance: float = 7.5,
                 controlnet_scale: float = 0.7,
                 strength: float = 0.75,
                 width: int = 1024,
                 height: int = 1024,
                 negative_prompt: str = "blurry, low quality, distorted, deformed",
                 **cfg) -> None:
        super().__init__(base_model=base_model, controlnet=controlnet, vae=vae,
                         cpu_offload=cpu_offload, lora_weights=lora_weights,
                         steps=steps, guidance=guidance,
                         controlnet_scale=controlnet_scale, strength=strength,
                         width=width, height=height, negative_prompt=negative_prompt, **cfg)
        self.base_model = base_model
        self.controlnet_id = controlnet
        self.vae_id = vae
        self.cpu_offload = bool(cpu_offload)
        self.lora_weights = lora_weights
        self.steps = int(steps)
        self.guidance = float(guidance)
        self.controlnet_scale = float(controlnet_scale)
        self.strength = float(strength)
        self.width = int(width)
        self.height = int(height)
        self.negative_prompt = negative_prompt
        self._pipe = None

    def load(self) -> None:
        import torch  # lazy — heavy
        from diffusers import (
            AutoencoderKL,
            ControlNetModel,
            StableDiffusionXLControlNetInpaintPipeline,
        )
        dtype = torch.float16
        logger.info("inpaint: loading depth ControlNet %s", self.controlnet_id)
        controlnet = ControlNetModel.from_pretrained(self.controlnet_id,
                                                     torch_dtype=dtype, variant="fp16")
        vae = AutoencoderKL.from_pretrained(self.vae_id, torch_dtype=dtype)
        logger.info("inpaint: assembling pipeline (%s)", self.base_model)
        pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            self.base_model, controlnet=controlnet, vae=vae,
            torch_dtype=dtype, variant="fp16")
        if self.lora_weights:
            logger.info("inpaint: loading LoRA %s", self.lora_weights)
            pipe.load_lora_weights(self.lora_weights)
        if self.cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
        self._pipe = pipe

    @staticmethod
    def _to_rgb_pil(arr: np.ndarray, width: int, height: int, *, gray_ok: bool = True):
        """BGR/gray uint8 ndarray → RGB PIL at (width, height)."""
        from PIL import Image
        a = arr
        if a.ndim == 2:
            a = np.repeat(a[:, :, None], 3, axis=2)
        else:
            a = a[:, :, ::-1]                       # BGR → RGB
        pil = Image.fromarray(np.ascontiguousarray(a)).convert("RGB")
        if pil.size != (width, height):
            pil = pil.resize((width, height), Image.BILINEAR)
        return pil

    @staticmethod
    def _to_mask_pil(arr: np.ndarray, width: int, height: int):
        from PIL import Image
        a = arr if arr.ndim == 2 else arr[:, :, 0]
        pil = Image.fromarray(np.ascontiguousarray(a)).convert("L")
        if pil.size != (width, height):
            pil = pil.resize((width, height), Image.NEAREST)
        return pil

    def generate(self, prompt: str, control_image: np.ndarray, *,
                 seed: int = -1, base_image: np.ndarray = None,
                 mask_image: np.ndarray = None, **params) -> np.ndarray:
        if base_image is None or mask_image is None:
            raise ValueError("sdxl_inpaint needs base_image + mask_image "
                             "(use the copy_paste scaffold source)")
        if self._pipe is None:
            self.load()
        import torch

        base_pil = self._to_rgb_pil(base_image, self.width, self.height)
        ctrl_pil = self._to_rgb_pil(control_image, self.width, self.height)
        mask_pil = self._to_mask_pil(mask_image, self.width, self.height)

        gen = None
        if seed is not None and int(seed) >= 0:
            gen = torch.Generator(device="cpu").manual_seed(int(seed))
        result = self._pipe(
            prompt=prompt,
            negative_prompt=params.get("negative_prompt", self.negative_prompt),
            image=base_pil, mask_image=mask_pil, control_image=ctrl_pil,
            strength=params.get("strength", self.strength),
            controlnet_conditioning_scale=params.get("controlnet_scale", self.controlnet_scale),
            num_inference_steps=params.get("steps", self.steps),
            guidance_scale=params.get("guidance", self.guidance),
            width=self.width, height=self.height, generator=gen,
        )
        rgb = np.asarray(result.images[0])
        return np.ascontiguousarray(rgb[:, :, ::-1])              # → BGR

    def close(self) -> None:
        self._pipe = None
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
