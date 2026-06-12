"""SDXL + depth ControlNet generator — Phases 5 (init) + 7 (minting).

The 12 GB RTX 5070 recipe (config: project.yaml ``phases.generation``):

  - ``StableDiffusionXLControlNetPipeline`` with
    ``diffusers/controlnet-depth-sdxl-1.0`` and the base
    ``stabilityai/stable-diffusion-xl-base-1.0`` (both UNGATED — no HF login).
  - Everything fp16 (``variant="fp16"``); the 2.6B UNet fits the card without
    quantization, so there are no NF4 / text-encoder escape hatches here.
  - The VAE is swapped for ``madebyollin/sdxl-vae-fp16-fix`` — the stock SDXL
    VAE produces NaNs in fp16.
  - Project LoRA injected with ``pipe.load_lora_weights(...)`` when configured.
  - ``pipe.enable_model_cpu_offload()`` by default — keeps VRAM headroom for
    anything else running on the card.
"""

from __future__ import annotations

import logging

import numpy as np

from .base import IMAGE_GENERATORS, ImageGenerator

logger = logging.getLogger(__name__)


@IMAGE_GENERATORS.register("sdxl_controlnet")
class SdxlControlNetGenerator(ImageGenerator):
    def __init__(self,
                 base_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
                 controlnet: str = "diffusers/controlnet-depth-sdxl-1.0",
                 vae: str = "madebyollin/sdxl-vae-fp16-fix",
                 cpu_offload: bool = True,
                 lora_weights: str | None = None,
                 steps: int = 30,
                 guidance: float = 7.5,
                 controlnet_scale: float = 0.85,
                 width: int = 1024,
                 height: int = 1024,
                 negative_prompt: str = "blurry, low quality, distorted, deformed",
                 **cfg) -> None:
        super().__init__(base_model=base_model, controlnet=controlnet, vae=vae,
                         cpu_offload=cpu_offload, lora_weights=lora_weights,
                         steps=steps, guidance=guidance,
                         controlnet_scale=controlnet_scale, width=width,
                         height=height, negative_prompt=negative_prompt, **cfg)
        self.base_model = base_model
        self.controlnet_id = controlnet
        self.vae_id = vae
        self.cpu_offload = bool(cpu_offload)
        self.lora_weights = lora_weights
        self.steps = int(steps)
        self.guidance = float(guidance)
        self.controlnet_scale = float(controlnet_scale)
        self.width = int(width)
        self.height = int(height)
        self.negative_prompt = negative_prompt
        self._pipe = None

    # ---- Phase 5: pipeline init ----

    def load(self) -> None:
        import torch  # lazy — heavy
        from diffusers import (
            AutoencoderKL,
            ControlNetModel,
            StableDiffusionXLControlNetPipeline,
        )

        dtype = torch.float16

        logger.info("generator: loading ControlNet %s", self.controlnet_id)
        controlnet = ControlNetModel.from_pretrained(
            self.controlnet_id, torch_dtype=dtype, variant="fp16")

        logger.info("generator: loading fp16-fix VAE %s", self.vae_id)
        vae = AutoencoderKL.from_pretrained(self.vae_id, torch_dtype=dtype)

        logger.info("generator: assembling pipeline (%s)", self.base_model)
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            self.base_model, controlnet=controlnet, vae=vae,
            torch_dtype=dtype, variant="fp16")
        if self.lora_weights:
            logger.info("generator: loading LoRA %s", self.lora_weights)
            pipe.load_lora_weights(self.lora_weights)
        if self.cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
        self._pipe = pipe
        logger.info("generator: pipeline ready")

    # ---- Phase 7: minting ----

    @staticmethod
    def _prep_control(control_image: np.ndarray, width: int, height: int):
        """uint8 HxW (depth) or HxWx3 → RGB PIL at the target size."""
        from PIL import Image
        ctrl = control_image
        if ctrl.ndim == 2:
            ctrl = np.repeat(ctrl[:, :, None], 3, axis=2)
        pil = Image.fromarray(np.ascontiguousarray(ctrl)).convert("RGB")
        if pil.size != (width, height):
            pil = pil.resize((width, height), Image.BILINEAR)
        return pil

    def generate(self, prompt: str, control_image: np.ndarray, *,
                 seed: int = -1, **params) -> np.ndarray:
        if self._pipe is None:
            self.load()
        import torch

        pil_ctrl = self._prep_control(control_image, self.width, self.height)

        gen = None
        if seed is not None and int(seed) >= 0:
            gen = torch.Generator(device="cpu").manual_seed(int(seed))
        result = self._pipe(
            prompt=prompt,
            negative_prompt=params.get("negative_prompt", self.negative_prompt),
            image=pil_ctrl,
            controlnet_conditioning_scale=params.get("controlnet_scale",
                                                     self.controlnet_scale),
            num_inference_steps=params.get("steps", self.steps),
            guidance_scale=params.get("guidance", self.guidance),
            width=self.width, height=self.height,
            generator=gen,
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
