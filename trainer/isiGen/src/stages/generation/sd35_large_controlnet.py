"""SD 3.5 Large + depth ControlNet generator — Phases 5 (init) + 7 (minting).

The 12 GB RTX 5070 recipe (config: project.yaml ``phases.generation``):

  - ``StableDiffusion3ControlNetPipeline`` with
    ``stabilityai/stable-diffusion-3.5-large-controlnet-depth`` and the base
    ``stabilityai/stable-diffusion-3.5-large`` (both GATED on HF — login first).
  - The 8B MMDiT transformer loads in NF4 via
    ``diffusers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16)``.
  - T5-XXL (text_encoder_3): NF4 too (``text_encoder_3: nf4``), or dropped
    entirely (``text_encoder_3: none`` → CLIP-only prompts) when the 12 GB of
    SYSTEM RAM — the tighter budget than VRAM under
    ``enable_model_cpu_offload()`` — runs out.
  - Project LoRA injected with ``pipe.load_lora_weights(...)`` when configured.
  - ``pipe.enable_model_cpu_offload()`` — never ``.to("cuda")`` whole.
"""

from __future__ import annotations

import logging

import numpy as np

from .base import IMAGE_GENERATORS, ImageGenerator

logger = logging.getLogger(__name__)


@IMAGE_GENERATORS.register("sd35_large_controlnet")
class Sd35LargeControlNetGenerator(ImageGenerator):
    def __init__(self,
                 base_model: str = "stabilityai/stable-diffusion-3.5-large",
                 controlnet: str = "stabilityai/stable-diffusion-3.5-large-controlnet-depth",
                 quantization: str = "nf4",
                 text_encoder_3: str = "nf4",
                 cpu_offload: bool = True,
                 lora_weights: str | None = None,
                 steps: int = 28,
                 guidance: float = 4.5,
                 controlnet_scale: float = 0.85,
                 width: int = 1024,
                 height: int = 1024,
                 negative_prompt: str = "blurry, low quality, distorted, deformed",
                 **cfg) -> None:
        super().__init__(base_model=base_model, controlnet=controlnet,
                         quantization=quantization, text_encoder_3=text_encoder_3,
                         cpu_offload=cpu_offload, lora_weights=lora_weights,
                         steps=steps, guidance=guidance,
                         controlnet_scale=controlnet_scale, width=width,
                         height=height, negative_prompt=negative_prompt, **cfg)
        self.base_model = base_model
        self.controlnet_id = controlnet
        self.quantization = quantization
        self.text_encoder_3 = text_encoder_3
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
            BitsAndBytesConfig,
            SD3ControlNetModel,
            SD3Transformer2DModel,
            StableDiffusion3ControlNetPipeline,
        )

        dtype = torch.bfloat16
        nf4 = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=dtype)

        logger.info("generator: loading ControlNet %s", self.controlnet_id)
        controlnet = SD3ControlNetModel.from_pretrained(self.controlnet_id,
                                                        torch_dtype=dtype)

        logger.info("generator: loading transformer (%s, %s)",
                    self.base_model, self.quantization)
        quant = nf4 if self.quantization == "nf4" else None
        transformer = SD3Transformer2DModel.from_pretrained(
            self.base_model, subfolder="transformer",
            quantization_config=quant, torch_dtype=dtype)

        extra: dict = {}
        if self.text_encoder_3 == "none":
            extra["text_encoder_3"] = None
            extra["tokenizer_3"] = None
            logger.info("generator: T5-XXL dropped (text_encoder_3: none)")
        elif self.text_encoder_3 == "nf4":
            from transformers import BitsAndBytesConfig as HfBnb
            from transformers import T5EncoderModel
            logger.info("generator: loading T5-XXL in NF4")
            extra["text_encoder_3"] = T5EncoderModel.from_pretrained(
                self.base_model, subfolder="text_encoder_3",
                quantization_config=HfBnb(load_in_4bit=True,
                                          bnb_4bit_quant_type="nf4",
                                          bnb_4bit_compute_dtype=dtype),
                torch_dtype=dtype)

        logger.info("generator: assembling pipeline")
        pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
            self.base_model, controlnet=controlnet, transformer=transformer,
            torch_dtype=dtype, **extra)
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

    def generate(self, prompt: str, control_image: np.ndarray, *,
                 seed: int = -1, **params) -> np.ndarray:
        if self._pipe is None:
            self.load()
        import torch
        from PIL import Image

        # Control map: uint8 HxW (depth) or HxWx3 → RGB PIL at the target size.
        ctrl = control_image
        if ctrl.ndim == 2:
            ctrl = np.repeat(ctrl[:, :, None], 3, axis=2)
        pil_ctrl = Image.fromarray(np.ascontiguousarray(ctrl)).convert("RGB")
        if pil_ctrl.size != (self.width, self.height):
            pil_ctrl = pil_ctrl.resize((self.width, self.height), Image.BILINEAR)

        gen = None
        if seed is not None and int(seed) >= 0:
            gen = torch.Generator(device="cpu").manual_seed(int(seed))
        result = self._pipe(
            prompt=prompt,
            negative_prompt=params.get("negative_prompt", self.negative_prompt),
            control_image=pil_ctrl,
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
