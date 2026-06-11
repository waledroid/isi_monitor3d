"""SD3.5 QLoRA trainer — Phase 4.

A memory-disciplined LoRA fine-tune of SD 3.5 on the project's curated images +
anti-bleed captions, sized for the 12 GB RTX 5070:

  1. **Precompute prompt embeddings** once per image with the pipeline's own
     ``encode_prompt`` (CLIP-L/G bf16 + T5-XXL in NF4), then FREE the encoders.
  2. **Precompute VAE latents** once per image (bf16 VAE), then FREE the VAE.
  3. Load ONLY the MMDiT transformer — **NF4-quantized** (QLoRA) — attach a
     PEFT LoRA (rank from config) on the attention projections, enable
     gradient checkpointing, train with **8-bit AdamW** on the flow-matching
     objective exactly as diffusers' SD3 training script defines it
     (logit-normal timestep sampling; ``target = noise - latents``).
  4. Save ``pytorch_lora_weights.safetensors`` loadable by
     ``pipe.load_lora_weights(run_dir)`` — set it as
     ``phases.generation.lora_weights`` for phase 5/7.

Config: project.yaml ``phases.lora`` (rank, resolution, max_steps, lr,
batch_size, grad_accum, base_model — defaults to the generation base).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from .base import LORA_TRAINERS, LoraTrainer

if TYPE_CHECKING:
    from ...core.project import ProjectConfig

logger = logging.getLogger(__name__)


@LORA_TRAINERS.register("diffusers_sd3")
class DiffusersSd3LoraTrainer(LoraTrainer):
    def __init__(self, base_model: str | None = None, rank: int = 16,
                 resolution: int = 512, max_steps: int = 2000, lr: float = 1e-4,
                 batch_size: int = 1, grad_accum: int = 4, seed: int = 42,
                 checkpoint_every: int = 500, **cfg) -> None:
        super().__init__(base_model=base_model, rank=rank, resolution=resolution,
                         max_steps=max_steps, lr=lr, batch_size=batch_size,
                         grad_accum=grad_accum, seed=seed,
                         checkpoint_every=checkpoint_every, **cfg)
        self.base_model = base_model or "stabilityai/stable-diffusion-3.5-large"
        self.rank = int(rank)
        self.resolution = int(resolution)
        self.max_steps = int(max_steps)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.grad_accum = int(grad_accum)
        self.seed = int(seed)
        self.checkpoint_every = int(checkpoint_every)

    # ---- data collection ----

    @staticmethod
    def _collect(project_dir: Path) -> list[tuple[Path, str]]:
        from ...core.manifest import Manifest
        manifest = Manifest.load(project_dir)
        items = []
        for rec in manifest.active():
            if getattr(rec, "synthetic", False) or not rec.caption_path:
                continue
            img = project_dir / rec.image
            cap = project_dir / rec.caption_path
            if img.exists() and cap.exists():
                items.append((img, cap.read_text().strip()))
        if not items:
            raise ValueError("lora: no curated images with captions — run phases 1-3 first")
        return items

    # ---- training ----

    def train(self, project: ProjectConfig, run_dir: Path) -> Path:
        import gc

        import numpy as np
        import torch
        import torch.nn.functional as F
        from diffusers import (
            AutoencoderKL,
            BitsAndBytesConfig,
            FlowMatchEulerDiscreteScheduler,
            SD3Transformer2DModel,
            StableDiffusion3Pipeline,
        )
        from diffusers.training_utils import compute_density_for_timestep_sampling
        from peft import LoraConfig
        from PIL import Image

        # The trainer is invoked with project_dir injected via cfg by the runner.
        project_dir = Path(self.cfg.get("project_dir") or ".")
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        device = "cuda"
        dtype = torch.bfloat16
        torch.manual_seed(self.seed)

        items = self._collect(project_dir)
        logger.info("lora: %d training images", len(items))

        # ---- 1. prompt embeddings (encoders loaded once, then freed) ----
        from transformers import BitsAndBytesConfig as HfBnb
        from transformers import T5EncoderModel
        logger.info("lora: encoding %d captions (CLIP-L/G + T5-XXL nf4)", len(items))
        te3 = T5EncoderModel.from_pretrained(
            self.base_model, subfolder="text_encoder_3",
            quantization_config=HfBnb(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                      bnb_4bit_compute_dtype=dtype),
            torch_dtype=dtype)
        enc_pipe = StableDiffusion3Pipeline.from_pretrained(
            self.base_model, transformer=None, vae=None,
            text_encoder_3=te3, torch_dtype=dtype)
        enc_pipe.text_encoder.to(device)
        enc_pipe.text_encoder_2.to(device)
        prompt_embeds_all, pooled_all = [], []
        with torch.no_grad():
            for _, caption in items:
                pe, _, pool, _ = enc_pipe.encode_prompt(
                    prompt=caption, prompt_2=caption, prompt_3=caption,
                    device=device, num_images_per_prompt=1,
                    do_classifier_free_guidance=False)
                prompt_embeds_all.append(pe.cpu())
                pooled_all.append(pool.cpu())
        del enc_pipe, te3
        gc.collect()
        torch.cuda.empty_cache()

        # ---- 2. VAE latents (VAE loaded once, then freed) ----
        logger.info("lora: encoding %d images to latents @ %dpx",
                    len(items), self.resolution)
        vae = AutoencoderKL.from_pretrained(self.base_model, subfolder="vae",
                                            torch_dtype=dtype).to(device)
        latents_all = []
        res = self.resolution
        with torch.no_grad():
            for img_path, _ in items:
                img = Image.open(img_path).convert("RGB")
                # center-crop to square then resize — keeps aspect of the subject
                s = min(img.size)
                left = (img.width - s) // 2
                top = (img.height - s) // 2
                img = img.crop((left, top, left + s, top + s)).resize((res, res),
                                                                      Image.BILINEAR)
                x = torch.from_numpy(np.array(img)).float() / 127.5 - 1.0
                x = x.permute(2, 0, 1)[None].to(device, dtype=dtype)
                lat = vae.encode(x).latent_dist.sample()
                lat = (lat - vae.config.shift_factor) * vae.config.scaling_factor
                latents_all.append(lat.cpu())
        del vae
        gc.collect()
        torch.cuda.empty_cache()

        # ---- 3. NF4 transformer + LoRA ----
        logger.info("lora: loading transformer (NF4) + LoRA r=%d", self.rank)
        transformer = SD3Transformer2DModel.from_pretrained(
            self.base_model, subfolder="transformer",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype),
            torch_dtype=dtype)
        transformer.requires_grad_(False)
        transformer.add_adapter(LoraConfig(
            r=self.rank, lora_alpha=self.rank, init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"]))
        transformer.enable_gradient_checkpointing()
        lora_params = [p for p in transformer.parameters() if p.requires_grad]
        n_train = sum(p.numel() for p in lora_params)
        logger.info("lora: %d trainable params", n_train)

        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(lora_params, lr=self.lr, weight_decay=1e-4)
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.base_model, subfolder="scheduler")
        sigmas_table = scheduler.sigmas.to(device, dtype=dtype)
        timesteps_table = scheduler.timesteps.to(device)

        def sample_sigmas(bsz):
            u = compute_density_for_timestep_sampling(
                weighting_scheme="logit_normal", batch_size=bsz,
                logit_mean=0.0, logit_std=1.0, mode_scale=1.29)
            idx = (u * scheduler.config.num_train_timesteps).long().clamp(
                0, len(timesteps_table) - 1)
            return sigmas_table[idx], timesteps_table[idx]

        # ---- 4. training loop (flow matching: target = noise - latents) ----
        rng = torch.Generator(device="cpu").manual_seed(self.seed)
        losses = []
        for step in range(1, self.max_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0
            for _ in range(self.grad_accum):
                pick = torch.randint(0, len(items), (self.batch_size,),
                                     generator=rng).tolist()
                latents = torch.cat([latents_all[i] for i in pick]).to(device, dtype=dtype)
                pe = torch.cat([prompt_embeds_all[i] for i in pick]).to(device, dtype=dtype)
                pool = torch.cat([pooled_all[i] for i in pick]).to(device, dtype=dtype)
                noise = torch.randn_like(latents)
                sigmas, timesteps = sample_sigmas(latents.shape[0])
                sig = sigmas.view(-1, 1, 1, 1)
                noisy = (1.0 - sig) * latents + sig * noise
                pred = transformer(hidden_states=noisy, timestep=timesteps,
                                   encoder_hidden_states=pe,
                                   pooled_projections=pool, return_dict=False)[0]
                target = noise - latents
                loss = F.mse_loss(pred.float(), target.float()) / self.grad_accum
                loss.backward()
                accum_loss += float(loss)
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            optimizer.step()
            losses.append(accum_loss)
            if step % 25 == 0 or step == 1:
                logger.info("lora: step %d/%d  loss %.4f", step, self.max_steps,
                            sum(losses[-25:]) / min(25, len(losses)))
            if self.checkpoint_every and step % self.checkpoint_every == 0:
                self._save(transformer, run_dir / f"checkpoint-{step}")

        weights = self._save(transformer, run_dir)
        (run_dir / "report.md").write_text(self._report(project, len(items), losses))
        logger.info("lora: done — weights at %s", weights)
        return weights

    @staticmethod
    def _save(transformer, out_dir: Path) -> Path:
        from diffusers import StableDiffusion3Pipeline
        from peft.utils import get_peft_model_state_dict
        out_dir.mkdir(parents=True, exist_ok=True)
        StableDiffusion3Pipeline.save_lora_weights(
            out_dir,
            transformer_lora_layers=get_peft_model_state_dict(transformer))
        return out_dir / "pytorch_lora_weights.safetensors"

    def _report(self, project, n_images: int, losses: list[float]) -> str:
        import statistics
        tail = losses[-100:] if len(losses) >= 100 else losses
        return (
            f"# LoRA training — {project.name}\n\n"
            f"- base: `{self.base_model}` (NF4 QLoRA)\n"
            f"- rank {self.rank} · {self.resolution}px · {self.max_steps} steps · "
            f"lr {self.lr} · batch {self.batch_size}x{self.grad_accum} accum\n"
            f"- images: {n_images}\n"
            f"- final loss (mean of last {len(tail)}): {statistics.mean(tail):.4f}\n\n"
            f"Use it: set `phases.generation.lora_weights` to this run dir's\n"
            f"`pytorch_lora_weights.safetensors` and run phase 7.\n"
        )
