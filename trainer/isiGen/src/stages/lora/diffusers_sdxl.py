"""SDXL LoRA trainer — Phase 4.

A memory-disciplined LoRA fine-tune of SDXL on the project's curated images +
anti-bleed captions, sized for the 12 GB RTX 5070:

  1. **Precompute prompt embeddings** once per image with the pipeline's own
     ``encode_prompt`` (the two CLIPs — no T5 in SDXL), then FREE the encoders.
  2. **Precompute VAE latents** once per image with the **fp16-fix VAE**
     (the stock SDXL VAE NaNs in fp16), then FREE the VAE.
  3. Load ONLY the UNet (fp16) — attach a PEFT LoRA (rank from config) on the
     attention projections, cast the LoRA params to fp32
     (``cast_training_params``), enable gradient checkpointing, train with
     **8-bit AdamW** on the standard epsilon-prediction objective
     (``DDPMScheduler.add_noise``; ``target = noise``) under
     ``torch.autocast``.  SDXL's micro-conditioning ``add_time_ids`` is fixed
     at ``[res, res, 0, 0, res, res]`` — training crops are centered squares.
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


@LORA_TRAINERS.register("diffusers_sdxl")
class DiffusersSdxlLoraTrainer(LoraTrainer):
    def __init__(self, base_model: str | None = None, rank: int = 16,
                 resolution: int = 768, max_steps: int = 2000, lr: float = 1e-4,
                 batch_size: int = 1, grad_accum: int = 4, seed: int = 42,
                 checkpoint_every: int = 500,
                 vae: str = "madebyollin/sdxl-vae-fp16-fix", **cfg) -> None:
        super().__init__(base_model=base_model, rank=rank, resolution=resolution,
                         max_steps=max_steps, lr=lr, batch_size=batch_size,
                         grad_accum=grad_accum, seed=seed,
                         checkpoint_every=checkpoint_every, vae=vae, **cfg)
        self.base_model = base_model or "stabilityai/stable-diffusion-xl-base-1.0"
        self.vae_id = vae
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
            DDPMScheduler,
            StableDiffusionXLPipeline,
            UNet2DConditionModel,
        )
        from diffusers.training_utils import cast_training_params
        from peft import LoraConfig
        from PIL import Image

        # The trainer is invoked with project_dir injected via cfg by the runner.
        project_dir = Path(self.cfg.get("project_dir") or ".")
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        device = "cuda"
        dtype = torch.float16
        torch.manual_seed(self.seed)

        items = self._collect(project_dir)
        logger.info("lora: %d training images", len(items))

        # ---- 1. prompt embeddings (encoders loaded once, then freed) ----
        logger.info("lora: encoding %d captions (CLIP-L + CLIP-G)", len(items))
        enc_pipe = StableDiffusionXLPipeline.from_pretrained(
            self.base_model, unet=None, vae=None,
            torch_dtype=dtype, variant="fp16")
        enc_pipe.text_encoder.to(device)
        enc_pipe.text_encoder_2.to(device)
        prompt_embeds_all, pooled_all = [], []
        with torch.no_grad():
            for _, caption in items:
                pe, _, pool, _ = enc_pipe.encode_prompt(
                    prompt=caption, prompt_2=caption,
                    device=device, num_images_per_prompt=1,
                    do_classifier_free_guidance=False)
                prompt_embeds_all.append(pe.cpu())
                pooled_all.append(pool.cpu())
        del enc_pipe
        gc.collect()
        torch.cuda.empty_cache()

        # ---- 2. VAE latents (fp16-fix VAE loaded once, then freed) ----
        logger.info("lora: encoding %d images to latents @ %dpx",
                    len(items), self.resolution)
        vae = AutoencoderKL.from_pretrained(self.vae_id,
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
                lat = lat * vae.config.scaling_factor
                latents_all.append(lat.cpu())
        del vae
        gc.collect()
        torch.cuda.empty_cache()

        # ---- 3. fp16 UNet + LoRA (params in fp32) ----
        logger.info("lora: loading UNet (fp16) + LoRA r=%d", self.rank)
        unet = UNet2DConditionModel.from_pretrained(
            self.base_model, subfolder="unet",
            torch_dtype=dtype, variant="fp16").to(device)
        unet.requires_grad_(False)
        unet.add_adapter(LoraConfig(
            r=self.rank, lora_alpha=self.rank, init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"]))
        cast_training_params(unet, dtype=torch.float32)
        unet.enable_gradient_checkpointing()
        lora_params = [p for p in unet.parameters() if p.requires_grad]
        n_train = sum(p.numel() for p in lora_params)
        logger.info("lora: %d trainable params", n_train)

        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(lora_params, lr=self.lr, weight_decay=1e-4)
        scheduler = DDPMScheduler.from_pretrained(self.base_model,
                                                  subfolder="scheduler")

        # SDXL micro-conditioning: original size, crop top-left, target size.
        # Training crops are centered squares at `res`, so this is constant.
        time_ids = torch.tensor([res, res, 0, 0, res, res],
                                device=device, dtype=dtype)

        # ---- 4. training loop (epsilon prediction: target = noise) ----
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
                timesteps = torch.randint(
                    0, scheduler.config.num_train_timesteps,
                    (latents.shape[0],), generator=rng).to(device)
                noisy = scheduler.add_noise(latents, noise, timesteps)
                cond = {"text_embeds": pool,
                        "time_ids": time_ids.expand(latents.shape[0], -1)}
                with torch.autocast("cuda", dtype=dtype):
                    pred = unet(noisy, timesteps, encoder_hidden_states=pe,
                                added_cond_kwargs=cond, return_dict=False)[0]
                loss = F.mse_loss(pred.float(), noise.float()) / self.grad_accum
                loss.backward()
                accum_loss += float(loss)
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            optimizer.step()
            losses.append(accum_loss)
            if step % 25 == 0 or step == 1:
                logger.info("lora: step %d/%d  loss %.4f", step, self.max_steps,
                            sum(losses[-25:]) / min(25, len(losses)))
            if self.checkpoint_every and step % self.checkpoint_every == 0:
                self._save(unet, run_dir / f"checkpoint-{step}")

        weights = self._save(unet, run_dir)
        (run_dir / "report.md").write_text(self._report(project, len(items), losses))
        logger.info("lora: done — weights at %s", weights)
        return weights

    @staticmethod
    def _save(unet, out_dir: Path) -> Path:
        from diffusers import StableDiffusionXLPipeline
        from peft.utils import get_peft_model_state_dict
        out_dir.mkdir(parents=True, exist_ok=True)
        StableDiffusionXLPipeline.save_lora_weights(
            out_dir,
            unet_lora_layers=get_peft_model_state_dict(unet))
        return out_dir / "pytorch_lora_weights.safetensors"

    def _report(self, project, n_images: int, losses: list[float]) -> str:
        import statistics
        tail = losses[-100:] if len(losses) >= 100 else losses
        return (
            f"# LoRA training — {project.name}\n\n"
            f"- base: `{self.base_model}` (fp16 UNet, fp32 LoRA)\n"
            f"- rank {self.rank} · {self.resolution}px · {self.max_steps} steps · "
            f"lr {self.lr} · batch {self.batch_size}x{self.grad_accum} accum\n"
            f"- images: {n_images}\n"
            f"- final loss (mean of last {len(tail)}): {statistics.mean(tail):.4f}\n\n"
            f"Use it: set `phases.generation.lora_weights` to this run dir's\n"
            f"`pytorch_lora_weights.safetensors` and run phase 7.\n"
        )
