# isiGen generation stack: SD3.5-Large → SDXL + depth ControlNet

**Date:** 2026-06-12 · **Status:** approved

## Why

SD3.5 Medium was requested for resource/download reasons, but Stability only
ships ControlNets for SD3.5 **Large** — and ControlNet is what makes isiGen's
masks align "by construction". SDXL + `diffusers/controlnet-depth-sdxl-1.0`
gives the same depth-forced geometry at ~9.5 GB download, faster inference,
and cheaper LoRA training on the 12 GB RTX 5070. The never-GPU-verified SD3.5
implementations are deleted (recoverable from git).

## Changes

1. **Generator `sdxl_controlnet`** (`src/stages/generation/sdxl_controlnet.py`,
   replaces `sd35_large_controlnet.py`):
   `StableDiffusionXLControlNetPipeline` + `ControlNetModel`, fp16
   (`variant="fp16"`), VAE swapped for `madebyollin/sdxl-vae-fp16-fix`
   (configurable `vae` key; stock SDXL VAE NaNs in fp16). No quantization
   knobs (`quantization` / `text_encoder_3` keys dropped — 2.6B UNet fits).
   `cpu_offload` kept, default `true`. Same `generate()` contract
   (depth map → RGB PIL → pipe → BGR); defaults `steps: 30`,
   `guidance: 7.5`, `controlnet_scale: 0.85`, `1024×1024`. Control-map
   preprocessing factored into a static `_prep_control()` for hermetic tests.

2. **LoRA trainer `diffusers_sdxl`** (`src/stages/lora/diffusers_sdxl.py`,
   replaces `diffusers_sd3.py`): same 3-stage memory discipline
   (precompute prompt embeds → free encoders; precompute latents with the
   fp16-fix VAE → free; train UNet-only PEFT LoRA `to_q/k/v/out.0` +
   grad checkpointing + 8-bit AdamW). Objective = SDXL's: `DDPMScheduler`,
   epsilon prediction (`target = noise`), `added_cond_kwargs`
   (pooled embeds + `add_time_ids` fixed at `[res,res,0,0,res,res]` since
   training crops are centered squares). LoRA params trained in fp32
   (`cast_training_params`), forward under `torch.autocast`. Default
   `resolution: 768` (1024 fits but leaves no margin). Saves
   `pytorch_lora_weights.safetensors` via
   `StableDiffusionXLPipeline.save_lora_weights` — generator-loadable.

3. **Config:** `configs/project_template.yaml` + `data/pallets_demo/project.yaml`
   switch to the SDXL ids + `vae` key; `runners.py` plugin-name defaults updated.

4. **Tests:** `test_stubs.py` registry names + construction updated; new
   hermetic `_prep_control` test. No GPU/downloads in tests.

5. **Docs:** README model list + HF-login note (SDXL is ungated — login only
   needed if a gated model is configured).

## Out of scope

Studio (phase cards are generic), scaffolds/captions/export (depth-map + mask
contract unchanged), any quality tuning of the minted images (needs GPU runs).
