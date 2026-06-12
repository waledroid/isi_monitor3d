# isiGen — synthetic dataset generator (SDXL + depth ControlNet)

A reusable, **project-based** pipeline that turns 50–100 real photos per class
into an unlimited, perfectly-labeled synthetic dataset:

1. **Curate** real images (keep environments — no tight crops)
2. **Dual-layer maps** — DepthAnythingV2 / Canny *control maps* (for generation)
   + SAM2 color-coded *ground-truth masks* (the future labels)
3. **Anti-bleed captions** — unique trigger word per class (`ISI_PLT`, …) +
   exhaustive background description
4. **LoRA training** (SDXL UNet, fp16 + grad-checkpointing + 8-bit AdamW)
5. **Pipeline init** — SDXL + depth ControlNet + fp16-fix VAE, CPU offload
6. **Synthetic scaffolds** — procedural layouts → paired control map + mask
7. **Mint** — ControlNet forces geometry, prompts randomize backgrounds
8. **Auto-label + filter** — mask aligns by construction; CLIP-score filter; YOLO-seg export

Architecture mirrors `trainer/isidet`: ABC + registry per seam (8 seams), YAML
config per project, an isolated conda env, and headless CLI scripts — plus
**isiGen Studio**, a FastAPI web app whose pages are the per-phase visualizers.

## Setup

isiGen runs in the existing **`isi-train`** conda env (it already carries the
Blackwell-ready torch 2.10+cu128 the RTX 5070 needs). Only a few additions:

```bash
conda run -n isi-train pip install diffusers bitsandbytes sentencepiece pydantic-settings pytest
conda run -n isi-train pip install --no-build-isolation \
    "git+https://github.com/facebookresearch/sam2.git"
# sanity
conda run -n isi-train python -c "import torch, diffusers, sam2; print(torch.cuda.get_device_capability())"  # (12, 0)
# All generation models (SDXL base, depth ControlNet, fp16-fix VAE) are UNGATED
# — no HF login needed. Pre-fetch them (≈9.5 GB) with:
conda run -n isi-train hf download stabilityai/stable-diffusion-xl-base-1.0 --include "*.json" "*.txt" "*fp16.safetensors"
conda run -n isi-train hf download diffusers/controlnet-depth-sdxl-1.0 --include "*.json" "*fp16.safetensors"
conda run -n isi-train hf download madebyollin/sdxl-vae-fp16-fix
```

To recreate `isi-train` from scratch on a new machine, use the repo-root spec:
`conda env create -f isi-train.yml -n isi-train` (pins live in
`requirements-isi-train.txt` next to it).

## Quickstart (CLI)

```bash
cd trainer/isiGen && conda activate isi-train
python scripts/create_project.py --name pallets_v1 \
    --classes palette:ISI_PLT carton:ISI_CRTN polybag:ISI_PLYBG
python scripts/run_curate.py --project pallets_v1 --source /path/photos --class-name palette
python scripts/run_maps.py --project pallets_v1 --stage all       # depth + canny + SAM2 masks
python scripts/run_captions.py --project pallets_v1               # anti-bleed captions
python scripts/run_lora_train.py --project pallets_v1             # P4: SDXL LoRA (hours, GPU)
#   → set phases.generation.lora_weights to the printed weights path
python scripts/run_scaffolds.py --project pallets_v1 --count 200  # P6: paired control+mask
python scripts/run_generate.py --project pallets_v1 [--limit 10]  # P5+7: mint (SDXL fp16)
python scripts/run_export.py --project pallets_v1                 # P8: CLIP filter + YOLO-seg/LabelMe
# trained next door: point trainer/isidet's dataset_path at
#   data/pallets_v1/export/yolo_seg
```

## isiGen Studio (web)

```bash
python scripts/run_studio.py            # http://localhost:8200  (ISIGEN_PORT to change)
```

Pages: **Projects** → **Phase board** (run phases, live job log) → **Curate
gallery** (retag / exclude) → **Maps viewer** (image | depth | canny | mask,
SAM2 prompt canvas: click = +point, shift-click = −point, drag = box) →
**Caption editor** (edits are never overwritten by re-runs). One background
job at a time — the 12 GB GPU guarantee.

## Layout

```
configs/project_template.yaml   # per-project schema (classes, phase params)
scripts/run_*.py                # headless CLI per phase
src/core/                       # registry, manifest (jsonl), project config, phase runners
src/stages/<seam>/              # base.py = ABC + registry; implementations register themselves
src/studio/                     # FastAPI app (routers, templates, vanilla-JS pages)
data/<project>/                 # raw/, maps/{depth,canny,mask}/, captions/, manifest.jsonl
tests/                          # hermetic — no GPU, no downloads
```

Run tests: `conda run -n isi-train python -m pytest tests -q` (from `trainer/isiGen`).
