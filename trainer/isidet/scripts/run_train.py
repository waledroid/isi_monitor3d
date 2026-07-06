#!/usr/bin/env python3
import importlib
import os
import sys
import argparse
import yaml
import logging
from pathlib import Path

# 1. System Path Setup — INSERT at the front, don't append: a leftover
# PYTHONPATH pointing at another isidet checkout (e.g. the old ~/logistic
# tree) would otherwise shadow THIS tree's src/ — observed as a run silently
# training with old code (imgsz ignored, date-only run dirs, missing hooks).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 2. Registry and hooks (hooks have no optional dependencies so always safe to import)
from src.shared.registry import TRAINERS
import src.training.hooks

# Fail loudly if the imported src is not OURS (shadowing would waste a full
# training run on stale code).
import src as _src
_src_root = Path(_src.__file__).resolve().parent.parent
if _src_root != PROJECT_ROOT:
    print(f"FATAL: 'src' imported from {_src_root}, not {PROJECT_ROOT}. "
          f"Check PYTHONPATH (echo $PYTHONPATH) for a stale isidet checkout.",
          file=sys.stderr)
    sys.exit(1)

# Trainer module registry — add new trainers here without touching any other code
_TRAINER_MODULES = {
    'yolo':    'src.training.trainers.yolo',
    'yolov26': 'src.training.trainers.yolo',
    'rfdetr':  'src.training.trainers.rfdetr',
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning the merged result."""
    merged = base.copy()
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged

# Setup standard industrial logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("Ignition")


def _configure_shared_memory() -> None:
    """Prevent the WSL2 '/dev/shm bus error' DataLoader crash.

    PyTorch DataLoader workers hand sample tensors to the main process through
    POSIX shared memory (``/dev/shm``). On WSL2 that mount is small (here ~5.9 GB),
    and 1024px segmentation batches across several workers can exhaust it mid-run
    → ``Unexpected bus error ... insufficient shared memory (shm)`` and the
    training dies. The ``file_system`` sharing strategy passes tensors via on-disk
    file descriptors instead, sidestepping the /dev/shm limit entirely.

    Process-global (not an Ultralytics arg), so it survives ``--resume`` — which
    restores the checkpoint's original ``workers`` and would otherwise re-trigger
    the same crash. Must run in the main process before any worker is forked.
    """
    try:
        import torch.multiprocessing as mp
        mp.set_sharing_strategy("file_system")
        logger.info("🧠 Memory: torch mp sharing strategy = 'file_system' (WSL /dev/shm-safe)")
    except Exception as exc:  # a memory tweak must never block training
        logger.warning(f"⚠️  could not set mp sharing strategy: {exc}")


def main():
    _configure_shared_memory()
    # 3. CLI Argument Parser (Perfectly preserved!)
    parser = argparse.ArgumentParser(description="Industrial Training Pipeline")
    parser.add_argument('--config', type=str, default='configs/train.yaml', help='Path to master config')
    parser.add_argument('--resume', type=str, default=None, help='Path to last.pt to resume training')
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config

    if not config_path.exists():
        logger.error(f"❌ Master Config missing at {config_path}")
        sys.exit(1)

    # 4. Load Master Config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 5. 🛑 THE MERGE: Dynamically read the optimizer config from train.yaml
    if 'optimizer_config' in config:
        optim_path = PROJECT_ROOT / config['optimizer_config']
        if optim_path.exists():
            logger.info(f"🔗 Merging Secondary Config: {optim_path.name}")
            with open(optim_path, 'r') as f:
                optim_config = yaml.safe_load(f)
                config = _deep_merge(config, optim_config)
        else:
            logger.error(f"❌ Optimizer config not found at {optim_path}")
            sys.exit(1)
    else:
        logger.warning("⚠️ No 'optimizer_config' found in train.yaml. Using default parameters.")

    # 6. RESUME MODE LOGIC (Perfectly preserved for YOLO!)
    if args.resume:
        resume_file = PROJECT_ROOT / args.resume
        if not resume_file.exists():
            logger.error(f"❌ Cannot resume. File not found: {resume_file}")
            sys.exit(1)
        config['resume_path'] = str(resume_file)
        logger.info(f"🔄 RESUME MODE INITIATED: Will continue from {resume_file.name}")

    project_name = config.get('project_name', 'isiDetector')
    model_type = config.get('model_type')

    logger.info(f"🚀 INITIALIZING PROJECT: {project_name.upper()}")
    logger.info(f"🔍 Requested Model Architecture: {model_type}")

    # 7. Lazy-load only the trainer that is actually needed
    trainer_module = _TRAINER_MODULES.get(model_type)
    if trainer_module is None:
        logger.error(f"❌ '{model_type}' is not in _TRAINER_MODULES. Register it in run_train.py.")
        sys.exit(1)
    importlib.import_module(trainer_module)

    try:
        # 8. Registry Instantiation
        TrainerClass = TRAINERS.get(model_type)
        
        trainer = TrainerClass(config)

        # 9. Execute the Pipeline
        trainer.train()

        # 10. Post-Training Validation & Export
        trainer.evaluate()

        if config.get('export_model', True):
            trainer.export(format='onnx')

        # 11. Per-train report (Markdown in the run dir). Best-effort: a report
        # failure must never fail a successful training pipeline. run_train.py
        # lives in scripts/, which is on sys.path[0], so this import resolves.
        try:
            from make_train_report import generate_report
            generate_report(Path(trainer.output_dir))
        except Exception as e:
            logger.warning(f"⚠️ report generation skipped: {e}")

        logger.info("✅ PIPELINE EXECUTION COMPLETELY SUCCESSFUL.")

    except KeyError as e:
        logger.error(e)
        logger.error("💡 Did you spell the model_type correctly in train.yaml?")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"💥 FATAL ERROR during pipeline execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
