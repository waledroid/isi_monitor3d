import logging
from src.shared.registry import HOOKS

logger = logging.getLogger(__name__)

@HOOKS.register('IndustrialLogger')
class IndustrialLogger:
    """A clean, epoch-level logger for Industrial Computer Vision."""
    
    def before_train(self, trainer):
        # Print the Header once at the very start
        header = (
            f"\n{'Epoch':>8} {'GPU_mem':>10} {'box_loss':>10} {'seg_loss':>10} "
            f"{'cls_loss':>10} {'dfl_loss':>10} {'Instances':>10} {'Size':>8}"
        )
        print(header)
        print("-" * len(header))

    def after_epoch(self, trainer):
        epoch_str = f"{trainer.current_epoch + 1}/{trainer.config.get('epochs', 30)}"

        import torch
        try:
            gpu_mem = f"{torch.cuda.memory_reserved(0) / 1e9:.2f}G" if torch.cuda.is_available() else "N/A"
        except Exception:
            gpu_mem = "N/A"

        lc = getattr(trainer, 'loss_components', {})
        def _fmt(key):
            return f"{lc[key]:>10.4f}" if key in lc else f"{'--':>10}"

        row = (
            f"{epoch_str:>8} {gpu_mem:>10} {_fmt('box')} {_fmt('seg')} "
            f"{_fmt('cls')} {_fmt('dfl')} {'--':>10} {trainer.config.get('image_size', 640):>8}"
        )
        print(row)
        
    def after_train(self, trainer):
        print("-" * 80)
        logger.info(f"✅ Training Session Complete. Best weights saved to {trainer.output_dir}")
