import os

# Must be set BEFORE any CUDA/PyTorch operations are performed
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import logging
import types
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from pathlib import Path
from src.training.base_trainer import BaseTrainer
from src.shared.registry import TRAINERS

try:
    from rfdetr import RFDETRSegMedium, RFDETRSegSmall, RFDETRSegNano
    from rfdetr.utilities import box_ops
except ImportError:
    raise ImportError("❌ Missing dependency. Run: pip install rfdetr")

logger = logging.getLogger(__name__)


@TRAINERS.register('rfdetr')
class RFDETRTrainer(BaseTrainer):
    """Trainer for RF-DETR instance segmentation, wrapping Roboflow's library.

    Registered under ``'rfdetr'``. Set ``model_type: "rfdetr"`` in
    ``configs/train.yaml`` to activate it.

    Uses a **DINOv2** backbone with deformable attention — a transformer
    architecture that requires no NMS and excels on small datasets or
    heavily overlapping objects.

    Key behaviours:

    - Applies a CUDA memory-safe monkey-patch to the postprocessing
      layer (offloads large mask interpolations >1 MP to CPU).
    - Uses **dual learning rates**: ``lr`` for the detection head
      (1e-4) and ``lr_encoder`` for the DINOv2 backbone (1e-5, 10×
      lower to avoid catastrophic forgetting).
    - Injects a per-epoch callback that captures loss history for
      post-training metric plots (``metrics_plot.png``).
    - :meth:`evaluate` reads results directly from the callback history
      collected during training — no separate validation pass needed.

    Attributes:
        model_size: ``'n'`` (Nano), ``'s'`` (Small, DINOv2-S/14),
            or ``'m'`` (Medium, DINOv2-B/14, default).
        dataset_path: ``Path`` to the COCO-format dataset root
            (must contain ``train/``, ``valid/``, ``test/`` subdirs
            with ``_annotations.coco.json`` files).
        history: List of per-epoch metric dicts accumulated by the
            ``on_fit_epoch_end`` callback during training.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.model_size = config.get('model_size', 'm')
        self.dataset_path = Path(config.get('dataset_path'))
        self.history = []

    def build_model(self):
        """Initialise the RF-DETR model and apply the memory-safe postprocess patch.

        Loads ``RFDETRSegNano``, ``RFDETRSegSmall``, or ``RFDETRSegMedium`` (DINOv2 pretrained
        weights are downloaded automatically on first run).

        After loading, monkey-patches the postprocessing layer so that
        mask interpolations larger than 1 MP are computed on CPU instead
        of GPU. This prevents CUDA OOM errors when processing high-
        resolution frames (e.g. 5 MP industrial cameras).

        Also sets ``self.num_select`` (top-K detections kept per image)
        from ``config['num_select']`` (default 100).
        """
        torch.cuda.empty_cache()
        logger.info(f"🏗️ Initializing RF-DETR SEGMENTATION ({self.model_size}) with DINOv2...")

        if self.model_size == 'n':
            self.model = RFDETRSegNano()
        elif self.model_size == 's':
            self.model = RFDETRSegSmall()
        else:
            self.model = RFDETRSegMedium()

        # 🛑 MONKEY-PATCH: Memory Safe Post-Processing
        # Offloads large mask interpolations (e.g. 5MP images) to CPU to prevent CUDA OOM
        num_select = self.config.get('num_select', 100)

        def patched_forward(post_self, outputs, target_sizes):
            out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]
            out_masks = outputs.get("pred_masks", None)

            prob = out_logits.sigmoid()
            topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), post_self.num_select, dim=1)
            scores = topk_values
            topk_boxes = topk_indexes // out_logits.shape[2]
            labels = topk_indexes % out_logits.shape[2]
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
            boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

            img_h, img_w = target_sizes.unbind(1)
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
            boxes = boxes * scale_fct[:, None, :]

            results = []
            if out_masks is not None:
                for i in range(out_masks.shape[0]):
                    res_i = {"scores": scores[i], "labels": labels[i], "boxes": boxes[i]}
                    k_idx = topk_boxes[i]
                    masks_i = torch.gather(
                        out_masks[i],
                        0,
                        k_idx.unsqueeze(-1).unsqueeze(-1).repeat(1, out_masks.shape[-2], out_masks.shape[-1]),
                    )
                    h, w = target_sizes[i].tolist()

                    # 🛡️ SAFETY CHECK: If mask interpolation > 1MP, offload to CPU to save VRAM
                    if h * w > 1024 * 1024:
                        device = masks_i.device
                        masks_i = F.interpolate(
                            masks_i.cpu().unsqueeze(1),
                            size=(int(h), int(w)),
                            mode="bilinear",
                            align_corners=False,
                        ).to(device)
                    else:
                        masks_i = F.interpolate(
                            masks_i.unsqueeze(1),
                            size=(int(h), int(w)),
                            mode="bilinear",
                            align_corners=False,
                        )
                    res_i["masks"] = masks_i > 0.0
                    results.append(res_i)
            else:
                results = [{"scores": s, "labels": l, "boxes": b} for s, l, b in zip(scores, labels, boxes)]
            return results

        self.model.model.postprocess.forward = types.MethodType(patched_forward, self.model.model.postprocess)
        self.model.model.postprocess.num_select = num_select
        self.num_select = num_select
        logger.info(f"🛡️ Memory-Safe Post-Processing active (num_select: {num_select})")

    def _inject_framework_hooks(self):
        """Bridge RF-DETR's ``on_fit_epoch_end`` callback to BaseTrainer hooks.

        Appends a callback to ``model.callbacks['on_fit_epoch_end']``
        that, after each epoch:

        1. Appends the full metrics dict to ``self.history``.
        2. Updates ``self.current_epoch`` and ``self.current_loss``.
        3. Populates ``self.loss_components`` from any ``loss``-named
           fields in the callback data.
        4. Calls ``self.call_hooks('after_epoch')``.
        5. Calls :meth:`~src.training.base_trainer.BaseTrainer._flush_memory`
           to aggressively clear VRAM and RAM after every epoch
           (critical for long training runs on 12 GB VRAM).
        """
        def log_metrics_callback(data):
            self.history.append(data)
            self.current_epoch = int(data.get('epoch', self.current_epoch))
            self.current_loss = float(data.get('train/loss', 0.0))
            # Expose any loss-named fields for IndustrialLogger
            self.loss_components = {
                k: float(v) for k, v in data.items()
                if isinstance(v, (int, float)) and 'loss' in k.lower()
            }
            self.call_hooks('after_epoch')
            self._flush_memory()

        if not hasattr(self.model, 'callbacks'):
            self.model.callbacks = {"on_fit_epoch_end": []}
        self.model.callbacks["on_fit_epoch_end"].append(log_metrics_callback)

    def train(self):
        if self.model is None:
            self.build_model()

        # 1. Timestamped output dir — set before hooks so before_train sees the correct path
        self._setup_run_dir(fmt="%d-%m-%Y_%H%M")

        self._inject_framework_hooks()

        epochs = self.config.get('epochs', 100)
        batch_size = self.config.get('batch_size', 2)
        resolution = self.config.get('image_size', 448)  # 448 is safer than 640 for 12GB VRAM
        mixed_precision = self.config.get('mixed_precision', True)

        logger.info(f"🔥 Starting RF-DETR SEGMENTATION training (Resolution: {resolution}px)...")
        self._flush_memory()
        logger.info(f"📂 Outputting logs & weights to: {self.output_dir}")
        self.call_hooks('before_train')

        self.model.train(
            dataset_dir=str(self.dataset_path),
            output_dir=str(self.output_dir),
            epochs=epochs,
            batch_size=batch_size,
            resolution=resolution,

            lr=self.optim_cfg.get('lr', 1e-4),
            lr_encoder=self.optim_cfg.get('lr_encoder', 1e-5),
            weight_decay=self.optim_cfg.get('weight_decay', 1e-4),

            use_ema=self.tricks_cfg.get('use_ema', True),
            grad_accum_steps=self.tricks_cfg.get('grad_accum_steps', 4),

            early_stopping=self.es_cfg.get('enabled', True),
            early_stopping_patience=self.es_cfg.get('patience', 7),

            num_select=self.num_select,

            mixed_precision=mixed_precision,
            num_workers=self.tricks_cfg.get('num_workers', 1),  # Low for WSL stability
            pin_memory=self.tricks_cfg.get('pin_memory', False),

            log_every_n_steps=1
        )

        self.call_hooks('after_train')
        self._plot_metrics()

    def _plot_metrics(self):
        """Generate and save a training metrics plot from callback history or metrics.csv."""
        if self.history:
            df = pd.DataFrame(self.history)
        else:
            csv_path = self.output_dir / 'metrics.csv'
            if not csv_path.exists():
                logger.warning("⚠️ No metrics data available to plot.")
                return
            df = pd.read_csv(csv_path)

        if df.empty:
            return

        if 'epoch' in df.columns:
            df = df.groupby('epoch', as_index=False).last()

        logger.info("📈 Generating metrics plot...")
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('RF-DETR Segmentation Training Metrics', fontsize=14)

        epoch_col = df['epoch'] if 'epoch' in df.columns else df.index

        def _plot_panel(ax, cols_labels, title, ylabel):
            plotted = False
            for col, label, color in cols_labels:
                if col in df.columns:
                    valid = df[col].dropna()
                    if not valid.empty:
                        ax.plot(epoch_col[valid.index], valid, label=label, color=color)
                        plotted = True
            ax.set_title(title)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(ylabel)
            if plotted:
                ax.legend()
            ax.grid(True)

        _plot_panel(axes[0, 0],
                    [('train/loss', 'Train Loss', 'blue'), ('val/loss', 'Val Loss', 'orange')],
                    'Loss', 'Loss')

        _plot_panel(axes[0, 1],
                    [('val/mAP_50', 'mAP@50', 'green'), ('val/mAP_50_95', 'mAP@50-95', 'teal')],
                    'Detection mAP', 'mAP')

        _plot_panel(axes[0, 2],
                    [('val/segm_mAP_50', 'Segm mAP@50', 'purple'), ('val/segm_mAP_50_95', 'Segm mAP@50-95', 'violet')],
                    'Segmentation mAP', 'mAP')

        _plot_panel(axes[1, 0],
                    [('val/precision', 'Precision', 'red'), ('val/recall', 'Recall', 'darkorange'), ('val/F1', 'F1', 'brown')],
                    'Precision / Recall / F1', 'Score')

        _plot_panel(axes[1, 1],
                    [('val/AP/carton', 'AP carton', 'green'), ('val/AP/polybag', 'AP polybag', 'orange')],
                    'Per-Class AP', 'AP')

        _plot_panel(axes[1, 2],
                    [('train/lr', 'LR', 'steelblue')],
                    'Learning Rate', 'LR')

        plt.tight_layout()
        plot_path = self.output_dir / 'metrics_plot.png'
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"✅ Metrics plot saved to {plot_path}")

    def evaluate(self) -> dict:
        """Return the last epoch's validation metrics from training history.

        Reads from ``self.history`` collected during training — no
        additional inference pass is needed.

        Returns:
            A metrics dictionary with keys:

            - ``'mAP50'`` — detection mAP @ IoU 0.50
            - ``'mAP50_95'`` — detection mAP @ IoU 0.50–0.95
            - ``'mask_mAP50'`` — segmentation mAP @ 0.50
            - ``'mask_mAP50_95'`` — segmentation mAP @ 0.50–0.95

            Returns ``{'mAP50': 0.0, 'mAP50_95': 0.0}`` if training
            history is empty (e.g. training crashed before first epoch).
        """
        if not self.history:
            logger.warning("⚠️ No training history available — evaluate() skipped.")
            return {'mAP50': 0.0, 'mAP50_95': 0.0}

        last = self.history[-1]
        metrics = {
            'mAP50': float(last.get('val/mAP_50', 0.0)),
            'mAP50_95': float(last.get('val/mAP_50_95', 0.0)),
            'mask_mAP50': float(last.get('val/segm_mAP_50', 0.0)),
            'mask_mAP50_95': float(last.get('val/segm_mAP_50_95', 0.0)),
        }
        logger.info(
            f"📐 RF-DETR Evaluation → mAP50: {metrics['mAP50']:.4f} | "
            f"mAP50-95: {metrics['mAP50_95']:.4f} | "
            f"Seg mAP50: {metrics['mask_mAP50']:.4f}"
        )
        return metrics

    def export(self, format: str = 'onnx'):
        """Exports the RF-DETR Segmentation model to ONNX."""
        if format.lower() != 'onnx':
            logger.warning(f"⚠️ RF-DETR only supports ONNX export natively. Ignoring requested format: {format}")

        logger.info("📦 Exporting RF-DETR Seg to ONNX...")

        # RF-DETR's export method takes an output directory, not a format string
        self.model.export(
            output_dir=str(self.output_dir),
            simplify=True
        )

        onnx_path = self.output_dir / "inference_model.onnx"
        logger.info(f"✅ ONNX Export complete: {onnx_path}")

        # Auto-convert to all deployment formats (optimized ONNX, OpenVINO, TensorRT)
        try:
            from src.inference.export_engine import run_pipeline
            logger.info("🔄 Running deployment format conversions ...")
            run_pipeline(model_dir=self.output_dir, formats={'onnx', 'openvino', 'tensorrt'})
        except Exception as e:
            logger.warning(f"⚠️ Auto-conversion skipped: {e}")

        return str(onnx_path)
