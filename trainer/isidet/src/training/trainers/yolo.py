import sys
import os
import contextlib
import yaml
import logging
import torch
import gc
from pathlib import Path
from ultralytics import YOLO

from src.training.base_trainer import BaseTrainer
from src.shared.registry import TRAINERS

logger = logging.getLogger(__name__)


@TRAINERS.register('yolov26')
@TRAINERS.register('yolo')
class YOLOTrainer(BaseTrainer):
    """Trainer for Ultralytics YOLO (detection or segmentation).

    Registered under ``'yolo'`` and ``'yolov26'`` — both keys resolve to
    this class. Set ``model_type: "yolo"`` in the config to activate it.

    The weights file is config-driven via ``weights`` (e.g. ``yolo11n.pt``
    for YOLOv11 detection), defaulting to ``yolo11{model_size}.pt``. ONNX
    export is config-driven via ``export_nms`` / ``export_opset`` — set
    ``export_nms: false`` + ``export_opset: 17`` to emit the raw head the
    ISI Monitor 3D Backbone's ``yolo_onnx`` detector consumes.

    Key behaviours:

    - Auto-generates ``data/isi_3k_dataset/data.yaml`` from
      ``train.yaml`` on first run (single source of truth for class
      names and dataset paths).
    - Bridges Ultralytics' ``on_train_epoch_end`` callback to
      :meth:`~src.training.base_trainer.BaseTrainer.call_hooks` so all
      registered hooks (e.g. ``IndustrialLogger``) fire correctly.
    - Reads augmentation keys directly from the config dict and injects
      them into ``model.train()`` as keyword arguments.
    - Supports seamless resume: pass ``--resume path/to/last.pt`` to
      ``run_train.py`` and training continues from the last checkpoint.

    Attributes:
        model_size: One of ``'n'``, ``'s'``, ``'m'``, ``'l'``, ``'x'``.
            Determines which pretrained weights to load.
        dataset_path: ``Path`` to the YOLO-format dataset root.
        data_yaml_path: ``str`` path to the generated ``data.yaml``.

    Example:
        ```python
        import yaml
        from scripts.run_train import _deep_merge
        from src.training.trainers.yolo import YOLOTrainer

        with open('isidet/configs/train.yaml') as f:
            config = yaml.safe_load(f)
        with open('isidet/configs/optimizers/yolo_optim.yaml') as f:
            config = _deep_merge(config, yaml.safe_load(f))

        trainer = YOLOTrainer(config)
        trainer.train()
        metrics = trainer.evaluate()
        trainer.export(format='onnx')
        ```
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.model_size = config.get('model_size', 'm')
        self.dataset_path = Path(config.get('dataset_path', 'data/isi_3k_dataset'))
        self.data_yaml_path = self._prepare_data_yaml()

    def _prepare_data_yaml(self) -> str:
        """Generate the ``data.yaml`` that Ultralytics requires, if missing.

        Reads ``nc`` and ``class_names`` from ``train.yaml``, making the
        config the single source of truth. Does nothing if the file
        already exists.

        Returns:
            Absolute path to ``data.yaml`` as a string.
        """
        yaml_path = self.dataset_path / 'data.yaml'

        if not yaml_path.exists():
            data_dict = {
                'path': str(self.dataset_path.absolute()),
                'train': 'images/train',
                'val': 'images/val',
                'test': 'images/test',
                'nc': self.config.get('nc', 2),
                'names': self.config.get('class_names', ['carton', 'polybag'])
            }
            # Pose models need keypoint metadata in data.yaml. Detect a pose run
            # from the weights suffix (Ultralytics' own task signal) and inject
            # kpt_shape + flip_idx, both overridable from the train config. Default
            # to the COCO-17 layout (ankles 15/16 are the foot nodes we project).
            if '-pose' in str(self.config.get('weights', '')):
                data_dict['kpt_shape'] = self.config.get('kpt_shape', [17, 3])
                data_dict['flip_idx'] = self.config.get(
                    'flip_idx',
                    [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15],
                )
            with open(yaml_path, 'w') as f:
                yaml.dump(data_dict, f, default_flow_style=False)
            logger.info(f"📄 Generated YOLO data.yaml at {yaml_path} with classes: {data_dict['names']}")

        return str(yaml_path)

    def build_model(self):
        """Load YOLO weights — fresh pretrained or from a checkpoint.

        Checks ``config['resume_path']`` (set by ``--resume`` CLI flag).
        If present, loads that checkpoint; otherwise loads the weights named
        by ``config['weights']`` (e.g. ``yolo11n.pt`` for YOLOv11 detection),
        falling back to ``yolo11{model_size}.pt`` when ``weights`` is unset.
        Ultralytics auto-downloads the pretrained file on first use.
        """
        resume_path = self.config.get('resume_path')

        if resume_path:
            logger.info(f"🏗️ Loading Checkpoint for Resume: {resume_path}")
            self.model = YOLO(resume_path)
        else:
            model_name = self.config.get('weights') or f"yolo11{self.model_size}.pt"
            logger.info(f"🏗️ Building Fresh Model: {model_name}")
            self.model = YOLO(model_name)

    def _inject_framework_hooks(self):
        """Bridge Ultralytics ``on_train_epoch_end`` to BaseTrainer hooks.

        Wires an Ultralytics callback that:

        1. Copies ``trainer.epoch`` → ``self.current_epoch``.
        2. Extracts the scalar total loss from ``trainer.tloss``.
        3. Extracts per-component losses from ``trainer.loss_items``
           into ``self.loss_components`` (keys: ``box``, ``seg``,
           ``cls``, ``dfl``).
        4. Calls ``self.call_hooks('after_epoch')``.

        This is the bridge between Ultralytics' internal callback
        system and IsiDetector's hook system.
        """
        def on_train_epoch_end(trainer):
            self.current_epoch = trainer.epoch

            if hasattr(trainer, 'tloss') and trainer.tloss is not None:
                self.current_loss = float(trainer.tloss.sum())
            else:
                self.current_loss = 0.0

            # Populate per-component losses for IndustrialLogger. Detection
            # YOLO reports 3 items (box, cls, dfl); segmentation reports 4
            # (box, seg, cls, dfl). Pick the name set by length so labels stay
            # correct across both model types.
            if hasattr(trainer, 'loss_items') and trainer.loss_items is not None:
                try:
                    items = trainer.loss_items.tolist()
                    names = ['box', 'seg', 'cls', 'dfl'] if len(items) == 4 else ['box', 'cls', 'dfl']
                    self.loss_components = {k: float(v) for k, v in zip(names, items)}
                except Exception:
                    self.loss_components = {}

            self.call_hooks('after_epoch')

        self.model.add_callback("on_train_epoch_end", on_train_epoch_end)

    def train(self):
        """Run the full YOLO training pipeline.

        Execution order:

        1. :meth:`build_model` (if model not already loaded).
        2. :meth:`_setup_run_dir` — creates ``models/yolo/DD-MM-YYYY/``.
        3. :meth:`_inject_framework_hooks` — wires Ultralytics callbacks.
        4. ``call_hooks('before_train')``.
        5. ``model.train(...)`` — full Ultralytics training loop.
        6. ``call_hooks('after_train')``.

        All hyperparameters (lr, scheduler, augmentations, early stopping)
        are read from the merged config. Augmentation keys present in the
        config (``fliplr``, ``mosaic``, ``hsv_h``, etc.) are injected
        automatically into Ultralytics via ``**kwargs``.

        Note:
            Workers are locked at 2 for WSL memory stability.
            ``device=0`` targets the first GPU.
        """
        if self.model is None:
            self.build_model()

        # 1. Run dir = "<model>_e<epochs>_<imgsz>px_<timestamp>" so the folder shows
        #    which model version + epoch count + input size produced it (imgsz is more
        #    useful at a glance than the date), and same-day runs don't clash.
        #    e.g. models/yolo/yolo26l_e200_640px_21-05-2026_21-55-41/
        weights = self.config.get('weights') or f"yolo11{self.model_size}.pt"
        model_ver = Path(weights).stem                       # yolo11l, yolo26l, ...
        epochs = self.config.get('epochs', 300)
        img_size = self.config.get('imgsz') or self.config.get('image_size', 640)
        run_tag = f"{model_ver}_e{epochs}_{img_size}px"
        self._setup_run_dir(fmt="%d-%m-%Y_%H-%M-%S", tag=run_tag)

        self._inject_framework_hooks()

        # 2. Base parameters
        epochs = self.config.get('epochs', 300)
        batch_size = self.config.get('batch_size', 16)
        img_size = self.config.get('imgsz') or self.config.get('image_size', 640)

        # 3. Augmentation keys (YOLO-specific, pulled dynamically from config)
        yolo_aug_keys = [
            'hsv_h', 'hsv_s', 'hsv_v',
            'degrees', 'translate', 'scale', 'shear', 'perspective',
            'fliplr', 'flipud',
            'mosaic', 'close_mosaic', 'mixup', 'copy_paste', 'erasing',
        ]
        yolo_kwargs = {k: v for k, v in self.config.items() if k in yolo_aug_keys}

        # Optional: inject camera-feed degradation transforms (motion blur, JPEG
        # compression, sensor noise, downscale) into Ultralytics' Albumentations
        # pipeline so training mimics the low-res H.264 sub-stream the Backbone
        # actually infers on. Best-effort: guarded so an Ultralytics-internal
        # change can never break the training run.
        if self.config.get('camera_aug', False):
            self._inject_camera_augmentations()

        is_resuming = bool(self.config.get('resume_path'))

        if is_resuming:
            logger.info(f"⏩ Fast-forwarding training to interrupted epoch (Max {epochs})...")
        else:
            logger.info(f"🔥 Starting Fresh YOLO training for {epochs} epochs at {img_size}px...")

        if yolo_kwargs:
            logger.info(f"🧬 Applied Augmentations: {yolo_kwargs}")

        logger.info(f"📂 Outputting logs & weights to: {self.output_dir}")
        self.call_hooks('before_train')

        # 4. Calculate YOLO's lrf (lrf = min_lr / base_lr)
        sched_cfg = self.optim_cfg.get('scheduler', {})
        lr0 = self.optim_cfg.get('lr', 0.01)
        eta_min = sched_cfg.get('eta_min', 0.0001)
        lrf = (eta_min / lr0) if lr0 > 0 else 0.01

        # 5. Launch native Ultralytics training
        self.model.train(
            data=self.data_yaml_path,
            epochs=epochs,
            resume=is_resuming,
            batch=batch_size,
            imgsz=img_size,

            project=str(self.output_dir.parent),
            name=self.output_dir.name,
            exist_ok=True,

            device=0,
            amp=self.config.get('mixed_precision', True),
            # Dataloader workers — each holds its own augmentation pipeline +
            # buffered (mosaic) batches, so this is a real host-RAM lever on WSL.
            # Lower to 1 (or 0) if the WSL VM thrashes swap during training.
            workers=self.config.get('workers', 2),
            verbose=False,
            plots=True,

            optimizer=self.optim_cfg.get('type', 'auto'),
            lr0=lr0,
            lrf=lrf,
            weight_decay=self.optim_cfg.get('weight_decay', 0.0005),
            warmup_epochs=sched_cfg.get('warmup_epochs', 3.0),
            cos_lr=(sched_cfg.get('type') == 'CosineAnnealing'),

            patience=self.es_cfg.get('patience', 50) if self.es_cfg.get('enabled', True) else 0,
            save_period=self.ckpt_cfg.get('save_frequency', -1),

            **yolo_kwargs
        )

        self.call_hooks('after_train')

    def _inject_camera_augmentations(self) -> None:
        """Add warehouse camera-feed degradations to Ultralytics' Albumentations.

        Ultralytics builds a fixed ``Albumentations`` transform internally and
        exposes no config knob for it. We monkey-patch that class's ``__init__``
        so that — in addition to its defaults — every training image may be
        motion-blurred, JPEG-recompressed, noised, and downscaled, mimicking the
        low-res H.264 sub-stream the Backbone infers on. This is the single
        highest-value domain adaptation for this deployment.

        Best-effort and fully guarded: any import/internals change just logs a
        warning and leaves the native augmentations in place.
        """
        try:
            import albumentations as A
            from ultralytics.data import augment as ua

            extra = [
                A.MotionBlur(blur_limit=(3, 7), p=0.2),
                A.ImageCompression(quality_range=(30, 70), p=0.2),
                A.GaussNoise(std_range=(0.02, 0.12), p=0.2),
                # More aggressive downscale → simulates LONG-DISTANCE / low-res
                # capture so the model learns small, blurry far-away pallets.
                A.Downscale(scale_range=(0.35, 0.85), p=0.25),
            ]
            _orig_init = ua.Albumentations.__init__

            # Forward *all* args (Ultralytics 8.4's signature is
            # ``__init__(self, p=1.0, transforms=None)``), then append our
            # extra image-level transforms. Ultralytics' Albumentations is
            # applied to the image only, so we re-compose WITHOUT bbox_params.
            def _patched_init(alb_self, *args, **kwargs):
                _orig_init(alb_self, *args, **kwargs)
                try:
                    t = getattr(alb_self, "transform", None)
                    if t is not None:
                        alb_self.transform = A.Compose(list(t.transforms) + extra)
                        logger.info("🎥 Camera-feed augmentations injected (blur/JPEG/noise/downscale).")
                except Exception as exc:   # noqa: BLE001 — never block training
                    logger.warning(f"⚠️ Camera-aug compose failed, using defaults: {exc}")

            ua.Albumentations.__init__ = _patched_init
        except Exception as exc:   # noqa: BLE001
            logger.warning(f"⚠️ Camera-aug injection skipped ({exc}); native augmentations only.")

    def evaluate(self) -> dict:
        """Run post-training validation with WSL-safe memory management.

        Detection-only (no masks). Flushes GPU and RAM before validation to
        avoid OOM on the memory spike caused by loading the full validation
        set. Stdout is suppressed to hide Ultralytics' progress bar glitches.

        Returns:
            A metrics dictionary with keys:

            - ``'mAP50'`` — bounding-box mAP @ IoU 0.50
            - ``'mAP50_95'`` — bounding-box mAP @ IoU 0.50–0.95
            - ``'speed_ms'`` — total inference time per image in ms
        """
        logger.info("📐 Running Lightweight Evaluation (Reduced Workers)...")
        self._flush_memory()

        with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull):
            results = self.model.val(
                data=self.data_yaml_path,
                batch=8,    # Lower batch for final validation to prevent OOM
                workers=2,
                verbose=False
            )

        box_map50 = results.box.map50
        box_map = results.box.map
        speed_ms = sum(results.speed.values())

        print("\n" + "═" * 45)
        print(f" {'VALIDATION SUMMARY (detection)':^43} ")
        print("═" * 45)
        print(f" {'mAP @ 50':<22} | {box_map50:>14.4f}")
        print(f" {'mAP @ 50-95':<22} | {box_map:>14.4f}")
        print(f" {'Inference speed (ms)':<22} | {speed_ms:>14.2f}")
        print("═" * 45 + "\n")

        return {
            'mAP50': float(box_map50),
            'mAP50_95': float(box_map),
            'speed_ms': float(speed_ms),
        }

    def export(self, format: str = 'onnx'):
        """Export trained weights to a deployment format.

        ONNX export flags are config-driven:

        - ``export_nms`` (default ``True``): bake NMS into the graph for a
          self-contained model. **Set ``False``** to emit the raw YOLO head
          ``(1, 4+nc, 8400)`` that the ISI Monitor 3D Backbone's ``yolo_onnx``
          detector expects (it runs its own NMS in ``detection/postprocess.py``).
        - ``export_opset`` (default ``12``): use ``17`` for the Backbone path.

        Args:
            format: Export format. ``'onnx'`` (default) uses the flags above
                plus ``simplify=True``, ``dynamic=False``. Any other string is
                passed directly to Ultralytics.

        Returns:
            Path to the exported model file as a string.
        """
        logger.info(f"📦 Exporting model to {format} for production...")
        img_size = self.config.get('imgsz') or self.config.get('image_size', 640)

        if format == 'onnx':
            nms = self.config.get('export_nms', True)
            opset = self.config.get('export_opset', 12)
            # dynamic=False → input locked to imgsz (fastest, but inference size is
            # fixed). dynamic=True → the ONNX accepts any 32-multiple size, letting
            # the runtime pick the inference resolution (the dashboard's imgsz
            # slider) to trade speed for accuracy. Slightly slower at imgsz than a
            # static export, but far more flexible.
            dynamic = bool(self.config.get('export_dynamic', False))
            logger.info(f"   ONNX flags: imgsz={img_size}, nms={nms}, opset={opset}, "
                        f"dynamic={dynamic} "
                        f"({'self-contained' if nms else 'raw head — Backbone yolo_onnx compatible'})")
            export_path = self.model.export(
                format='onnx',
                imgsz=img_size,
                opset=opset,
                nms=nms,
                simplify=True,    # Graph optimization
                dynamic=dynamic,
            )
        else:
            export_path = self.model.export(format=format)

        logger.info(f"✅ ONNX export complete: {export_path}")

        # Deployment formats are config-driven. The .pt is the training
        # checkpoint (always written by Ultralytics); ONNX is produced above;
        # anything else (openvino, tensorrt) is generated from the ONNX via the
        # export engine. Default: onnx + openvino (no tensorrt).
        deploy = set(self.config.get('export_formats', ['onnx', 'openvino'])) - {'onnx', 'pt'}
        if deploy:
            try:
                from src.inference.export_engine import run_pipeline
                logger.info(f"🔄 Converting to deployment formats: {sorted(deploy)} ...")
                run_pipeline(model_dir=Path(export_path).parent, formats=deploy, imgsz=img_size)
            except Exception as e:
                logger.warning(f"⚠️ Deployment-format conversion skipped: {e}")

        return export_path
