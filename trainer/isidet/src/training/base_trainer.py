import gc
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

from src.shared.registry import HOOKS

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """Abstract base class that every IsiDetector trainer must inherit from.

    Provides shared infrastructure so concrete trainers only implement
    model-specific logic. Handles output directory management, config
    parsing, hook lifecycle, and memory utilities.

    The orchestration entry point (`run_train.py`) calls `.train()`,
    `.evaluate()`, and `.export()` without knowing which model is
    underneath — this class is that contract.

    Attributes:
        config: The fully-merged config dictionary (train.yaml +
            optimizer yaml).
        model_name: Value of ``config['model_type']`` (e.g. ``"yolo"``).
        output_dir: Timestamped ``Path`` where weights and logs are
            written. Set to the base directory on construction; updated
            to a run-stamped subdirectory by :meth:`_setup_run_dir`
            at the start of :meth:`train`.
        model: The underlying framework model object (``None`` until
            :meth:`build_model` is called).
        current_epoch: Epoch index updated each epoch by the concrete
            trainer. Read by hooks such as ``IndustrialLogger``.
        current_loss: Scalar total loss for the current epoch. Updated
            by the concrete trainer.
        loss_components: Per-component losses for the current epoch
            (e.g. ``{'box': 0.4, 'seg': 0.2, 'cls': 0.1, 'dfl': 0.3}``).
            Updated inside :meth:`_inject_framework_hooks`.
        hooks: List of instantiated hook objects, built from
            ``config['hooks']`` at construction time.
        optim_cfg: ``config['optimizer']`` sub-dict (pre-parsed).
        es_cfg: ``config['early_stopping']`` sub-dict (pre-parsed).
        ckpt_cfg: ``config['checkpoint']`` sub-dict (pre-parsed).
        tricks_cfg: ``config['training_tricks']`` sub-dict (pre-parsed).

    Example:
        ```python
        from src.training.trainers.yolo import YOLOTrainer

        trainer = YOLOTrainer(merged_config)
        trainer.train()
        trainer.evaluate()
        trainer.export(format='onnx')
        ```
    """

    def __init__(self, config: dict):
        self.config = config
        self.model_name = config.get('model_type', 'unknown_model')

        # 1. Base output directory (trainers call _setup_run_dir() at the start of train())
        self.output_dir = Path(config.get('output_dir', 'models')) / self.model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 2. Universal state variables (populated during training)
        self.model = None
        self.current_epoch = 0
        self.current_loss = 0.0
        self.loss_components: dict = {}  # e.g. {'box': 0.4, 'seg': 0.2, 'cls': 0.1, 'dfl': 0.3}

        # 3. Pre-parse shared config sub-sections
        self._parse_common_config()

        # 4. Initialize hooks dynamically from the YAML config
        self.hooks = []
        hook_names = config.get('hooks', [])
        for h_name in hook_names:
            try:
                hook_class = HOOKS.get(h_name)
                self.hooks.append(hook_class())
                logger.debug(f"🪝 Attached Hook: {h_name}")
            except KeyError:
                logger.error(f"❌ Hook '{h_name}' not found in Registry — skipping.")

    def _parse_common_config(self) -> None:
        """Pre-parse shared config sub-sections into instance attributes.

        Called automatically in ``__init__``. Trainers use
        ``self.optim_cfg``, ``self.es_cfg``, ``self.ckpt_cfg``, and
        ``self.tricks_cfg`` instead of calling ``self.config.get(...)``
        repeatedly throughout their code.
        """
        self.optim_cfg = self.config.get('optimizer', {})
        self.es_cfg = self.config.get('early_stopping', {})
        self.ckpt_cfg = self.config.get('checkpoint', {})
        self.tricks_cfg = self.config.get('training_tricks', {})

    def _setup_run_dir(self, fmt: str = "%d-%m-%Y", tag: str | None = None) -> None:
        """Create a timestamped output sub-directory and update ``self.output_dir``.

        Must be called at the **start** of :meth:`train`, before
        :meth:`call_hooks` fires ``before_train``, so that hooks
        reading ``trainer.output_dir`` see the correct path.

        Args:
            fmt: ``datetime.strftime`` format string for the sub-folder
                name. Defaults to ``"%d-%m-%Y"`` (day-month-year).
                RF-DETR uses ``"%d-%m-%Y_%H%M"`` for finer granularity.
            tag: Optional descriptive prefix for the run folder (e.g.
                ``"yolo11l_e100"``), so the dir reads
                ``<tag>_<timestamp>`` instead of just ``<timestamp>``.

        Example:
            ```python
            # Inside a concrete trainer's train():
            self._setup_run_dir(fmt="%d-%m-%Y", tag="yolo11l_e100")
            # self.output_dir is now e.g. models/yolo/yolo11l_e100_02-04-2026/
            ```
        """
        run_date = datetime.now().strftime(fmt)
        run_name = f"{tag}_{run_date}" if tag else run_date
        base_project_dir = Path(self.config.get('output_dir', 'models')) / self.model_name
        self.output_dir = base_project_dir / run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _flush_memory(self) -> None:
        """Release GPU cache and run Python garbage collection.

        Safe to call at any point. No-op if CUDA is unavailable.
        Called periodically in the inference loop (~every 60 s) and
        before/after heavy operations like evaluation.
        """
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def call_hooks(self, stage: str) -> None:
        """Broadcast a training stage event to all registered hooks.

        Each hook is isolated — if a hook raises an exception it is
        logged and training continues uninterrupted.

        Args:
            stage: One of ``'before_train'``, ``'before_epoch'``,
                ``'after_epoch'``, or ``'after_train'``. Hooks only
                need to implement the stages they care about.

        Note:
            Hooks receive the trainer instance as their sole argument,
            giving them read access to ``current_epoch``,
            ``current_loss``, ``loss_components``, ``output_dir``,
            and the full ``config``.
        """
        for hook in self.hooks:
            if hasattr(hook, stage):
                try:
                    getattr(hook, stage)(self)
                except Exception as e:
                    logger.error(f"❌ Hook '{type(hook).__name__}.{stage}' raised: {e}")

    # ==========================================
    # ENFORCED METHODS (Children MUST implement)
    # ==========================================

    @abstractmethod
    def build_model(self):
        """Initialise the model architecture and load weights to the GPU.

        Called automatically by :meth:`train` if ``self.model`` is
        ``None``. May also be called explicitly before training.
        Must set ``self.model`` to a usable object.
        """
        pass

    @abstractmethod
    def _inject_framework_hooks(self):
        """Wire this framework's native callbacks to :meth:`call_hooks`.

        Must be called at the start of :meth:`train`, after
        :meth:`build_model` and before the native training loop begins.

        **Contract:** at minimum, the wired callback must:

        1. Update ``self.current_epoch`` once per epoch.
        2. Update ``self.current_loss`` (scalar total loss).
        3. Populate ``self.loss_components`` with per-component losses
           when available (keys: ``'box'``, ``'seg'``, ``'cls'``, ``'dfl'``).
        4. Call ``self.call_hooks('after_epoch')``.

        This method is abstract so that forgetting to implement it
        raises ``TypeError`` at instantiation — hooks are guaranteed
        to fire if the class can be constructed.
        """
        pass

    @abstractmethod
    def train(self):
        """Run the full training pipeline.

        Must call ``self.call_hooks('before_train')`` before the loop
        and ``self.call_hooks('after_train')`` after. Must also call
        :meth:`_setup_run_dir` and :meth:`_inject_framework_hooks`
        before the native training loop starts.
        """
        pass

    @abstractmethod
    def evaluate(self) -> dict:
        """Run post-training validation and return a metrics dictionary.

        Returns:
            A dict containing at minimum::

                {'mAP50': float, 'mAP50_95': float}

            Optional keys: ``'mask_mAP50'``, ``'mask_mAP50_95'``,
            ``'speed_ms'``. Hooks and downstream tooling may rely on
            ``mAP50`` being present.
        """
        pass

    @abstractmethod
    def export(self, format: str = 'onnx'):
        """Export trained weights to a deployment format.

        Args:
            format: Target format string. ``'onnx'`` is supported by
                both trainers. YOLO additionally supports any format
                accepted by Ultralytics (``'openvino'``, ``'tflite'``,
                etc.).

        Returns:
            Path to the exported model file as a string.
        """
        pass
