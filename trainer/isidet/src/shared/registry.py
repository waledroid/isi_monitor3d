import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class Registry:
    """Maps string keys to classes for config-driven, decoupled instantiation.

    Used to implement the Strategy pattern throughout IsiDetector.
    Classes register themselves with a decorator at import time; the
    orchestration layer retrieves them by name at runtime.

    Three global instances are created at module level:

    - ``TRAINERS`` — trainer classes (``'yolo'``, ``'rfdetr'``, …)
    - ``HOOKS`` — hook classes (``'IndustrialLogger'``, …)
    - ``PREPROCESSORS`` — preprocessor classes

    Args:
        name: Human-readable label for this registry (used in log
            and error messages).

    Example:
        ```python
        # Registering a new trainer
        @TRAINERS.register('mymodel')
        class MyModelTrainer(BaseTrainer):
            ...

        # Retrieving and instantiating it
        TrainerClass = TRAINERS.get('mymodel')
        trainer = TrainerClass(config)
        ```

        To add a new trainer to the pipeline:

        1. Implement ``BaseTrainer`` in a new file under
           ``src/training/trainers/``.
        2. Decorate the class with ``@TRAINERS.register('name')``.
        3. Add the module path to ``_TRAINER_MODULES`` in
           ``scripts/run_train.py``.
        4. Set ``model_type: "name"`` in ``configs/train.yaml``.
    """

    def __init__(self, name: str):
        self._name = name
        self._module_dict = dict()

    def register(self, name: str = None):
        """Decorator that registers a class under the given key.

        Args:
            name: Registry key. Defaults to the class ``__name__``
                if omitted. Supports multiple registrations on the
                same class (e.g. ``'yolo'`` and ``'yolov26'``).

        Returns:
            The original class unchanged (decorator passthrough).
        """
        def _register(cls):
            module_name = name if name else cls.__name__
            if module_name in self._module_dict:
                logger.warning(f"⚠️ Overwriting existing module '{module_name}' in {self._name} registry!")

            self._module_dict[module_name] = cls
            logger.debug(f"🔌 Registered '{module_name}' into {self._name}")
            return cls
        return _register

    def get(self, name: str):
        """Retrieve a class by its registered key.

        Args:
            name: The key the class was registered under.

        Returns:
            The registered class (not an instance — call it yourself).

        Raises:
            KeyError: If ``name`` is not in the registry. The error
                message includes the list of valid keys.
        """
        if name not in self._module_dict:
            available = list(self._module_dict.keys())
            raise KeyError(f"❌ '{name}' not found in {self._name} registry. Available: {available}")
        return self._module_dict[name]


# Global singletons — import these directly
PREPROCESSORS = Registry('Preprocessors')
TRAINERS = Registry('Trainers')
HOOKS = Registry('Hooks')
