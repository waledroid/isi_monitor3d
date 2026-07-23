"""Native TensorRT `.engine` execution behind the ONNX-Runtime session API.

The fast-path answer to minute-long lazy TRT-EP engine builds: a prebuilt,
serialized engine (produced by ``tools/onnx_to_engine.py``) deserializes in
seconds. Every detector plugin talks to its session through exactly three
surfaces — ``get_inputs()/get_outputs()``, ``get_providers()`` and
``run(None, {name: batch})`` — so this class reproduces those three and
``build_onnx_session`` dispatches on the file suffix; the plugins never know
which backend they got.

Portability contract (CLAUDE.md): the ``.onnx`` stays the portable source of
truth — an ``.engine`` is a PER-MACHINE compiled artifact (GPU arch + TensorRT
version specific; the cache files literally encode ``sm120``). Every engine
ships with a sidecar ``<engine>.json`` recording its provenance; loading
validates it and fails with an actionable message instead of a CUDA crash
when an engine is copied to the wrong machine or TRT was upgraded.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_PROVIDER_TAG = "TensorrtEngineFile"


def sidecar_path(engine_path: str | Path) -> Path:
    return Path(str(engine_path) + ".json")


def read_sidecar(engine_path: str | Path) -> dict | None:
    """The engine's provenance record, or None when absent/corrupt."""
    p = sidecar_path(engine_path)
    try:
        return json.loads(p.read_text()) if p.exists() else None
    except (OSError, json.JSONDecodeError):
        logger.warning("trt_session: unreadable sidecar %s", p)
        return None


def validate_sidecar(meta: dict | None, trt_version: str, gpu_name: str | None
                     ) -> list[str]:
    """Mismatch messages (empty = fine). TRT major.minor must match — engines
    do not deserialize across TRT feature releases; GPU mismatch is fatal too
    (an sm120 engine will not run on an Orin). No sidecar → one warning."""
    if meta is None:
        return ["no sidecar metadata (provenance unknown — regenerate with "
                "tools/onnx_to_engine.py to get mismatch protection)"]
    problems: list[str] = []
    built = str(meta.get("tensorrt_version") or "")
    if built and built.split(".")[:2] != trt_version.split(".")[:2]:
        problems.append(
            f"engine built with TensorRT {built}, runtime is {trt_version} "
            "— rebuild the engine (tools/onnx_to_engine.py)")
    want_gpu = str(meta.get("gpu_name") or "")
    if want_gpu and gpu_name and want_gpu != gpu_name:
        problems.append(
            f"engine built for {want_gpu!r}, this machine has {gpu_name!r} "
            "— engines are per-GPU; rebuild locally")
    return problems


class _IoSpec:
    """ORT ``NodeArg`` look-alike: just ``.name`` and ``.shape``."""

    def __init__(self, name: str, shape: list) -> None:
        self.name = name
        self.shape = shape

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"_IoSpec({self.name!r}, {self.shape!r})"


class TrtEngineSession:
    """Executes a serialized TensorRT engine with the ORT session surface."""

    def __init__(self, engine_path: str | Path) -> None:
        import tensorrt as trt
        from cuda.bindings import runtime as cudart

        self._trt = trt
        self._cudart = cudart
        self._path = Path(engine_path)

        meta = read_sidecar(self._path)
        gpu = self._gpu_name()
        problems = validate_sidecar(meta, trt.__version__, gpu)
        hard = [p for p in problems if "rebuild" in p]
        for p in problems:
            logger.warning("trt_session[%s]: %s", self._path.name, p)
        if hard:
            raise RuntimeError(
                f"cannot load {self._path.name}: " + "; ".join(hard))

        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)
        engine = runtime.deserialize_cuda_engine(self._path.read_bytes())
        if engine is None:
            raise RuntimeError(
                f"TensorRT refused to deserialize {self._path.name} — the "
                "engine likely targets another GPU/TRT build; regenerate it "
                "with tools/onnx_to_engine.py")
        self._engine = engine
        self._ctx = engine.create_execution_context()
        err, self._stream = cudart.cudaStreamCreate()
        self._check(err, "cudaStreamCreate")

        self._inputs: list[_IoSpec] = []
        self._outputs: list[_IoSpec] = []
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            shape = [int(d) if int(d) >= 0 else "dyn"
                     for d in engine.get_tensor_shape(name)]
            spec = _IoSpec(name, shape)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self._inputs.append(spec)
            else:
                self._outputs.append(spec)
        # device buffers, (re)allocated per tensor as shapes demand
        self._dev: dict[str, tuple[int, int]] = {}   # name -> (ptr, nbytes)

    # ---- the three ORT surfaces -------------------------------------------

    def get_inputs(self) -> list[_IoSpec]:
        return self._inputs

    def get_outputs(self) -> list[_IoSpec]:
        return self._outputs

    def get_providers(self) -> list[str]:
        return [_PROVIDER_TAG]

    def get_modelmeta(self):
        """ORT look-alike so ``read_embedded_class_names`` works unchanged:
        the sidecar's ``class_names`` surface as the ``names`` metadata the
        Ultralytics export embeds in the source onnx."""
        meta = read_sidecar(self._path) or {}
        names = meta.get("class_names") or []
        mm = type("_ModelMeta", (), {})()
        mm.custom_metadata_map = (
            {"names": repr({i: n for i, n in enumerate(names)})} if names else {})
        return mm

    def run(self, _out_names, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        trt, cudart = self._trt, self._cudart
        for name, arr in feed.items():
            arr = np.ascontiguousarray(arr)
            self._ctx.set_input_shape(name, tuple(arr.shape))
            ptr = self._ensure(name, arr.nbytes)
            err, = cudart.cudaMemcpyAsync(
                ptr, arr.ctypes.data, arr.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self._stream)
            self._check(err, "H2D")
            self._ctx.set_tensor_address(name, ptr)

        outs: list[tuple[np.ndarray, int]] = []
        for spec in self._outputs:
            shape = tuple(int(d) for d in self._ctx.get_tensor_shape(spec.name))
            dtype = np.dtype(trt.nptype(self._engine.get_tensor_dtype(spec.name)))
            host = np.empty(shape, dtype=dtype)
            ptr = self._ensure(spec.name, host.nbytes)
            self._ctx.set_tensor_address(spec.name, ptr)
            outs.append((host, ptr))

        if not self._ctx.execute_async_v3(self._stream):
            raise RuntimeError(f"TensorRT execution failed ({self._path.name})")
        for host, ptr in outs:
            err, = cudart.cudaMemcpyAsync(
                host.ctypes.data, ptr, host.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self._stream)
            self._check(err, "D2H")
        err, = cudart.cudaStreamSynchronize(self._stream)
        self._check(err, "cudaStreamSynchronize")
        # fp16 heads flow into float32 NumPy decode paths transparently
        return [h.astype(np.float32) if h.dtype == np.float16 else h
                for h, _ in outs]

    # ---- helpers -----------------------------------------------------------

    def _ensure(self, name: str, nbytes: int) -> int:
        cur = self._dev.get(name)
        if cur and cur[1] >= nbytes:
            return cur[0]
        cudart = self._cudart
        if cur:
            cudart.cudaFree(cur[0])
        err, ptr = cudart.cudaMalloc(max(nbytes, 1))
        self._check(err, f"cudaMalloc({name})")
        self._dev[name] = (ptr, nbytes)
        return ptr

    def _gpu_name(self) -> str | None:
        try:
            cudart = self._cudart
            err, props = cudart.cudaGetDeviceProperties(0)
            if int(err) == 0:
                name = props.name
                return name.decode() if isinstance(name, bytes) else str(name)
        except Exception:  # pragma: no cover - defensive
            pass
        return None

    def _check(self, err, what: str) -> None:
        if int(err) != 0:
            raise RuntimeError(f"CUDA error in {what}: {err}")

    def __del__(self):  # pragma: no cover - interpreter teardown
        try:
            for ptr, _ in self._dev.values():
                self._cudart.cudaFree(ptr)
            self._cudart.cudaStreamDestroy(self._stream)
        except Exception:
            pass
