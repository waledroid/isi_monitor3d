import pytest


@pytest.fixture(autouse=True)
def _isolate_trt_env(monkeypatch):
    """Legacy guard: ISI3D_TRT is no longer read (TensorRT = native .engine
    paths only), but a stale value in a dev shell must not resurrect the old
    TRT-EP behaviour if an older branch's code runs under this tree."""
    monkeypatch.delenv("ISI3D_TRT", raising=False)
