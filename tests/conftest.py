import pytest


@pytest.fixture(autouse=True)
def _isolate_trt_env(monkeypatch):
    """The TRT opt-in travels as ISI3D_TRT (set by Orchestrator/isistream
    entry points). Building an orchestrator in one test must not leak the
    flag into the next test's session construction."""
    monkeypatch.delenv("ISI3D_TRT", raising=False)
