"""Uvicorn entry point — ``python -m isi_gateway``."""

from __future__ import annotations

import logging

import uvicorn

from .app import create_app
from .config import Settings


class _AccessLogFilter(logging.Filter):
    """Suppress high-frequency polling endpoints from the access log."""

    _NOISY_PATHS = ("/healthz",)

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(p in msg for p in self._NOISY_PATHS)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    logging.getLogger("uvicorn.access").addFilter(_AccessLogFilter())
    cfg = Settings()
    app = create_app(cfg)
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="info")


if __name__ == "__main__":
    main()
