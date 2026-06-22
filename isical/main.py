"""uvicorn entry — `python -m isical`."""

from __future__ import annotations

import logging

import uvicorn

from .app import create_app
from .config import Settings


class _QuietPolls(logging.Filter):
    """Drop access-log noise from the 1s job-log + capture-status polling."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if '" 200' not in msg:
            return True
        return not any(p in msg for p in ("/api/jobs", "/capture/status", "/stream/"))


def main() -> None:
    cfg = Settings()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    logging.getLogger("uvicorn.access").addFilter(_QuietPolls())
    uvicorn.run(create_app(cfg), host=cfg.host, port=cfg.port, log_level="info")


if __name__ == "__main__":
    main()
