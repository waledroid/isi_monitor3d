"""Uvicorn entry point — `python -m monitor_web`."""

from __future__ import annotations

import logging

import uvicorn

from .app import create_app
from .config import Settings


class _AccessLogNoiseFilter(logging.Filter):
    """Drop uvicorn access-log lines for the frequently-polled, zero-information
    endpoints so the terminal shows real events instead of a flood of
    ``GET /api/status 200 OK``. The dashboard polls these every 1-5 s."""

    _NOISY_PATHS = ("/api/status", "/api/logs", "/api/calibrate/status")

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if any(p in msg for p in self._NOISY_PATHS):
            return False
        # Static-asset revalidations (304 Not Modified) are pure noise too.
        return not ("/static/" in msg and " 304" in msg)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    logging.getLogger("uvicorn.access").addFilter(_AccessLogNoiseFilter())
    cfg = Settings()
    app = create_app(cfg)
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="info")


if __name__ == "__main__":
    main()
