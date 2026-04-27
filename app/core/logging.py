import logging
import logging.handlers
import sys
from contextvars import ContextVar
from pathlib import Path
from typing import Optional

from pythonjsonlogger.json import JsonFormatter

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class _RequestContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def configure_logging(level: str = "INFO", log_file: Optional[Path] = None) -> None:
    """Configure root + uvicorn loggers to emit JSON lines to stdout and optionally to a file."""
    lvl = getattr(logging, level.upper(), logging.INFO)

    fmt = JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(request_id)s %(message)s",
        rename_fields={"asctime": "timestamp", "levelname": "level", "name": "logger"},
    )
    ctx_filter = _RequestContextFilter()

    handlers: list[logging.Handler] = []

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    stream.addFilter(ctx_filter)
    handlers.append(stream)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        # WatchedFileHandler re-opens the file when its inode changes, so external
        # log rotation (logrotate / Alloy) works correctly across multiple workers.
        # RotatingFileHandler is not safe for multi-process use.
        fh = logging.handlers.WatchedFileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        fh.addFilter(ctx_filter)
        handlers.append(fh)

    root = logging.getLogger()
    root.handlers = handlers
    root.setLevel(lvl)

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        lg = logging.getLogger(name)
        lg.handlers = handlers
        lg.setLevel(lvl)
        lg.propagate = False
