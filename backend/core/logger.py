"""
Structured JSON logging for the Financial RAG Assistant.
Every module should import get_logger() and use it — never use print() for operational output.
"""
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


class JsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON for easy grep / post-processing."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Merge in any extra fields passed via extra={...} on the log call
        for key, val in record.__dict__.items():
            if key not in {
                "args", "asctime", "created", "exc_info", "exc_text", "filename",
                "funcName", "id", "levelname", "levelno", "lineno", "module",
                "msecs", "message", "msg", "name", "pathname", "process",
                "processName", "relativeCreated", "stack_info", "thread",
                "threadName",
            }:
                payload[key] = val
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def get_logger(name: str) -> logging.Logger:
    """
    Return a module-level logger that writes JSON to stdout and optionally to a file.

    Usage:
        from backend.core.logger import get_logger
        logger = get_logger(__name__)
        logger.info("node_ran", extra={"node": "retriever", "chunks": 5})
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured (avoids duplicate handlers in Streamlit reloads)

    logger.setLevel(logging.DEBUG)
    formatter = JsonFormatter()

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Optional file handler
    log_file = os.getenv("LOG_FILE")
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.propagate = False
    return logger
