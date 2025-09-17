from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any


class JsonFormatter(logging.Formatter):
    """Simple JSON log formatter compatible with Grafana/Loki pipelines."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.__dict__:
            extras = {
                key: value
                for key, value in record.__dict__.items()
                if key
                not in {
                    "msg",
                    "args",
                    "levelname",
                    "levelno",
                    "pathname",
                    "filename",
                    "module",
                    "exc_info",
                    "exc_text",
                    "stack_info",
                    "lineno",
                    "funcName",
                    "created",
                    "msecs",
                    "relativeCreated",
                    "thread",
                    "threadName",
                    "processName",
                    "process",
                    "message",
                }
            }
            if extras:
                payload.update(extras)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str, json_output: bool = True) -> None:
    """Configure root logging handler for the trading app."""

    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler()
    if json_output:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
    root.addHandler(handler)
    root.setLevel(level.upper())
