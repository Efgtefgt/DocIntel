"""Structured logging setup.

Log records are emitted as single-line JSON. Callers pass metadata through ``extra=``
(file size, page counts, durations, error codes). Document contents and extracted
field values must never be logged; keep that rule when adding new log statements.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
from datetime import datetime, timezone

from docintel.config.settings import Settings

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {"message", "asctime"}
_configured = False


class JsonFormatter(logging.Formatter):
    """Format records as JSON, including any ``extra`` metadata fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value if isinstance(value, (int, float, bool, str, type(None))) else str(value)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(settings: Settings) -> None:
    """Configure the ``docintel`` logger once per process."""
    global _configured
    if _configured:
        return
    logger = logging.getLogger("docintel")
    logger.setLevel(settings.log_level)
    logger.propagate = False

    console = logging.StreamHandler()
    console.setFormatter(JsonFormatter())
    logger.addHandler(console)

    if settings.log_to_file:
        try:
            settings.log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                settings.log_dir / "docintel.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8"
            )
            file_handler.setFormatter(JsonFormatter())
            logger.addHandler(file_handler)
        except OSError as exc:
            logger.warning("File logging disabled", extra={"reason": str(exc)})
    _configured = True
