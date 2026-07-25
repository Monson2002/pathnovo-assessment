import contextvars
import json
import logging
import sys
from typing import Any, Dict
from src.config import settings

_correlation_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id", default=""
)


def set_correlation_id(correlation_id: str) -> None:
    """Set the correlation id for all log records emitted in the current context."""
    _correlation_id_var.set(correlation_id)


def get_correlation_id() -> str:
    """Return the correlation id currently set for this context, if any."""
    return _correlation_id_var.get()


class CorrelationIdFilter(logging.Filter):
    """Injects the active correlation id (if any) onto every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        cid = _correlation_id_var.get()
        if cid:
            record.correlation_id = cid
        return True


class JSONFormatter(logging.Formatter):
    """Custom logging Formatter that outputs logs as structured JSON strings."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }

        if hasattr(record, "correlation_id") and getattr(record, "correlation_id"):
            log_entry["correlation_id"] = getattr(record, "correlation_id")

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


def setup_logging(level: str = settings.log_level) -> None:
    """Configure root logger with JSONFormatter."""
    root_logger = logging.getLogger()
    root_logger.setLevel(level.upper())

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(JSONFormatter())
    console_handler.addFilter(CorrelationIdFilter())
    root_logger.addHandler(console_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a logger instance for the given module name."""
    return logging.getLogger(name)
