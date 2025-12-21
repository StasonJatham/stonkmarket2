"""Structured logging configuration with request ID tracking."""

from __future__ import annotations

import logging
import sys
import json
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .config import settings


# Context variable for request ID tracking
request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


class StructuredFormatter(logging.Formatter):
    """JSON structured log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add request ID if available
        request_id = request_id_var.get()
        if request_id:
            log_data["request_id"] = request_id

        # Add extra fields
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        # Add exception info
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add source location in debug mode
        if settings.debug:
            log_data["location"] = {
                "file": record.filename,
                "line": record.lineno,
                "function": record.funcName,
            }

        return json.dumps(log_data, default=str)


class TextFormatter(logging.Formatter):
    """Human-readable text formatter for development."""

    def format(self, record: logging.LogRecord) -> str:
        request_id = request_id_var.get()
        rid = f"[{request_id[:8]}] " if request_id else ""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        base = f"{timestamp} {record.levelname:8} {rid}{record.name}: {record.getMessage()}"

        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)

        return base


class SensitiveDataFilter(logging.Filter):
    """Filter sensitive data from logs."""

    SENSITIVE_KEYS = {
        "password",
        "password_hash",
        "token",
        "secret",
        "authorization",
        "cookie",
        "session",
        "api_key",
        "access_token",
        "refresh_token",
        "auth_secret",
    }

    def filter(self, record: logging.LogRecord) -> bool:
        # Redact sensitive data from message
        message = record.getMessage().lower()
        for key in self.SENSITIVE_KEYS:
            if key in message:
                # Simple redaction - replace values after common patterns
                record.msg = self._redact_value(str(record.msg), key)
        return True

    def _redact_value(self, text: str, key: str) -> str:
        """Redact values after sensitive keys."""
        import re

        # Match patterns like "key=value" or "key: value" or "'key': 'value'"
        patterns = [
            rf'({key}\s*[=:]\s*)[^\s,}}\]]+',
            rf"('{key}'\s*:\s*)[^\s,}}\]]+",
            rf'("{key}"\s*:\s*)[^\s,}}\]]+',
        ]
        for pattern in patterns:
            text = re.sub(pattern, r"\1[REDACTED]", text, flags=re.IGNORECASE)
        return text


def setup_logging() -> None:
    """Configure application logging."""
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.log_level))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, settings.log_level))

    # Set formatter based on configuration
    if settings.log_format == "json":
        handler.setFormatter(StructuredFormatter())
    else:
        handler.setFormatter(TextFormatter())

    # Add sensitive data filter
    handler.addFilter(SensitiveDataFilter())

    root_logger.addHandler(handler)

    # Configure specific loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the stonkmarket prefix."""
    return logging.getLogger(f"stonkmarket.{name}")


class LoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that includes extra context."""

    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        extra = kwargs.get("extra", {})
        request_id = request_id_var.get()
        if request_id:
            extra["request_id"] = request_id
        if self.extra:
            extra.update(self.extra)
        kwargs["extra"] = extra
        return msg, kwargs
