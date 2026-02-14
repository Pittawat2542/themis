"""Utility helpers for configuring package-wide logging."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Mapping

from rich.logging import RichHandler
from rich.traceback import install as install_rich_traceback

from themis.utils import tracing

TRACE_LEVEL = 5
logging.addLevelName(TRACE_LEVEL, "TRACE")


def _trace(self, message, *args, **kwargs):
    if self.isEnabledFor(TRACE_LEVEL):
        self._log(TRACE_LEVEL, message, args, **kwargs)


logging.Logger.trace = _trace  # type: ignore[attr-defined]

_LEVELS: Mapping[str, int] = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
    "trace": TRACE_LEVEL,
}


class JSONFormatter(logging.Formatter):
    """Formatter that outputs JSON lines."""

    def format(self, record: logging.LogRecord) -> str:
        data: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "module": record.module,
            "line": record.lineno,
        }

        # Add trace context if available
        if hasattr(record, "trace_id"):
            data["trace_id"] = record.trace_id  # type: ignore
        if hasattr(record, "span_id"):
            data["span_id"] = record.span_id  # type: ignore

        # Add extra fields
        if hasattr(record, "extra"):
            data.update(record.extra)  # type: ignore

        # Add exception info if present
        if record.exc_info:
            data["exception"] = self.formatException(record.exc_info)

        return json.dumps(data)


class TraceContextFilter(logging.Filter):
    """Filter that injects trace context into log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        span = tracing.get_current_span()
        if span:
            # tracing.py uses span_id for the span's unique ID.
            # We treat the root span's ID as the trace_id if available,
            # but getting the root might be expensive or not thread-safe if implemented poorly.
            # For now, let's just log span_id.
            # Ideally, tracing.py should expose a `trace_id` separately.
            record.span_id = span.span_id  # type: ignore
            # record.trace_id = ... # tracing.py doesn't have a separate trace_id concept yet,
            # other than the root span's ID.

            # Simple workaround: Try to get root span for trace_id
            root = tracing.get_trace()
            if root:
                record.trace_id = root.span_id  # type: ignore

        return True


_CURRENT_LOG_FORMAT = "human"


def get_log_format() -> str:
    """Get the currently configured log format."""
    return _CURRENT_LOG_FORMAT


def configure_logging(
    level: str = "info",
    log_format: str = "human",
    log_file: str | None = None,
) -> None:
    """Configure root logging.

    Args:
        level: Logging level (debug, info, warning, error, critical)
        log_format: "human" (Rich) or "json"
        log_file: Optional path to write logs to
    """
    global _CURRENT_LOG_FORMAT
    _CURRENT_LOG_FORMAT = log_format

    install_rich_traceback()
    numeric_level = _LEVELS.get(level.lower(), logging.INFO)

    handlers: list[logging.Handler] = []

    # Console handler
    if log_format == "json":
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(JSONFormatter())
        handlers.append(console_handler)
    else:
        handlers.append(
            RichHandler(
                rich_tracebacks=True,
                markup=True,
                keywords=[
                    "Orchestrator",
                    "Runner",
                    "Model",
                    "Evaluation",
                    "Generation",
                    "Metric",
                    "Experiment",
                ],
            )
        )

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        if log_format == "json":
            file_handler.setFormatter(JSONFormatter())
        else:
            # Even if console is human, file logs are often better as standard text or JSON
            # For "human" mode, let's just use a standard text formatter for files to avoid control codes
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
            )
        handlers.append(file_handler)

    # Apply configuration
    logging.basicConfig(
        level=numeric_level,
        handlers=handlers,
        force=True,
    )

    # Add filter to root logger and all handlers to ensure it applies
    root_logger = logging.getLogger()
    trace_filter = TraceContextFilter()
    root_logger.addFilter(trace_filter)
    for handler in handlers:
        handler.addFilter(trace_filter)


__all__ = [
    "configure_logging",
    "get_log_format",
    "TRACE_LEVEL",
    "JSONFormatter",
    "TraceContextFilter",
]
