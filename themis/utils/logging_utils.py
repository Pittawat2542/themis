"""Utility helpers for configuring package-wide logging."""

from __future__ import annotations

import logging
from typing import Mapping

from rich.logging import RichHandler
from rich.traceback import install as install_rich_traceback

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


def configure_logging(level: str = "info") -> None:
    """Configure root logging with human-friendly formatting."""
    install_rich_traceback()
    numeric_level = _LEVELS.get(level.lower(), logging.INFO)
    
    logging.basicConfig(
        level=numeric_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, markup=True)],
        force=True,
    )


__all__ = ["configure_logging", "TRACE_LEVEL"]
