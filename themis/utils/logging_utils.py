"""Utility helpers for configuring package-wide logging."""

from __future__ import annotations

import logging
from typing import Mapping

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

    numeric_level = _LEVELS.get(level.lower(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


__all__ = ["configure_logging", "TRACE_LEVEL"]
