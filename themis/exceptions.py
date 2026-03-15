"""Deprecated compatibility shim for legacy exception imports."""

from __future__ import annotations

import warnings

from themis.errors import *  # noqa: F403
from themis.errors import __all__ as _ERRORS_ALL

warnings.warn(
    "themis.exceptions is deprecated; import exception types from themis.errors "
    "or the themis root package instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = list(_ERRORS_ALL)
