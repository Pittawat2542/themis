from __future__ import annotations

import importlib

from themis.errors.exceptions import ThemisError
from themis.types.enums import ErrorCode


def install_hint(extra: str) -> str:
    """Return the canonical install command for one optional dependency group."""
    return f'uv add "themis-eval[{extra}]"'


def import_optional(module_name: str, *, extra: str):
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ThemisError(
            code=ErrorCode.MISSING_OPTIONAL_DEPENDENCY,
            message=(
                f"Optional dependency '{module_name}' is required for this feature. "
                f"Install it with `{install_hint(extra)}`."
            ),
            details={"module": module_name, "extra": extra},
        ) from exc
