"""Shared helpers for Cyclopts-backed Themis CLIs."""

from __future__ import annotations

from collections.abc import Iterable
import sys
from typing import Any

from cyclopts import App
from cyclopts.exceptions import CycloptsError


def _system_exit_code(code: object) -> int:
    """Normalize ``SystemExit.code`` to a shell exit status."""

    if code is None:
        return 0
    if isinstance(code, int):
        return code
    return 1


def invoke_app(app: App, argv: Iterable[str] | None = None) -> int:
    """Execute one Cyclopts app and normalize shell-compatible exit statuses."""

    tokens = list(sys.argv[1:]) if argv is None else list(argv)
    try:
        result = app(
            tokens,
            exit_on_error=False,
            print_error=True,
            result_action="return_value",
        )
    except CycloptsError:
        return 1
    except SystemExit as exc:
        return _system_exit_code(exc.code)

    if result is None:
        return 0
    if isinstance(result, int):
        return result
    return 0


def command_name(function_name: str) -> str:
    """Convert one Python function name into the public hyphenated command."""

    return function_name.replace("_", "-")


def normalize_result(result: Any) -> int:
    """Normalize command return values into shell-compatible exit codes."""

    if result is None:
        return 0
    if isinstance(result, int):
        return result
    return 0
