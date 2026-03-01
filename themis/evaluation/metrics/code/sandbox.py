"""Safe code execution sandbox capabilities.

This module provides the low-level functions to execute untrusted generated code
in an isolated subprocess with timeouts and restricted builtins.
"""

from __future__ import annotations

import contextlib
import time
import tracemalloc
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ExecutionStatus(str, Enum):
    """Execution result status."""

    PASSED = "passed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class ExecutionResult:
    """Result of code execution."""

    status: ExecutionStatus
    passed: bool
    output: str = ""
    error: str | None = None
    duration: float = 0.0


def _blocked_import(*_args: Any, **_kwargs: Any):
    raise ImportError("Imports are disabled in execution sandbox")


def _safe_builtins() -> dict[str, Any]:
    """Return safe builtins for sandbox execution."""
    return {
        "__import__": _blocked_import,
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "filter": filter,
        "float": float,
        "int": int,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "pow": pow,
        "range": range,
        "reversed": reversed,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
        "Exception": Exception,
        "ValueError": ValueError,
        "TypeError": TypeError,
        "ArithmeticError": ArithmeticError,
        "ZeroDivisionError": ZeroDivisionError,
        "AssertionError": AssertionError,
        "MemoryError": MemoryError,
    }


def _apply_memory_limit(max_memory_mb: int) -> None:
    """Apply best-effort process memory limits."""
    if max_memory_mb <= 0:
        return

    try:
        import resource
    except Exception:
        return

    max_bytes = max_memory_mb * 1024 * 1024
    for limit_name in ("RLIMIT_AS", "RLIMIT_DATA"):
        limit = getattr(resource, limit_name, None)
        if limit is None:
            continue
        try:
            resource.setrlimit(limit, (max_bytes, max_bytes))
        except Exception:
            # Some platforms disallow changing specific limits.
            pass


def _execution_worker(
    *,
    code: str,
    function_name: str,
    test_input: Any,
    expected_output: Any,
    max_memory_mb: int,
    result_conn: Any,
) -> None:
    """Execute untrusted code in isolated process and send a result payload."""
    _apply_memory_limit(max_memory_mb)

    try:
        if max_memory_mb > 0:
            tracemalloc.start()

        restricted_globals = {"__builtins__": _safe_builtins()}
        local_vars: dict[str, Any] = {}
        exec(code, restricted_globals, local_vars)

        candidate = local_vars.get(function_name)
        if not callable(candidate):
            result_conn.send(
                {
                    "status": ExecutionStatus.ERROR.value,
                    "passed": False,
                    "output": "",
                    "error": f"Function '{function_name}' not found",
                }
            )
            return

        if isinstance(test_input, (list, tuple)):
            actual_output = candidate(*test_input)
        else:
            actual_output = candidate(test_input)

        if max_memory_mb > 0:
            _current_bytes, peak_bytes = tracemalloc.get_traced_memory()
            max_bytes = max_memory_mb * 1024 * 1024
            if peak_bytes > max_bytes:
                result_conn.send(
                    {
                        "status": ExecutionStatus.ERROR.value,
                        "passed": False,
                        "output": "",
                        "error": (
                            f"Memory limit exceeded: peak={peak_bytes} bytes, "
                            f"limit={max_bytes} bytes"
                        ),
                    }
                )
                return

        passed = actual_output == expected_output
        result_conn.send(
            {
                "status": (
                    ExecutionStatus.PASSED.value
                    if passed
                    else ExecutionStatus.FAILED.value
                ),
                "passed": passed,
                "output": str(actual_output),
                "error": None,
            }
        )
    except MemoryError:
        result_conn.send(
            {
                "status": ExecutionStatus.ERROR.value,
                "passed": False,
                "output": "",
                "error": "Memory limit exceeded",
            }
        )
    except BaseException as exc:
        result_conn.send(
            {
                "status": ExecutionStatus.ERROR.value,
                "passed": False,
                "output": "",
                "error": str(exc),
            }
        )
    finally:
        with contextlib.suppress(RuntimeError):
            tracemalloc.stop()
        with contextlib.suppress(Exception):
            result_conn.close()


def _execution_batch_worker(
    *,
    code: str,
    function_name: str,
    test_inputs: Sequence[Any],
    expected_outputs: Sequence[Any],
    max_memory_mb: int,
    result_conn: Any,
) -> None:
    """Execute all tests in one isolated process to reduce fork overhead."""
    _apply_memory_limit(max_memory_mb)
    try:
        if max_memory_mb > 0:
            tracemalloc.start()
        max_bytes = max_memory_mb * 1024 * 1024 if max_memory_mb > 0 else None

        restricted_globals = {"__builtins__": _safe_builtins()}
        local_vars: dict[str, Any] = {}
        exec(code, restricted_globals, local_vars)
        candidate = local_vars.get(function_name)
        if not callable(candidate):
            result_conn.send(
                {
                    "status": "error",
                    "error": f"Function '{function_name}' not found",
                    "results": [],
                }
            )
            return

        results: list[dict[str, Any]] = []
        for test_input, expected_output in zip(test_inputs, expected_outputs):
            case_start = time.perf_counter()
            try:
                if isinstance(test_input, (list, tuple)):
                    actual_output = candidate(*test_input)
                else:
                    actual_output = candidate(test_input)
                passed = actual_output == expected_output
                status = (
                    ExecutionStatus.PASSED.value
                    if passed
                    else ExecutionStatus.FAILED.value
                )
                error = None
                output = str(actual_output)
            except MemoryError:
                passed = False
                status = ExecutionStatus.ERROR.value
                error = "Memory limit exceeded"
                output = ""
            except BaseException as exc:
                passed = False
                status = ExecutionStatus.ERROR.value
                error = str(exc) or exc.__class__.__name__
                output = ""

            if max_bytes is not None:
                _current_bytes, peak_bytes = tracemalloc.get_traced_memory()
                if peak_bytes > max_bytes:
                    passed = False
                    status = ExecutionStatus.ERROR.value
                    error = (
                        f"Memory limit exceeded: peak={peak_bytes} bytes, "
                        f"limit={max_bytes} bytes"
                    )
                    output = ""

            results.append(
                {
                    "status": status,
                    "passed": passed,
                    "error": error,
                    "output": output,
                    "duration": time.perf_counter() - case_start,
                }
            )

        result_conn.send({"status": "ok", "results": results})
    except MemoryError:
        result_conn.send(
            {"status": "error", "error": "Memory limit exceeded", "results": []}
        )
    except BaseException as exc:
        result_conn.send({"status": "error", "error": str(exc), "results": []})
    finally:
        with contextlib.suppress(RuntimeError):
            tracemalloc.stop()
        with contextlib.suppress(Exception):
            result_conn.close()
