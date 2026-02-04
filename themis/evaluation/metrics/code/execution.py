"""Safe code execution for testing functional correctness.

This module executes untrusted generated code in an isolated subprocess with
timeouts and restricted builtins.
"""

from __future__ import annotations

import contextlib
import multiprocessing
import queue
import time
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

from themis.core.entities import MetricScore
from themis.interfaces import Metric


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
    result_queue: Any,
) -> None:
    """Execute untrusted code in isolated process and send a result payload."""
    _apply_memory_limit(max_memory_mb)

    try:
        restricted_globals = {"__builtins__": _safe_builtins()}
        local_vars: dict[str, Any] = {}
        exec(code, restricted_globals, local_vars)

        candidate = local_vars.get(function_name)
        if not callable(candidate):
            result_queue.put(
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

        passed = actual_output == expected_output
        result_queue.put(
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
    except BaseException as exc:
        result_queue.put(
            {
                "status": ExecutionStatus.ERROR.value,
                "passed": False,
                "output": "",
                "error": str(exc),
            }
        )


class ExecutionAccuracy(Metric):
    """Execute code and check against test cases in an isolated sandbox process."""

    requires_reference = True

    def __init__(
        self,
        timeout: float = 3.0,
        max_memory_mb: int = 512,
    ):
        self.name = "execution_accuracy"
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb

    def compute(
        self,
        *,
        prediction: Any,
        references: Sequence[Any],
        metadata: dict[str, Any] | None = None,
    ) -> MetricScore:
        code_str = str(prediction)

        if not references:
            return MetricScore(
                metric_name=self.name,
                value=0.0,
                details={"error": "No test cases provided"},
                metadata=metadata or {},
            )

        test_spec = references[0]
        if not isinstance(test_spec, dict):
            return MetricScore(
                metric_name=self.name,
                value=0.0,
                details={"error": "Test specification must be a dictionary"},
                metadata=metadata or {},
            )

        test_inputs = test_spec.get("inputs", [])
        expected_outputs = test_spec.get("expected", [])
        test_fn_name = test_spec.get("function_name", "solution")

        if len(test_inputs) != len(expected_outputs):
            return MetricScore(
                metric_name=self.name,
                value=0.0,
                details={"error": "Mismatch between inputs and expected outputs"},
                metadata=metadata or {},
            )

        results = []
        for test_input, expected in zip(test_inputs, expected_outputs):
            result = self._execute_test(
                code_str,
                test_fn_name,
                test_input,
                expected,
            )
            results.append(result)

        passed = sum(1 for r in results if r.passed)
        total = len(results)
        accuracy = passed / total if total > 0 else 0.0

        return MetricScore(
            metric_name=self.name,
            value=accuracy,
            details={
                "accuracy": accuracy,
                "passed": passed,
                "total": total,
                "results": [
                    {
                        "status": r.status.value,
                        "passed": r.passed,
                        "error": r.error,
                        "duration": r.duration,
                    }
                    for r in results
                ],
            },
            metadata=metadata or {},
        )

    def _execute_test(
        self,
        code: str,
        function_name: str,
        test_input: Any,
        expected_output: Any,
    ) -> ExecutionResult:
        start = time.perf_counter()
        try:
            ctx = multiprocessing.get_context("fork")
        except ValueError:
            ctx = multiprocessing.get_context("spawn")
        result_queue = ctx.Queue()
        process = ctx.Process(
            target=_execution_worker,
            kwargs={
                "code": code,
                "function_name": function_name,
                "test_input": test_input,
                "expected_output": expected_output,
                "max_memory_mb": self.max_memory_mb,
                "result_queue": result_queue,
            },
        )

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="This process .* is multi-threaded, use of fork\\(\\)",
                    category=DeprecationWarning,
                )
                process.start()
            process.join(self.timeout)

            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
                return ExecutionResult(
                    status=ExecutionStatus.TIMEOUT,
                    passed=False,
                    error=f"Execution timeout ({self.timeout}s)",
                    duration=time.perf_counter() - start,
                )

            payload = result_queue.get_nowait()
            return ExecutionResult(
                status=ExecutionStatus(payload.get("status", ExecutionStatus.ERROR.value)),
                passed=bool(payload.get("passed", False)),
                output=str(payload.get("output", "")),
                error=(
                    str(payload["error"])
                    if payload.get("error") is not None
                    else None
                ),
                duration=time.perf_counter() - start,
            )
        except queue.Empty:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                passed=False,
                error=f"Worker exited with code {process.exitcode}",
                duration=time.perf_counter() - start,
            )
        except BaseException as exc:
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                passed=False,
                error=str(exc),
                duration=time.perf_counter() - start,
            )
        finally:
            with contextlib.suppress(Exception):
                result_queue.close()
            with contextlib.suppress(Exception):
                result_queue.join_thread()


__all__ = ["ExecutionAccuracy", "ExecutionResult", "ExecutionStatus"]
