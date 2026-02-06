"""Safe code execution for testing functional correctness.

This module executes untrusted generated code in an isolated subprocess with
timeouts and restricted builtins.
"""

from __future__ import annotations

import contextlib
import multiprocessing
import sys
import time
import tracemalloc
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

        results = self._execute_tests_batch(
            code_str,
            test_fn_name,
            test_inputs,
            expected_outputs,
        )

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
        recv_conn, send_conn = ctx.Pipe(duplex=False)
        process = ctx.Process(
            target=_execution_worker,
            kwargs={
                "code": code,
                "function_name": function_name,
                "test_input": test_input,
                "expected_output": expected_output,
                "max_memory_mb": self.max_memory_mb,
                "result_conn": send_conn,
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
            with contextlib.suppress(Exception):
                # Parent does not write to the child pipe endpoint.
                send_conn.close()
            process.join(self.timeout)
            if process.is_alive() and sys.platform == "win32":
                # Allow extra time for spawn startup on Windows CI.
                process.join(max(1.0, self.timeout))

            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=1.0)
                return ExecutionResult(
                    status=ExecutionStatus.TIMEOUT,
                    passed=False,
                    error=f"Execution timeout ({self.timeout}s)",
                    duration=time.perf_counter() - start,
                )

            if not recv_conn.poll(0.1):
                return ExecutionResult(
                    status=ExecutionStatus.ERROR,
                    passed=False,
                    error=f"Worker exited with code {process.exitcode}",
                    duration=time.perf_counter() - start,
                )
            payload = recv_conn.recv()
            return ExecutionResult(
                status=ExecutionStatus(
                    payload.get("status", ExecutionStatus.ERROR.value)
                ),
                passed=bool(payload.get("passed", False)),
                output=str(payload.get("output", "")),
                error=(
                    str(payload["error"]) if payload.get("error") is not None else None
                ),
                duration=time.perf_counter() - start,
            )
        except BaseException as exc:
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=1.0)
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                passed=False,
                error=str(exc),
                duration=time.perf_counter() - start,
            )
        finally:
            with contextlib.suppress(Exception):
                recv_conn.close()
            with contextlib.suppress(Exception):
                send_conn.close()

    def _execute_tests_batch(
        self,
        code: str,
        function_name: str,
        test_inputs: Sequence[Any],
        expected_outputs: Sequence[Any],
    ) -> list[ExecutionResult]:
        start = time.perf_counter()
        try:
            ctx = multiprocessing.get_context("fork")
        except ValueError:
            ctx = multiprocessing.get_context("spawn")
        recv_conn, send_conn = ctx.Pipe(duplex=False)
        process = ctx.Process(
            target=_execution_batch_worker,
            kwargs={
                "code": code,
                "function_name": function_name,
                "test_inputs": list(test_inputs),
                "expected_outputs": list(expected_outputs),
                "max_memory_mb": self.max_memory_mb,
                "result_conn": send_conn,
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
            with contextlib.suppress(Exception):
                send_conn.close()

            # Scale timeout by number of tests while preserving the single-test semantics.
            timeout = max(self.timeout, self.timeout * max(1, len(test_inputs)))
            process.join(timeout)
            if process.is_alive() and sys.platform == "win32":
                process.join(max(1.0, timeout))

            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=1.0)
                duration = time.perf_counter() - start
                return [
                    ExecutionResult(
                        status=ExecutionStatus.TIMEOUT,
                        passed=False,
                        error=f"Execution timeout ({timeout}s)",
                        duration=duration,
                    )
                    for _ in test_inputs
                ]

            if not recv_conn.poll(0.1):
                duration = time.perf_counter() - start
                return [
                    ExecutionResult(
                        status=ExecutionStatus.ERROR,
                        passed=False,
                        error=f"Worker exited with code {process.exitcode}",
                        duration=duration,
                    )
                    for _ in test_inputs
                ]

            payload = recv_conn.recv()
            if payload.get("status") != "ok":
                duration = time.perf_counter() - start
                return [
                    ExecutionResult(
                        status=ExecutionStatus.ERROR,
                        passed=False,
                        error=str(payload.get("error", "unknown execution error")),
                        duration=duration,
                    )
                    for _ in test_inputs
                ]

            results: list[ExecutionResult] = []
            for item in payload.get("results", []):
                results.append(
                    ExecutionResult(
                        status=ExecutionStatus(
                            item.get("status", ExecutionStatus.ERROR.value)
                        ),
                        passed=bool(item.get("passed", False)),
                        output=str(item.get("output", "")),
                        error=(
                            str(item["error"])
                            if item.get("error") is not None
                            else None
                        ),
                        duration=float(item.get("duration", 0.0)),
                    )
                )
            return results
        except BaseException as exc:
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=1.0)
            duration = time.perf_counter() - start
            return [
                ExecutionResult(
                    status=ExecutionStatus.ERROR,
                    passed=False,
                    error=str(exc),
                    duration=duration,
                )
                for _ in test_inputs
            ]
        finally:
            with contextlib.suppress(Exception):
                recv_conn.close()
            with contextlib.suppress(Exception):
                send_conn.close()


__all__ = ["ExecutionAccuracy", "ExecutionResult", "ExecutionStatus"]
