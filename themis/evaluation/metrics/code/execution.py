"""Safe code execution for testing functional correctness.

This module delegates to the safe `sandbox` module to execute untrusted
generated code in an isolated subprocess with timeouts and restricted builtins.
"""

from __future__ import annotations

import contextlib
import multiprocessing
import time
from collections.abc import Sequence
from typing import Any

from themis.core.entities import MetricScore
from themis.interfaces import Metric
from themis.evaluation.metrics.code.sandbox import (
    ExecutionResult,
    ExecutionStatus,
    _execution_worker,
    _execution_batch_worker,
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
            ctx = multiprocessing.get_context("spawn")
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
            process.start()
            with contextlib.suppress(Exception):
                # Parent does not write to the child pipe endpoint.
                send_conn.close()
            process.join(self.timeout)
            if process.is_alive():
                # Allow extra time for spawn startup on all platforms.
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
            ctx = multiprocessing.get_context("spawn")
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
            process.start()
            with contextlib.suppress(Exception):
                send_conn.close()

            # Scale timeout by number of tests while preserving the single-test semantics.
            timeout = max(self.timeout, self.timeout * max(1, len(test_inputs)))
            process.join(timeout)
            if process.is_alive():
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
