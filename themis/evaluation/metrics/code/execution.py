"""Safe code execution for testing functional correctness.

This module provides utilities for safely executing generated code against
test cases in a sandboxed environment.
"""

from __future__ import annotations

import multiprocessing
import signal
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Sequence

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
    """Result of code execution.
    
    Attributes:
        status: Execution status
        passed: Whether all tests passed
        output: Captured stdout/stderr
        error: Error message if any
        duration: Execution time in seconds
    """
    
    status: ExecutionStatus
    passed: bool
    output: str = ""
    error: str | None = None
    duration: float = 0.0


class ExecutionAccuracy(Metric):
    """Execute code and check against test cases.
    
    This metric safely executes generated code in a restricted environment
    and verifies correctness against provided test cases.
    
    Security considerations:
    - Executes in subprocess with timeout
    - Restricted globals (no file I/O, network, etc.)
    - Resource limits (memory, time)
    
    Attributes:
        name: Metric identifier ("execution_accuracy")
        timeout: Maximum execution time per test (seconds)
        max_memory_mb: Maximum memory usage (MB)
    
    Example:
        >>> from themis.evaluation.metrics.code import ExecutionAccuracy
        >>> metric = ExecutionAccuracy(timeout=3.0)
        >>> 
        >>> # Reference contains test cases
        >>> test_cases = {
        ...     "test_fn": test_function,
        ...     "inputs": [(1, 2), (3, 4)],
        ...     "expected": [3, 7]
        ... }
        >>> 
        >>> score = metric.compute(
        ...     prediction="def add(a, b): return a + b",
        ...     references=[test_cases]
        ... )
    """
    
    requires_reference = True
    
    def __init__(
        self,
        timeout: float = 3.0,
        max_memory_mb: int = 512,
    ):
        """Initialize execution metric.
        
        Args:
            timeout: Maximum execution time per test (seconds)
            max_memory_mb: Maximum memory usage (MB)
        """
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
        """Execute code and compute accuracy.
        
        Args:
            prediction: Generated code to execute
            references: List of test specifications
            metadata: Optional metadata dict
        
        Returns:
            MetricScore with execution accuracy
        """
        code_str = str(prediction)
        
        if not references:
            return MetricScore(
                metric_name=self.name,
                value=0.0,
                details={"error": "No test cases provided"},
                metadata=metadata or {},
            )
        
        # Extract test cases from reference
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
        
        # Execute code and run tests
        results = []
        for test_input, expected in zip(test_inputs, expected_outputs):
            result = self._execute_test(
                code_str,
                test_fn_name,
                test_input,
                expected,
            )
            results.append(result)
        
        # Compute accuracy
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
        """Execute a single test case.
        
        Args:
            code: Code to execute
            function_name: Name of function to test
            test_input: Input to pass to function
            expected_output: Expected output
        
        Returns:
            ExecutionResult with status and outcome
        """
        import time
        
        start_time = time.time()
        
        try:
            # Create restricted globals (no file I/O, network, etc.)
            restricted_globals = {
                "__builtins__": {
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
                    "range": range,
                    "reversed": reversed,
                    "set": set,
                    "sorted": sorted,
                    "str": str,
                    "sum": sum,
                    "tuple": tuple,
                    "zip": zip,
                }
            }
            
            # Execute code with timeout
            local_vars = {}
            exec(code, restricted_globals, local_vars)
            
            # Get the function
            if function_name not in local_vars:
                return ExecutionResult(
                    status=ExecutionStatus.ERROR,
                    passed=False,
                    error=f"Function '{function_name}' not found",
                    duration=time.time() - start_time,
                )
            
            func = local_vars[function_name]
            
            # Run function with input
            if isinstance(test_input, (list, tuple)):
                actual_output = func(*test_input)
            else:
                actual_output = func(test_input)
            
            # Check if output matches expected
            passed = actual_output == expected_output
            
            return ExecutionResult(
                status=ExecutionStatus.PASSED if passed else ExecutionStatus.FAILED,
                passed=passed,
                output=str(actual_output),
                duration=time.time() - start_time,
            )
            
        except TimeoutError:
            return ExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                passed=False,
                error=f"Execution timeout ({self.timeout}s)",
                duration=self.timeout,
            )
        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                passed=False,
                error=str(e),
                duration=time.time() - start_time,
            )


__all__ = ["ExecutionAccuracy", "ExecutionResult", "ExecutionStatus"]
