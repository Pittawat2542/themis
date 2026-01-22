"""Pass@k metric for code generation evaluation.

Pass@k measures functional correctness by executing k generated code samples
and checking if any of them pass the test cases.

References:
    Chen et al. (2021). Evaluating Large Language Models Trained on Code.
    (HumanEval paper)
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from themis.core.entities import MetricScore
from themis.interfaces import Metric


def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    """Estimate pass@k using unbiased estimator.
    
    This is the standard estimator from the HumanEval paper.
    
    Args:
        n: Total number of samples generated
        c: Number of samples that passed
        k: k value for pass@k
    
    Returns:
        Estimated pass@k probability
    
    Example:
        >>> # Generated 10 samples, 3 passed, compute pass@1
        >>> estimate_pass_at_k(n=10, c=3, k=1)
        0.3
        
        >>> # Generated 100 samples, 30 passed, compute pass@10
        >>> estimate_pass_at_k(n=100, c=30, k=10)
        0.8926
    """
    if n - c < k:
        return 1.0
    
    # Unbiased estimator: 1 - C(n-c, k) / C(n, k)
    # = 1 - product((n-c-i)/(n-i) for i in range(k))
    result = 1.0
    for i in range(k):
        result *= (n - c - i) / (n - i)
    
    return 1.0 - result


class PassAtK(Metric):
    """Pass@k metric for code generation.
    
    Pass@k measures the probability that at least one of k generated samples
    passes all test cases. It's the standard metric for evaluating code
    generation models like Codex, CodeGen, etc.
    
    The metric requires:
    - Multiple samples per problem (num_samples >= k)
    - Test cases to execute against
    - Safe code execution environment
    
    Attributes:
        name: Metric identifier ("pass_at_k")
        k: Number of samples to consider
        timeout: Maximum execution time per sample (seconds)
        require_all_tests: Whether all tests must pass (vs any test)
    
    Example:
        >>> from themis.evaluation.metrics.code import PassAtK
        >>> metric = PassAtK(k=1)
        >>> score = metric.compute(
        ...     prediction={
        ...         "samples": ["def add(a, b): return a + b", ...],
        ...         "test_results": [True, False, ...],
        ...     },
        ...     references=[]
        ... )
        >>> print(f"Pass@1: {score.value:.2%}")
        Pass@1: 30.00%
    """
    
    requires_reference = False  # Uses test execution, not reference matching
    
    def __init__(
        self,
        k: int = 1,
        timeout: float = 3.0,
        require_all_tests: bool = True,
    ):
        """Initialize Pass@k metric.
        
        Args:
            k: Number of samples for pass@k estimation
            timeout: Maximum execution time per sample (seconds)
            require_all_tests: Whether all test cases must pass (default: True)
        """
        self.name = f"pass_at_{k}"
        self.k = k
        self.timeout = timeout
        self.require_all_tests = require_all_tests
    
    def compute(
        self,
        *,
        prediction: Any,
        references: Sequence[Any],
        metadata: dict[str, Any] | None = None,
    ) -> MetricScore:
        """Compute Pass@k score.
        
        Args:
            prediction: Dictionary containing:
                - "samples": List of generated code samples
                - "test_results": List of booleans (True if passed)
                - "execution_errors": Optional list of error messages
            references: Not used (test-based evaluation)
            metadata: Optional metadata dict
        
        Returns:
            MetricScore with estimated pass@k probability
        
        Note:
            The prediction should be prepared by ExecutionAccuracy metric
            or similar execution framework.
        """
        if not isinstance(prediction, dict):
            return MetricScore(
                metric_name=self.name,
                value=0.0,
                details={"error": "Prediction must be dict with samples and test_results"},
                metadata=metadata or {},
            )
        
        samples = prediction.get("samples", [])
        test_results = prediction.get("test_results", [])
        
        if not samples or not test_results:
            return MetricScore(
                metric_name=self.name,
                value=0.0,
                details={
                    "error": "Missing samples or test_results",
                    "num_samples": len(samples),
                    "num_results": len(test_results),
                },
                metadata=metadata or {},
            )
        
        # Count number of samples and passes
        n = len(test_results)
        c = sum(1 for result in test_results if result)
        
        # Estimate pass@k
        if n < self.k:
            # Not enough samples, use empirical rate
            pass_at_k = c / n if n > 0 else 0.0
            warning = f"Only {n} samples available for pass@{self.k}"
        else:
            pass_at_k = estimate_pass_at_k(n, c, self.k)
            warning = None
        
        return MetricScore(
            metric_name=self.name,
            value=pass_at_k,
            details={
                "k": self.k,
                "n_samples": n,
                "n_passed": c,
                "pass_rate": c / n if n > 0 else 0.0,
                "pass_at_k": pass_at_k,
                "warning": warning,
            },
            metadata=metadata or {},
        )


__all__ = ["PassAtK", "estimate_pass_at_k"]
