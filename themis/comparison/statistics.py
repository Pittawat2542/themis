"""Compatibility re-export for comparison statistical tests.

The canonical implementation now lives under `themis.evaluation.statistics`.
"""

from __future__ import annotations

from themis.evaluation.statistics.comparison_tests import (
    StatisticalTest,
    StatisticalTestResult,
    bootstrap_confidence_interval,
    mcnemar_test,
    permutation_test,
    t_test,
)

__all__ = [
    "StatisticalTest",
    "StatisticalTestResult",
    "t_test",
    "bootstrap_confidence_interval",
    "permutation_test",
    "mcnemar_test",
]
