"""Statistical analysis utilities for experiment evaluation results.

This module provides statistical analysis tools for computing confidence intervals,
significance tests, and statistical comparisons across experiment runs.
"""

from __future__ import annotations

from .bootstrap import bootstrap_ci
from .confidence_intervals import (
    compute_confidence_interval,
    compute_statistical_summary,
)
from .effect_sizes import cohens_d, cohens_h
from .hypothesis_tests import (
    compare_metrics,
    holm_bonferroni,
    paired_permutation_test,
    paired_t_test,
    permutation_test,
)
from .types import (
    BootstrapResult,
    ComparisonResult,
    ConfidenceInterval,
    EffectSize,
    PermutationTestResult,
    StatisticalSummary,
)

__all__ = [
    # Types
    "ConfidenceInterval",
    "StatisticalSummary",
    "ComparisonResult",
    "PermutationTestResult",
    "BootstrapResult",
    "EffectSize",
    # Confidence intervals
    "compute_confidence_interval",
    "compute_statistical_summary",
    # Hypothesis tests
    "compare_metrics",
    "permutation_test",
    "paired_permutation_test",
    "paired_t_test",
    "holm_bonferroni",
    # Bootstrap
    "bootstrap_ci",
    # Effect sizes
    "cohens_h",
    "cohens_d",
]
