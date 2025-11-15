"""Statistical analysis utilities for experiment evaluation results.

This module provides statistical analysis tools for computing confidence intervals,
significance tests, and statistical comparisons across experiment runs.
"""

from __future__ import annotations

from .types import (
    ConfidenceInterval,
    StatisticalSummary,
    ComparisonResult,
    PermutationTestResult,
    BootstrapResult,
    EffectSize,
)
from .confidence_intervals import (
    compute_confidence_interval,
    compute_statistical_summary,
)
from .hypothesis_tests import (
    compare_metrics,
    permutation_test,
    holm_bonferroni,
)
from .bootstrap import bootstrap_ci
from .effect_sizes import cohens_h, cohens_d

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
    "holm_bonferroni",
    # Bootstrap
    "bootstrap_ci",
    # Effect sizes
    "cohens_h",
    "cohens_d",
]
