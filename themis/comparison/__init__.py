"""Comparison engine for analyzing multiple experiment runs.

This module provides tools for comparing different models, prompts, or
configurations across multiple runs with statistical rigor.
"""

from themis.comparison.engine import ComparisonEngine, compare_runs
from themis.comparison.reports import ComparisonReport, ComparisonResult
from themis.comparison.statistics import (
    StatisticalTest,
    bootstrap_confidence_interval,
    permutation_test,
    t_test,
)

__all__ = [
    "ComparisonEngine",
    "compare_runs",
    "ComparisonReport",
    "ComparisonResult",
    "StatisticalTest",
    "bootstrap_confidence_interval",
    "permutation_test",
    "t_test",
]
