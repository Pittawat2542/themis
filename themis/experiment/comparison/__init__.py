"""Comparison engine and tools for analyzing multiple runs.

This module provides both lightweight tools for comparing aggregated outputs
across multiple experiment runs, as well as rigorous, sample-level statistical
significance testing (e.g., Bootstrap, T-Test).
"""

from themis.experiment.comparison.engine import ComparisonEngine, compare_runs
from themis.experiment.comparison.reports import ComparisonReport, ComparisonResult
from themis.experiment.comparison.entities import ComparisonRow, ConfigDiff
from themis.experiment.comparison.statistics import (
    MultiExperimentComparison,
    compare_experiments,
    diff_configs,
)
from themis.evaluation.statistics.comparison_tests import (
    StatisticalTest,
    bootstrap_confidence_interval,
    permutation_test,
    t_test,
)

__all__ = [
    # Engine & Reports
    "ComparisonEngine",
    "compare_runs",
    "ComparisonReport",
    "ComparisonResult",
    # Aggregation & Diff
    "ComparisonRow",
    "ConfigDiff",
    "MultiExperimentComparison",
    "compare_experiments",
    "diff_configs",
    # Statistical Tests
    "StatisticalTest",
    "bootstrap_confidence_interval",
    "permutation_test",
    "t_test",
]
