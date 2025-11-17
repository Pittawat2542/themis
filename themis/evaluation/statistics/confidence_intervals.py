"""Confidence interval computation."""

from __future__ import annotations

import math
from statistics import mean, stdev
from typing import List, Sequence

from themis.core import entities as core_entities

from .distributions import inverse_normal_cdf, t_critical_value
from .types import ConfidenceInterval, StatisticalSummary


def compute_confidence_interval(
    values: Sequence[float],
    confidence_level: float = 0.95,
) -> ConfidenceInterval:
    """Compute confidence interval for a sample mean using t-distribution.

    Args:
        values: Sequence of numeric values
        confidence_level: Confidence level (default: 0.95)

    Returns:
        ConfidenceInterval with bounds and statistics

    Raises:
        ValueError: If values is empty or has insufficient data
    """
    n = len(values)
    if n == 0:
        raise ValueError("Cannot compute confidence interval for empty sequence")
    if n == 1:
        # Single value - return degenerate interval
        val = float(values[0])
        return ConfidenceInterval(
            mean=val,
            lower=val,
            upper=val,
            confidence_level=confidence_level,
            sample_size=1,
        )

    sample_mean = mean(values)
    sample_std = stdev(values)

    # For large samples (n >= 30), use normal approximation with z-score
    # For small samples, use t-distribution critical value
    if n >= 30:
        # Normal approximation: use z-scores
        # For 95% CI: z = 1.96, for 99% CI: z = 2.576
        if abs(confidence_level - 0.95) < 0.01:
            critical_value = 1.96
        elif abs(confidence_level - 0.99) < 0.01:
            critical_value = 2.576
        elif abs(confidence_level - 0.90) < 0.01:
            critical_value = 1.645
        else:
            # General approximation using inverse normal CDF
            critical_value = inverse_normal_cdf((1 + confidence_level) / 2)
    else:
        # Small sample: use t-distribution critical value (approximation)
        critical_value = t_critical_value(n - 1, confidence_level)

    standard_error = sample_std / math.sqrt(n)
    margin_of_error = critical_value * standard_error

    return ConfidenceInterval(
        mean=sample_mean,
        lower=sample_mean - margin_of_error,
        upper=sample_mean + margin_of_error,
        confidence_level=confidence_level,
        sample_size=n,
    )


def compute_statistical_summary(
    scores: List[core_entities.MetricScore],
) -> StatisticalSummary:
    """Compute comprehensive statistical summary for metric scores.

    Args:
        scores: List of MetricScore objects

    Returns:
        StatisticalSummary with descriptive statistics

    Raises:
        ValueError: If scores is empty
    """
    if not scores:
        raise ValueError("Cannot compute statistical summary for empty scores list")

    metric_name = scores[0].metric_name
    values = [score.value for score in scores]
    n = len(values)

    # Sort for percentile calculations
    sorted_values = sorted(values)
    median_idx = n // 2
    if n % 2 == 0:
        median_value = (sorted_values[median_idx - 1] + sorted_values[median_idx]) / 2.0
    else:
        median_value = sorted_values[median_idx]

    # Compute confidence interval if we have enough data
    ci_95 = None
    if n >= 2:
        ci_95 = compute_confidence_interval(values, confidence_level=0.95)

    return StatisticalSummary(
        metric_name=metric_name,
        count=n,
        mean=mean(values),
        std=stdev(values) if n >= 2 else 0.0,
        min_value=min(values),
        max_value=max(values),
        median=median_value,
        confidence_interval_95=ci_95,
    )
