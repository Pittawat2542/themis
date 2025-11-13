"""Statistical analysis utilities for experiment evaluation results.

This module provides statistical analysis tools for computing confidence intervals,
significance tests, and statistical comparisons across experiment runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean, stdev
from typing import List, Sequence

from themis.core import entities as core_entities


@dataclass
class ConfidenceInterval:
    """Confidence interval for a metric.
    
    Attributes:
        mean: Sample mean of the metric
        lower: Lower bound of the confidence interval
        upper: Upper bound of the confidence interval
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        sample_size: Number of samples used
    """
    mean: float
    lower: float
    upper: float
    confidence_level: float
    sample_size: int

    @property
    def margin_of_error(self) -> float:
        """Return the margin of error (half-width of the interval)."""
        return (self.upper - self.lower) / 2.0

    @property
    def width(self) -> float:
        """Return the width of the confidence interval."""
        return self.upper - self.lower


@dataclass
class StatisticalSummary:
    """Statistical summary for a set of metric scores.
    
    Attributes:
        metric_name: Name of the metric
        count: Number of samples
        mean: Sample mean
        std: Sample standard deviation
        min_value: Minimum value
        max_value: Maximum value
        median: Median value
        confidence_interval_95: 95% confidence interval for the mean
    """
    metric_name: str
    count: int
    mean: float
    std: float
    min_value: float
    max_value: float
    median: float
    confidence_interval_95: ConfidenceInterval | None


@dataclass
class ComparisonResult:
    """Result of a statistical comparison between two metric sets.
    
    Attributes:
        metric_name: Name of the metric being compared
        baseline_mean: Mean of the baseline (control) group
        treatment_mean: Mean of the treatment group
        difference: Difference between treatment and baseline means
        relative_change: Relative change as a percentage
        t_statistic: t-test statistic
        p_value: p-value for the two-sample t-test
        is_significant: Whether the difference is statistically significant (p < 0.05)
        baseline_ci: 95% confidence interval for baseline mean
        treatment_ci: 95% confidence interval for treatment mean
    """
    metric_name: str
    baseline_mean: float
    treatment_mean: float
    difference: float
    relative_change: float
    t_statistic: float
    p_value: float
    is_significant: bool
    baseline_ci: ConfidenceInterval
    treatment_ci: ConfidenceInterval


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
            critical_value = _inverse_normal_cdf((1 + confidence_level) / 2)
    else:
        # Small sample: use t-distribution critical value (approximation)
        critical_value = _t_critical_value(n - 1, confidence_level)
    
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


def compare_metrics(
    baseline_scores: List[core_entities.MetricScore],
    treatment_scores: List[core_entities.MetricScore],
    significance_level: float = 0.05,
) -> ComparisonResult:
    """Perform two-sample t-test to compare baseline vs treatment metrics.
    
    Args:
        baseline_scores: Metric scores from baseline/control group
        treatment_scores: Metric scores from treatment group
        significance_level: Threshold for statistical significance (default: 0.05)
    
    Returns:
        ComparisonResult with comparison statistics
    
    Raises:
        ValueError: If either scores list is empty or metric names don't match
    """
    if not baseline_scores or not treatment_scores:
        raise ValueError("Both baseline and treatment scores must be non-empty")
    
    baseline_name = baseline_scores[0].metric_name
    treatment_name = treatment_scores[0].metric_name
    if baseline_name != treatment_name:
        raise ValueError(
            f"Metric names must match: baseline='{baseline_name}', "
            f"treatment='{treatment_name}'"
        )
    
    baseline_values = [score.value for score in baseline_scores]
    treatment_values = [score.value for score in treatment_scores]
    
    n1 = len(baseline_values)
    n2 = len(treatment_values)
    mean1 = mean(baseline_values)
    mean2 = mean(treatment_values)
    
    # Compute standard deviations
    std1 = stdev(baseline_values) if n1 >= 2 else 0.0
    std2 = stdev(treatment_values) if n2 >= 2 else 0.0
    
    # Two-sample t-test (Welch's t-test for unequal variances)
    if std1 == 0.0 and std2 == 0.0:
        # Both groups have no variance
        t_stat = 0.0 if mean1 == mean2 else float('inf')
        p_value = 1.0 if mean1 == mean2 else 0.0
    else:
        pooled_se = math.sqrt((std1 ** 2) / n1 + (std2 ** 2) / n2)
        if pooled_se == 0.0:
            t_stat = 0.0
            p_value = 1.0
        else:
            t_stat = (mean2 - mean1) / pooled_se
            # Degrees of freedom (Welch-Satterthwaite approximation)
            if std1 > 0 and std2 > 0:
                df = ((std1 ** 2 / n1 + std2 ** 2 / n2) ** 2) / (
                    (std1 ** 2 / n1) ** 2 / (n1 - 1) + (std2 ** 2 / n2) ** 2 / (n2 - 1)
                )
            else:
                df = max(n1, n2) - 1
            # Approximate p-value using t-distribution
            p_value = _t_to_p_value(abs(t_stat), df)
    
    difference = mean2 - mean1
    relative_change = (difference / mean1 * 100.0) if mean1 != 0 else float('inf')
    
    # Compute confidence intervals
    baseline_ci = compute_confidence_interval(baseline_values)
    treatment_ci = compute_confidence_interval(treatment_values)
    
    return ComparisonResult(
        metric_name=baseline_name,
        baseline_mean=mean1,
        treatment_mean=mean2,
        difference=difference,
        relative_change=relative_change,
        t_statistic=t_stat,
        p_value=p_value,
        is_significant=p_value < significance_level,
        baseline_ci=baseline_ci,
        treatment_ci=treatment_ci,
    )


# Helper functions for statistical distributions


def _inverse_normal_cdf(p: float) -> float:
    """Approximate inverse normal CDF (probit function) for standard normal.
    
    Uses Beasley-Springer-Moro approximation.
    """
    if p <= 0 or p >= 1:
        raise ValueError("Probability must be between 0 and 1")
    
    # Constants for approximation
    a = [2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637]
    b = [-8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833]
    c = [0.3374754822726147, 0.9761690190917186, 0.1607979714918209,
         0.0276438810333863, 0.0038405729373609, 0.0003951896511919,
         0.0000321767881768, 0.0000002888167364, 0.0000003960315187]
    
    # Transform to standard normal
    y = p - 0.5
    if abs(y) < 0.42:
        # Central region
        r = y * y
        x = y * (((a[3] * r + a[2]) * r + a[1]) * r + a[0]) / (
            ((b[3] * r + b[2]) * r + b[1]) * r + b[0] + 1.0
        )
        return x
    else:
        # Tail region
        r = p if y > 0 else 1 - p
        r = math.log(-math.log(r))
        x = c[0] + r * (c[1] + r * (c[2] + r * (c[3] + r * (c[4] + r * (
            c[5] + r * (c[6] + r * (c[7] + r * c[8])))))))
        if y < 0:
            x = -x
        return x


def _t_critical_value(df: int, confidence_level: float) -> float:
    """Approximate t-distribution critical value.
    
    This is a simplified approximation. For production use, consider scipy.stats.t.ppf.
    """
    # For common confidence levels and degrees of freedom, use lookup table
    # Otherwise, use normal approximation for large df
    if df >= 30:
        # Use normal approximation for large df
        alpha = (1 - confidence_level) / 2
        return _inverse_normal_cdf(1 - alpha)
    
    # Simplified lookup table for small df (two-tailed)
    # Format: {confidence_level: {df: critical_value}}
    lookup_95 = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        15: 2.131, 20: 2.086, 25: 2.060, 29: 2.045
    }
    lookup_99 = {
        1: 63.657, 2: 9.925, 3: 5.841, 4: 4.604, 5: 4.032,
        6: 3.707, 7: 3.499, 8: 3.355, 9: 3.250, 10: 3.169,
        15: 2.947, 20: 2.845, 25: 2.787, 29: 2.756
    }
    
    if abs(confidence_level - 0.95) < 0.01:
        lookup = lookup_95
    elif abs(confidence_level - 0.99) < 0.01:
        lookup = lookup_99
    else:
        # Fall back to normal approximation
        alpha = (1 - confidence_level) / 2
        return _inverse_normal_cdf(1 - alpha)
    
    # Find closest df in lookup table
    if df in lookup:
        return lookup[df]
    else:
        # Linear interpolation or nearest neighbor
        df_keys = sorted(lookup.keys())
        for i, key_df in enumerate(df_keys):
            if df < key_df:
                if i == 0:
                    return lookup[key_df]
                else:
                    # Interpolate between previous and current
                    prev_df = df_keys[i - 1]
                    weight = (df - prev_df) / (key_df - prev_df)
                    return lookup[prev_df] * (1 - weight) + lookup[key_df] * weight
        return lookup[df_keys[-1]]


def _t_to_p_value(t_stat: float, df: int) -> float:
    """Approximate two-tailed p-value for t-statistic.
    
    This is a simplified approximation. For production use, consider scipy.stats.t.cdf.
    """
    # For large df, use normal approximation
    if df >= 30:
        # Use normal distribution CDF
        p_one_tail = _normal_cdf(-abs(t_stat))
        return 2 * p_one_tail
    
    # For small df, use approximation
    # Very rough approximation: convert t to approximate p-value
    if abs(t_stat) < 0.5:
        return 1.0
    elif abs(t_stat) > 10:
        return 0.0001
    else:
        # Rough approximation using exponential decay
        base_p = math.exp(-abs(t_stat) * 0.5) * (df / (df + t_stat ** 2))
        return min(1.0, 2 * base_p)


def _normal_cdf(x: float) -> float:
    """Standard normal CDF using error function approximation."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


__all__ = [
    "ConfidenceInterval",
    "StatisticalSummary",
    "ComparisonResult",
    "compute_confidence_interval",
    "compute_statistical_summary",
    "compare_metrics",
]
