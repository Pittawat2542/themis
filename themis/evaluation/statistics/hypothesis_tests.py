"""Hypothesis testing functions."""

from __future__ import annotations

import math
import random
from statistics import mean, stdev
from typing import List, Literal, Sequence

from themis.core import entities as core_entities

from .confidence_intervals import compute_confidence_interval
from .distributions import t_to_p_value
from .types import ComparisonResult, PermutationTestResult


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
        t_stat = 0.0 if mean1 == mean2 else float("inf")
        p_value = 1.0 if mean1 == mean2 else 0.0
    else:
        pooled_se = math.sqrt((std1**2) / n1 + (std2**2) / n2)
        if pooled_se == 0.0:
            t_stat = 0.0
            p_value = 1.0
        else:
            t_stat = (mean2 - mean1) / pooled_se
            # Degrees of freedom (Welch-Satterthwaite approximation)
            if std1 > 0 and std2 > 0:
                df = ((std1**2 / n1 + std2**2 / n2) ** 2) / (
                    (std1**2 / n1) ** 2 / (n1 - 1) + (std2**2 / n2) ** 2 / (n2 - 1)
                )
            else:
                df = max(n1, n2) - 1
            # Approximate p-value using t-distribution
            p_value = t_to_p_value(abs(t_stat), int(df))

    difference = mean2 - mean1
    relative_change = (difference / mean1 * 100.0) if mean1 != 0 else float("inf")

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


def permutation_test(
    group_a: Sequence[float],
    group_b: Sequence[float],
    statistic: Literal["mean_diff", "median_diff"] = "mean_diff",
    n_permutations: int = 10000,
    seed: int | None = None,
) -> PermutationTestResult:
    """Perform permutation test to compare two groups.

    This non-parametric test does not assume normality and is robust
    to outliers and skewed distributions.

    Args:
        group_a: Values from first group
        group_b: Values from second group
        statistic: Test statistic to use ("mean_diff" or "median_diff")
        n_permutations: Number of permutation iterations (default: 10000)
        seed: Random seed for reproducibility

    Returns:
        PermutationTestResult with p-value and statistics

    Raises:
        ValueError: If either group is empty
    """
    if not group_a or not group_b:
        raise ValueError("Both groups must be non-empty")

    # Set random seed if provided
    if seed is not None:
        random.seed(seed)

    # Compute observed statistic
    def compute_stat(a: Sequence[float], b: Sequence[float]) -> float:
        if statistic == "mean_diff":
            return mean(b) - mean(a)
        elif statistic == "median_diff":
            import statistics

            return statistics.median(b) - statistics.median(a)
        else:
            raise ValueError(f"Unknown statistic: {statistic}")

    observed = compute_stat(group_a, group_b)

    # Combine all values for permutation
    combined = list(group_a) + list(group_b)
    n_a = len(group_a)

    # Permutation iterations
    count_extreme = 0
    for _ in range(n_permutations):
        # Shuffle and split into two groups
        random.shuffle(combined)
        perm_a = combined[:n_a]
        perm_b = combined[n_a:]

        # Compute permuted statistic
        perm_stat = compute_stat(perm_a, perm_b)

        # Two-tailed test: count if |perm_stat| >= |observed|
        if abs(perm_stat) >= abs(observed):
            count_extreme += 1

    p_value = count_extreme / n_permutations

    return PermutationTestResult(
        observed_statistic=observed,
        p_value=p_value,
        n_permutations=n_permutations,
        is_significant=p_value < 0.05,
    )


def holm_bonferroni(p_values: Sequence[float]) -> List[bool]:
    """Apply Holm-Bonferroni correction for multiple comparisons.

    This method controls the family-wise error rate (FWER) while being
    more powerful than the simple Bonferroni correction.

    Args:
        p_values: List of p-values from multiple tests

    Returns:
        List of boolean values indicating which tests remain significant
        after correction (True = significant, False = not significant)

    Example:
        >>> p_vals = [0.01, 0.04, 0.03, 0.20]
        >>> significant = holm_bonferroni(p_vals)
        >>> # Returns which tests are significant after correction
    """
    if not p_values:
        return []

    n = len(p_values)

    # Create (p-value, original_index) pairs and sort by p-value
    indexed_pvals = [(p, i) for i, p in enumerate(p_values)]
    indexed_pvals.sort(key=lambda x: x[0])

    # Apply Holm-Bonferroni sequential rejection
    results = [False] * n
    alpha = 0.05  # Standard significance level

    for rank, (p_val, orig_idx) in enumerate(indexed_pvals):
        # Adjusted threshold: alpha / (n - rank)
        threshold = alpha / (n - rank)

        if p_val < threshold:
            results[orig_idx] = True
        else:
            # Once we fail to reject, all subsequent tests also fail
            break

    return results
