"""Statistical tests for comparing experiment results.

This module provides various statistical tests to determine if differences
between runs are statistically significant.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class StatisticalTest(str, Enum):
    """Available statistical tests."""
    
    T_TEST = "t_test"
    BOOTSTRAP = "bootstrap"
    PERMUTATION = "permutation"
    NONE = "none"


@dataclass
class StatisticalTestResult:
    """Result of a statistical test.
    
    Attributes:
        test_name: Name of the test performed
        statistic: Test statistic value
        p_value: P-value (probability of observing this difference by chance)
        significant: Whether the difference is statistically significant
        confidence_level: Confidence level used (e.g., 0.95 for 95%)
        effect_size: Effect size (e.g., Cohen's d)
        confidence_interval: Confidence interval for the difference
    """
    
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    confidence_level: float = 0.95
    effect_size: float | None = None
    confidence_interval: tuple[float, float] | None = None
    
    def __str__(self) -> str:
        """Human-readable summary."""
        sig_str = "significant" if self.significant else "not significant"
        result = f"{self.test_name}: p={self.p_value:.4f} ({sig_str})"
        
        if self.effect_size is not None:
            result += f", effect_size={self.effect_size:.3f}"
        
        if self.confidence_interval is not None:
            low, high = self.confidence_interval
            result += f", CI=[{low:.3f}, {high:.3f}]"
        
        return result


def t_test(
    samples_a: Sequence[float],
    samples_b: Sequence[float],
    *,
    alpha: float = 0.05,
    paired: bool = True,
) -> StatisticalTestResult:
    """Perform a t-test to compare two sets of samples.
    
    Args:
        samples_a: First set of samples
        samples_b: Second set of samples
        alpha: Significance level (default: 0.05 for 95% confidence)
        paired: Whether to use paired t-test (default: True)
    
    Returns:
        StatisticalTestResult with test statistics and significance
    
    Raises:
        ValueError: If samples are empty or have mismatched lengths (for paired test)
    """
    if not samples_a or not samples_b:
        raise ValueError("Cannot perform t-test on empty samples")
    
    if paired and len(samples_a) != len(samples_b):
        raise ValueError(
            f"Paired t-test requires equal sample sizes. "
            f"Got {len(samples_a)} and {len(samples_b)}"
        )
    
    n_a = len(samples_a)
    n_b = len(samples_b)
    
    # Calculate means
    mean_a = sum(samples_a) / n_a
    mean_b = sum(samples_b) / n_b
    
    if paired:
        # Paired t-test: test on differences
        diffs = [a - b for a, b in zip(samples_a, samples_b)]
        mean_diff = sum(diffs) / len(diffs)
        
        # Standard deviation of differences
        var_diff = sum((d - mean_diff) ** 2 for d in diffs) / (len(diffs) - 1) if len(diffs) > 1 else 0
        se_diff = math.sqrt(var_diff / len(diffs))
        
        # T-statistic
        if se_diff > 1e-10:  # Non-zero standard error
            t_stat = mean_diff / se_diff
        elif abs(mean_diff) > 1e-10:  # Perfect consistency with non-zero difference
            t_stat = float('inf') if mean_diff > 0 else float('-inf')
        else:  # No difference at all
            t_stat = 0.0
        
        df = len(diffs) - 1
        
        # Effect size (Cohen's d for paired samples)
        sd_diff = math.sqrt(var_diff)
        effect_size = mean_diff / sd_diff if sd_diff > 1e-10 else (1.0 if abs(mean_diff) > 1e-10 else 0.0)
        
    else:
        # Independent samples t-test
        # Calculate pooled standard deviation
        var_a = sum((x - mean_a) ** 2 for x in samples_a) / (n_a - 1) if n_a > 1 else 0
        var_b = sum((x - mean_b) ** 2 for x in samples_b) / (n_b - 1) if n_b > 1 else 0
        
        pooled_sd = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
        se = pooled_sd * math.sqrt(1/n_a + 1/n_b)
        
        # T-statistic
        t_stat = (mean_a - mean_b) / se if se > 0 else 0.0
        df = n_a + n_b - 2
        
        # Effect size (Cohen's d)
        effect_size = (mean_a - mean_b) / pooled_sd if pooled_sd > 0 else 0.0
    
    # Approximate p-value using t-distribution
    # For simplicity, we use a conservative approximation
    # In practice, you'd use scipy.stats.t.sf for accurate p-values
    p_value = _approximate_t_test_p_value(abs(t_stat), df)
    
    # Confidence interval (approximate)
    # t_critical ≈ 2.0 for 95% CI and reasonable df
    t_critical = 2.0  # Conservative estimate
    margin = t_critical * (se_diff if paired else se)
    ci = (mean_a - mean_b - margin, mean_a - mean_b + margin)
    
    return StatisticalTestResult(
        test_name="t-test (paired)" if paired else "t-test (independent)",
        statistic=t_stat,
        p_value=p_value,
        significant=p_value < alpha,
        confidence_level=1 - alpha,
        effect_size=effect_size,
        confidence_interval=ci,
    )


def _approximate_t_test_p_value(t_stat: float, df: int) -> float:
    """Approximate p-value for t-test.
    
    This is a rough approximation. For accurate p-values, use scipy.stats.
    """
    # Very rough approximation based on standard normal
    # This gets less accurate for small df
    if df < 1:
        return 1.0
    
    # Convert to z-score approximation for large df
    if df > 30:
        z = t_stat
        # Approximate p-value for two-tailed test
        # P(|Z| > z) ≈ 2 * (1 - Φ(z))
        if z > 6:
            return 0.0
        elif z < 0.5:
            return 1.0
        else:
            # Rough approximation
            return min(1.0, 2 * math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi))
    
    # For small df, be conservative
    return min(1.0, 0.5 if t_stat < 2 else 0.1 if t_stat < 3 else 0.01)


def bootstrap_confidence_interval(
    samples_a: Sequence[float],
    samples_b: Sequence[float],
    *,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    statistic_fn: callable = None,
    seed: int | None = None,
) -> StatisticalTestResult:
    """Compute bootstrap confidence interval for difference between two samples.
    
    Uses bootstrap resampling to estimate the confidence interval for the
    difference in means (or other statistic) between two samples.
    
    Args:
        samples_a: First set of samples
        samples_b: Second set of samples
        n_bootstrap: Number of bootstrap iterations (default: 10000)
        confidence_level: Confidence level (default: 0.95)
        statistic_fn: Function to compute statistic (default: mean difference)
        seed: Random seed for reproducibility
    
    Returns:
        StatisticalTestResult with bootstrap confidence interval
    """
    if not samples_a or not samples_b:
        raise ValueError("Cannot perform bootstrap on empty samples")
    
    if seed is not None:
        random.seed(seed)
    
    # Default statistic: difference in means
    if statistic_fn is None:
        def statistic_fn(a, b):
            return sum(a) / len(a) - sum(b) / len(b)
    
    # Observed difference
    observed_diff = statistic_fn(samples_a, samples_b)
    
    # Bootstrap resampling
    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        resampled_a = [random.choice(samples_a) for _ in range(len(samples_a))]
        resampled_b = [random.choice(samples_b) for _ in range(len(samples_b))]
        
        diff = statistic_fn(resampled_a, resampled_b)
        bootstrap_diffs.append(diff)
    
    # Sort for percentile method
    bootstrap_diffs.sort()
    
    # Compute confidence interval
    alpha = 1 - confidence_level
    lower_idx = int(n_bootstrap * (alpha / 2))
    upper_idx = int(n_bootstrap * (1 - alpha / 2))
    
    ci = (bootstrap_diffs[lower_idx], bootstrap_diffs[upper_idx])
    
    # Check if 0 is in the confidence interval
    significant = not (ci[0] <= 0 <= ci[1])
    
    # Pseudo p-value: proportion of bootstrap samples with opposite sign
    p_value = sum(1 for d in bootstrap_diffs if (d * observed_diff) < 0) / n_bootstrap
    p_value = max(p_value, 1 / n_bootstrap)  # Lower bound
    
    return StatisticalTestResult(
        test_name=f"bootstrap (n={n_bootstrap})",
        statistic=observed_diff,
        p_value=p_value,
        significant=significant,
        confidence_level=confidence_level,
        confidence_interval=ci,
    )


def permutation_test(
    samples_a: Sequence[float],
    samples_b: Sequence[float],
    *,
    n_permutations: int = 10000,
    alpha: float = 0.05,
    statistic_fn: callable = None,
    seed: int | None = None,
) -> StatisticalTestResult:
    """Perform permutation test to compare two samples.
    
    Tests the null hypothesis that the two samples come from the same
    distribution by randomly permuting the labels and computing the test
    statistic.
    
    Args:
        samples_a: First set of samples
        samples_b: Second set of samples
        n_permutations: Number of permutations (default: 10000)
        alpha: Significance level (default: 0.05)
        statistic_fn: Function to compute statistic (default: difference in means)
        seed: Random seed for reproducibility
    
    Returns:
        StatisticalTestResult with permutation test results
    """
    if not samples_a or not samples_b:
        raise ValueError("Cannot perform permutation test on empty samples")
    
    if seed is not None:
        random.seed(seed)
    
    # Default statistic: absolute difference in means
    if statistic_fn is None:
        def statistic_fn(a, b):
            return abs(sum(a) / len(a) - sum(b) / len(b))
    
    # Observed statistic
    observed_stat = statistic_fn(samples_a, samples_b)
    
    # Combine all samples
    combined = list(samples_a) + list(samples_b)
    n_a = len(samples_a)
    n_total = len(combined)
    
    # Permutation testing
    more_extreme = 0
    for _ in range(n_permutations):
        # Shuffle and split
        shuffled = combined.copy()
        random.shuffle(shuffled)
        
        perm_a = shuffled[:n_a]
        perm_b = shuffled[n_a:]
        
        perm_stat = statistic_fn(perm_a, perm_b)
        
        if perm_stat >= observed_stat:
            more_extreme += 1
    
    # P-value: proportion of permutations as extreme as observed
    p_value = more_extreme / n_permutations
    
    return StatisticalTestResult(
        test_name=f"permutation (n={n_permutations})",
        statistic=observed_stat,
        p_value=p_value,
        significant=p_value < alpha,
        confidence_level=1 - alpha,
    )


def mcnemar_test(
    contingency_table: tuple[int, int, int, int],
    *,
    alpha: float = 0.05,
) -> StatisticalTestResult:
    """Perform McNemar's test for paired nominal data.
    
    Useful for comparing two models on the same test set, where you want to
    know if one model consistently outperforms the other.
    
    Args:
        contingency_table: 2x2 contingency table as (n_00, n_01, n_10, n_11)
            where n_ij = number of samples where model A predicts i and model B predicts j
            (0 = incorrect, 1 = correct)
        alpha: Significance level
    
    Returns:
        StatisticalTestResult with McNemar's test results
    """
    n_00, n_01, n_10, n_11 = contingency_table
    
    # Only discordant pairs matter
    b = n_01  # A wrong, B correct
    c = n_10  # A correct, B wrong
    
    if b + c == 0:
        # No discordant pairs
        return StatisticalTestResult(
            test_name="McNemar's test",
            statistic=0.0,
            p_value=1.0,
            significant=False,
            confidence_level=1 - alpha,
        )
    
    # McNemar's statistic with continuity correction
    chi_square = ((abs(b - c) - 1) ** 2) / (b + c)
    
    # Approximate p-value (chi-square with 1 df)
    # For chi-square > 3.84, p < 0.05
    # For chi-square > 6.63, p < 0.01
    if chi_square > 10.83:
        p_value = 0.001
    elif chi_square > 6.63:
        p_value = 0.01
    elif chi_square > 3.84:
        p_value = 0.05
    else:
        # Rough linear approximation
        p_value = 1.0 - (chi_square / 3.84) * 0.95
    
    return StatisticalTestResult(
        test_name="McNemar's test",
        statistic=chi_square,
        p_value=p_value,
        significant=p_value < alpha,
        confidence_level=1 - alpha,
    )


__all__ = [
    "StatisticalTest",
    "StatisticalTestResult",
    "t_test",
    "bootstrap_confidence_interval",
    "permutation_test",
    "mcnemar_test",
]
