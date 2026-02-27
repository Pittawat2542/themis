"""Bootstrap resampling for confidence intervals."""

from __future__ import annotations

import math
import random

from statistics import mean
from collections.abc import Callable, Sequence

from .types import BootstrapResult

from themis.exceptions import MetricError


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    """Return linearly interpolated percentile from a pre-sorted sequence."""
    if not sorted_values:
        raise MetricError("Cannot compute percentile of empty sequence")
    if quantile <= 0:
        return float(sorted_values[0])
    if quantile >= 1:
        return float(sorted_values[-1])

    idx = (len(sorted_values) - 1) * quantile
    lower = math.floor(idx)
    upper = math.ceil(idx)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = idx - lower
    return float(
        sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction
    )


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[Sequence[float]], float] = mean,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    seed: int | None = None,
) -> BootstrapResult:
    """Compute bootstrap confidence interval for a statistic.

    Bootstrap resampling provides non-parametric confidence intervals
    without assuming normality of the underlying distribution.

    Args:
        values: Sample values
        statistic: Function to compute on each bootstrap sample (default: mean)
        n_bootstrap: Number of bootstrap iterations (default: 10000)
        confidence_level: Confidence level (default: 0.95)
        seed: Random seed for reproducibility

    Returns:
        BootstrapResult with CI bounds and point estimate

    Raises:
        ValueError: If values is empty

    Example:
        >>> values = [1.2, 2.3, 3.1, 2.8, 3.5]
        >>> result = bootstrap_ci(values, statistic=mean, n_bootstrap=10000)
        >>> print(f"Mean: {result.statistic:.2f}, 95% CI: [{result.ci_lower:.2f}, {result.ci_upper:.2f}]")
    """
    if not values:
        raise MetricError("Cannot compute bootstrap CI for empty sequence")

    rng = random.Random(seed)

    n = len(values)
    values_list = list(values)

    # Compute observed statistic
    observed_stat = statistic(values_list)

    # Bootstrap iterations
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        resample = rng.choices(values_list, k=n)
        boot_stat = statistic(resample)
        bootstrap_stats.append(boot_stat)

    # Sort bootstrap statistics
    bootstrap_stats.sort()

    # Compute percentile CI
    alpha = 1 - confidence_level
    lower_q = alpha / 2
    upper_q = 1 - alpha / 2

    return BootstrapResult(
        statistic=observed_stat,
        ci_lower=_percentile(bootstrap_stats, lower_q),
        ci_upper=_percentile(bootstrap_stats, upper_q),
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
    )
