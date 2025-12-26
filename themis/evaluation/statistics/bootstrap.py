"""Bootstrap resampling for confidence intervals."""

from __future__ import annotations

import random
from statistics import mean
from typing import Callable, Sequence

from .types import BootstrapResult


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
        raise ValueError("Cannot compute bootstrap CI for empty sequence")

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
    lower_idx = int(n_bootstrap * alpha / 2)
    upper_idx = int(n_bootstrap * (1 - alpha / 2))

    # Ensure indices are within bounds
    lower_idx = max(0, min(lower_idx, n_bootstrap - 1))
    upper_idx = max(0, min(upper_idx, n_bootstrap - 1))

    return BootstrapResult(
        statistic=observed_stat,
        ci_lower=bootstrap_stats[lower_idx],
        ci_upper=bootstrap_stats[upper_idx],
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
    )
