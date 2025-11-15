"""Effect size measures for statistical comparisons."""

from __future__ import annotations

import math
from typing import Sequence

from .types import EffectSize


def cohens_h(p1: float, p2: float) -> EffectSize:
    """Compute Cohen's h effect size for comparing two proportions.

    Cohen's h measures the distance between two proportions using
    the arcsine transformation. This is useful for comparing success
    rates, accuracy proportions, etc.

    Args:
        p1: Proportion for group 1 (e.g., baseline accuracy)
        p2: Proportion for group 2 (e.g., treatment accuracy)

    Returns:
        EffectSize with value and interpretation

    Interpretation:
        - |h| < 0.2: negligible
        - 0.2 <= |h| < 0.5: small
        - 0.5 <= |h| < 0.8: medium
        - |h| >= 0.8: large

    Example:
        >>> # Baseline: 65% accuracy, Treatment: 75% accuracy
        >>> effect = cohens_h(0.65, 0.75)
        >>> print(f"Effect: {effect.value:.3f} ({effect.interpretation})")
    """
    # Arcsine transformation
    phi1 = 2 * math.asin(math.sqrt(p1))
    phi2 = 2 * math.asin(math.sqrt(p2))

    h = phi2 - phi1

    # Interpret effect size
    abs_h = abs(h)
    if abs_h < 0.2:
        interpretation = "negligible"
    elif abs_h < 0.5:
        interpretation = "small"
    elif abs_h < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"

    return EffectSize(
        name="cohen_h",
        value=h,
        interpretation=interpretation,
    )


def cohens_d(group1: Sequence[float], group2: Sequence[float]) -> EffectSize:
    """Compute Cohen's d effect size for comparing two means.

    Cohen's d measures the standardized difference between two group means.
    This is the most common effect size for t-tests.

    Args:
        group1: Values from first group (e.g., baseline)
        group2: Values from second group (e.g., treatment)

    Returns:
        EffectSize with value and interpretation

    Interpretation:
        - |d| < 0.2: negligible
        - 0.2 <= |d| < 0.5: small
        - 0.5 <= |d| < 0.8: medium
        - |d| >= 0.8: large

    Example:
        >>> baseline = [1.2, 1.5, 1.3, 1.4]
        >>> treatment = [1.8, 2.0, 1.9, 2.1]
        >>> effect = cohens_d(baseline, treatment)
    """
    from statistics import mean, stdev

    n1 = len(group1)
    n2 = len(group2)

    if n1 < 2 or n2 < 2:
        raise ValueError("Each group must have at least 2 values")

    mean1 = mean(group1)
    mean2 = mean(group2)
    std1 = stdev(group1)
    std2 = stdev(group2)

    # Pooled standard deviation
    pooled_std = math.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

    if pooled_std == 0:
        # No variance - return 0 if means are equal, infinity otherwise
        if mean1 == mean2:
            d = 0.0
        else:
            d = float("inf")
    else:
        d = (mean2 - mean1) / pooled_std

    # Interpret effect size
    abs_d = abs(d)
    if abs_d < 0.2:
        interpretation = "negligible"
    elif abs_d < 0.5:
        interpretation = "small"
    elif abs_d < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"

    return EffectSize(
        name="cohen_d",
        value=d,
        interpretation=interpretation,
    )
