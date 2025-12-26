"""Helper functions for statistical distributions."""

from __future__ import annotations

import math


def inverse_normal_cdf(p: float) -> float:
    """Approximate inverse normal CDF (probit function) for standard normal.

    Uses Beasley-Springer-Moro approximation.

    Args:
        p: Probability value between 0 and 1

    Returns:
        z-score corresponding to probability p

    Raises:
        ValueError: If p is not between 0 and 1
    """
    if p <= 0 or p >= 1:
        raise ValueError("Probability must be between 0 and 1")

    # Constants for approximation
    a = [2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637]
    b = [-8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833]
    c = [
        0.3374754822726147,
        0.9761690190917186,
        0.1607979714918209,
        0.0276438810333863,
        0.0038405729373609,
        0.0003951896511919,
        0.0000321767881768,
        0.0000002888167364,
        0.0000003960315187,
    ]

    # Transform to standard normal
    y = p - 0.5
    if abs(y) < 0.42:
        # Central region
        r = y * y
        x = (
            y
            * (((a[3] * r + a[2]) * r + a[1]) * r + a[0])
            / (((b[3] * r + b[2]) * r + b[1]) * r + b[0] + 1.0)
        )
        return x
    else:
        # Tail region
        r = p if y > 0 else 1 - p
        r = math.log(-math.log(r))
        x = c[0] + r * (
            c[1]
            + r
            * (
                c[2]
                + r
                * (c[3] + r * (c[4] + r * (c[5] + r * (c[6] + r * (c[7] + r * c[8])))))
            )
        )
        if y < 0:
            x = -x
        return x


def t_critical_value(df: int, confidence_level: float) -> float:
    """Approximate t-distribution critical value.

    This is a simplified approximation. For production use, consider scipy.stats.t.ppf.

    Args:
        df: Degrees of freedom
        confidence_level: Confidence level (e.g., 0.95)

    Returns:
        Critical value for two-tailed test
    """
    try:
        from scipy import stats
    except Exception:  # pragma: no cover - optional dependency
        stats = None

    if stats is not None:
        alpha = (1 - confidence_level) / 2
        return float(stats.t.ppf(1 - alpha, df))

    # For common confidence levels and degrees of freedom, use lookup table
    # Otherwise, use normal approximation for large df
    if df >= 30:
        # Use normal approximation for large df
        alpha = (1 - confidence_level) / 2
        return inverse_normal_cdf(1 - alpha)

    # Simplified lookup table for small df (two-tailed)
    # Format: {confidence_level: {df: critical_value}}
    lookup_95 = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        15: 2.131,
        20: 2.086,
        25: 2.060,
        29: 2.045,
    }
    lookup_99 = {
        1: 63.657,
        2: 9.925,
        3: 5.841,
        4: 4.604,
        5: 4.032,
        6: 3.707,
        7: 3.499,
        8: 3.355,
        9: 3.250,
        10: 3.169,
        15: 2.947,
        20: 2.845,
        25: 2.787,
        29: 2.756,
    }

    if abs(confidence_level - 0.95) < 0.01:
        lookup = lookup_95
    elif abs(confidence_level - 0.99) < 0.01:
        lookup = lookup_99
    else:
        # Fall back to normal approximation
        alpha = (1 - confidence_level) / 2
        return inverse_normal_cdf(1 - alpha)

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


def t_to_p_value(t_stat: float, df: int) -> float:
    """Approximate two-tailed p-value for t-statistic.

    This is a simplified approximation. For production use, consider scipy.stats.t.cdf.

    Args:
        t_stat: t-statistic value
        df: Degrees of freedom

    Returns:
        Two-tailed p-value
    """
    try:
        from scipy import stats
    except Exception:  # pragma: no cover - optional dependency
        stats = None

    if stats is not None:
        p_one_tail = stats.t.cdf(-abs(t_stat), df)
        return float(2 * p_one_tail)

    # For large df, use normal approximation
    if df >= 30:
        # Use normal distribution CDF
        p_one_tail = normal_cdf(-abs(t_stat))
        return 2 * p_one_tail

    # For small df, use approximation
    # Very rough approximation: convert t to approximate p-value
    if abs(t_stat) < 0.5:
        return 1.0
    elif abs(t_stat) > 10:
        return 0.0001
    else:
        # Rough approximation using exponential decay
        base_p = math.exp(-abs(t_stat) * 0.5) * (df / (df + t_stat**2))
        return min(1.0, 2 * base_p)


def normal_cdf(x: float) -> float:
    """Standard normal CDF using error function approximation.

    Args:
        x: Value to evaluate CDF at

    Returns:
        Cumulative probability
    """
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))
