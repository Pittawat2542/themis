"""Statistical result types and dataclasses."""

from __future__ import annotations

from dataclasses import dataclass


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


@dataclass
class PermutationTestResult:
    """Result of a permutation test.

    Attributes:
        observed_statistic: Observed test statistic
        p_value: Permutation test p-value
        n_permutations: Number of permutations performed
        is_significant: Whether result is significant at alpha=0.05
    """

    observed_statistic: float
    p_value: float
    n_permutations: int
    is_significant: bool


@dataclass
class BootstrapResult:
    """Result of bootstrap resampling.

    Attributes:
        statistic: Point estimate of the statistic
        ci_lower: Lower bound of bootstrap CI
        ci_upper: Upper bound of bootstrap CI
        confidence_level: Confidence level used
        n_bootstrap: Number of bootstrap iterations
    """

    statistic: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    n_bootstrap: int


@dataclass
class EffectSize:
    """Effect size measure.

    Attributes:
        name: Name of effect size measure (e.g., "cohen_h", "cohen_d")
        value: Effect size value
        interpretation: Text interpretation (e.g., "small", "medium", "large")
    """

    name: str
    value: float
    interpretation: str
