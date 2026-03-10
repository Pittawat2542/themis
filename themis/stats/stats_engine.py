"""Statistical helpers for aggregate reporting and paired comparisons."""

from __future__ import annotations

from typing import Any, cast

from pydantic import BaseModel

from themis._optional import import_optional

np = import_optional("numpy", extra="stats")
stats = import_optional("scipy.stats", extra="stats")


class ComparisonResult(BaseModel):
    """Result of a statistical comparison between two sets of paired observations."""

    baseline_mean: float | None = None
    treatment_mean: float | None = None
    delta_mean: float | None = None
    p_value: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    ci_level: float
    method: str


class StatsEngine:
    """
    Computes aggregate summaries and paired statistical comparisons.
    """

    def aggregate(
        self,
        df: object,
        group_by: list[str],
        metric_col: str = "metric_value",
    ) -> object:
        """
        Groups data and computes standard mean, median, variance.
        """
        frame = cast(Any, df)
        agg = frame.groupby(group_by)[metric_col].agg(
            ["mean", "median", "var", "count"]
        )
        return agg

    def paired_bootstrap(
        self,
        baseline_scores: object,
        treatment_scores: object,
        n_resamples: int = 9999,
        ci: float = 0.95,
    ) -> ComparisonResult:
        """
        Performs a paired bootstrap hypothesis test for the mean difference.
        Null hypothesis: mean_delta == 0
        """
        baseline = cast(Any, baseline_scores)
        treatment = cast(Any, treatment_scores)

        if len(baseline) != len(treatment):
            raise ValueError(
                "Bootstrap paired testing requires equal length score arrays."
            )

        deltas = treatment - baseline
        delta_mean = float(np.mean(deltas))

        # Zero variance bypass (scipy bootstrap throws degenerate data warnings)
        if np.all(deltas == deltas[0]):
            return ComparisonResult(
                baseline_mean=float(np.mean(baseline)),
                treatment_mean=float(np.mean(treatment)),
                delta_mean=delta_mean,
                p_value=1.0,
                ci_lower=delta_mean,
                ci_upper=delta_mean,
                ci_level=ci,
                method="exact_zero_variance",
            )

        # We use scipy's bootstrap
        res = stats.bootstrap(
            (deltas,),
            np.mean,
            n_resamples=n_resamples,
            confidence_level=ci,
            method="BCa",  # Bias-Corrected and Accelerated
            random_state=42,
        )

        # p-value using Wilcoxon signed-rank
        _, p_val = stats.wilcoxon(deltas)

        return ComparisonResult(
            baseline_mean=float(np.mean(baseline)),
            treatment_mean=float(np.mean(treatment)),
            delta_mean=float(delta_mean),
            p_value=float(p_val),
            ci_lower=float(res.confidence_interval.low),
            ci_upper=float(res.confidence_interval.high),
            ci_level=ci,
            method="bootstrap_BCa_wilcoxon",
        )

    def adjust_p_values(
        self,
        p_values: list[float],
        *,
        method: str = "none",
    ) -> list[float]:
        """Apply a supported multiple-comparison correction to p-values."""
        if method == "none":
            return list(p_values)
        if method == "holm":
            indexed = sorted(enumerate(p_values), key=lambda item: item[1])
            adjusted = [0.0] * len(p_values)
            running_max = 0.0
            total = len(p_values)
            for rank, (index, value) in enumerate(indexed, start=1):
                candidate = min(1.0, (total - rank + 1) * value)
                running_max = max(running_max, candidate)
                adjusted[index] = running_max
            return adjusted
        if method == "bh":
            indexed = sorted(
                enumerate(p_values), key=lambda item: item[1], reverse=True
            )
            adjusted = [0.0] * len(p_values)
            running_min = 1.0
            total = len(p_values)
            for reverse_rank, (index, value) in enumerate(indexed, start=1):
                rank = total - reverse_rank + 1
                candidate = min(1.0, value * total / rank)
                running_min = min(running_min, candidate)
                adjusted[index] = running_min
            return adjusted
        raise ValueError(f"Unsupported p-value adjustment method: {method}")
