"""Evaluation report data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import List, Sequence, Dict, Literal

from themis.evaluation.statistics import (
    bootstrap_ci,
    permutation_test,
    cohens_h,
    cohens_d,
)
from themis.evaluation.statistics.types import (
    BootstrapResult,
    PermutationTestResult,
    EffectSize,
)
from themis.core import entities as core_entities
from themis.evaluation.statistics import (
    bootstrap_ci,
    permutation_test,
    cohens_h,
    cohens_d,
    holm_bonferroni,
)


@dataclass
class EvaluationFailure:
    sample_id: str | None
    message: str


@dataclass
class MetricAggregate:
    name: str
    count: int
    mean: float
    per_sample: List[core_entities.MetricScore]

    @classmethod
    def from_scores(
        cls, name: str, scores: List[core_entities.MetricScore]
    ) -> "MetricAggregate":
        if not scores:
            return cls(name=name, count=0, mean=0.0, per_sample=[])
        return cls(
            name=name,
            count=len(scores),
            mean=mean(score.value for score in scores),
            per_sample=scores,
        )


@dataclass
class EvaluationReport:
    metrics: dict[str, MetricAggregate]
    failures: List[EvaluationFailure]
    records: List[core_entities.EvaluationRecord]
    slices: dict[str, dict[str, MetricAggregate]] = field(default_factory=dict)


def _metric_values(report: EvaluationReport, metric_name: str) -> list[float]:
    agg = report.metrics.get(metric_name)
    if not agg:
        return []
    return [s.value for s in agg.per_sample]


def ci_for_metric(
    report: EvaluationReport,
    metric_name: str,
    confidence_level: float = 0.95,
    n_bootstrap: int = 10000,
) -> BootstrapResult:
    values = _metric_values(report, metric_name)
    if not values:
        raise ValueError(f"No scores for metric '{metric_name}'")
    return bootstrap_ci(
        values, n_bootstrap=n_bootstrap, confidence_level=confidence_level
    )


def permutation_test_for_metric(
    report_a: EvaluationReport,
    report_b: EvaluationReport,
    metric_name: str,
    statistic: Literal["mean_diff", "median_diff"] = "mean_diff",
    n_permutations: int = 10000,
    seed: int | None = None,
) -> PermutationTestResult:
    values_a = _metric_values(report_a, metric_name)
    values_b = _metric_values(report_b, metric_name)
    if not values_a or not values_b:
        raise ValueError(f"Both reports must have scores for metric '{metric_name}'")
    return permutation_test(
        values_a,
        values_b,
        statistic=statistic,
        n_permutations=n_permutations,
        seed=seed,
    )


def cohens_h_for_metric(
    report_a: EvaluationReport,
    report_b: EvaluationReport,
    metric_name: str,
) -> EffectSize:
    agg_a = report_a.metrics.get(metric_name)
    agg_b = report_b.metrics.get(metric_name)
    if not agg_a or not agg_b:
        raise ValueError(f"Both reports must have aggregate for metric '{metric_name}'")
    return cohens_h(agg_a.mean, agg_b.mean)


def cohens_d_for_metric(
    report_a: EvaluationReport,
    report_b: EvaluationReport,
    metric_name: str,
) -> EffectSize:
    values_a = _metric_values(report_a, metric_name)
    values_b = _metric_values(report_b, metric_name)
    if len(values_a) < 2 or len(values_b) < 2:
        raise ValueError("Each group must have at least 2 values for Cohen's d")
    return cohens_d(values_a, values_b)


def _slice_metric_values(
    report: EvaluationReport, slice_name: str, metric_name: str
) -> list[float]:
    slice_map = report.slices.get(slice_name)
    if not slice_map:
        return []
    agg = slice_map.get(metric_name)
    if not agg:
        return []
    return [s.value for s in agg.per_sample]


def ci_for_slice_metric(
    report: EvaluationReport,
    slice_name: str,
    metric_name: str,
    confidence_level: float = 0.95,
    n_bootstrap: int = 10000,
) -> BootstrapResult:
    values = _slice_metric_values(report, slice_name, metric_name)
    if not values:
        raise ValueError(
            f"No scores for metric '{metric_name}' in slice '{slice_name}'"
        )
    return bootstrap_ci(
        values, n_bootstrap=n_bootstrap, confidence_level=confidence_level
    )


def compare_reports_with_holm(
    report_a: EvaluationReport,
    report_b: EvaluationReport,
    metric_names: Sequence[str],
    statistic: Literal["mean_diff", "median_diff"] = "mean_diff",
    n_permutations: int = 10000,
    seed: int | None = None,
) -> Dict[str, object]:
    p_values: list[float] = []
    pt_results: Dict[str, PermutationTestResult] = {}
    for name in metric_names:
        pt = permutation_test_for_metric(
            report_a,
            report_b,
            name,
            statistic=statistic,
            n_permutations=n_permutations,
            seed=seed,
        )
        pt_results[name] = pt
        p_values.append(pt.p_value)
    corrected = holm_bonferroni(p_values)
    return {
        "per_metric": pt_results,
        "holm_significant": corrected,
    }


def confusion_matrix(
    labels_true: Sequence[str], labels_pred: Sequence[str]
) -> Dict[str, Dict[str, int]]:
    if len(labels_true) != len(labels_pred):
        raise ValueError("labels_true and labels_pred must have same length")
    cm: Dict[str, Dict[str, int]] = {}
    for t, p in zip(labels_true, labels_pred):
        cm.setdefault(t, {})
        cm[t][p] = cm[t].get(p, 0) + 1
    return cm


__all__ = [
    "EvaluationFailure",
    "MetricAggregate",
    "EvaluationReport",
    "ci_for_metric",
    "ci_for_slice_metric",
    "permutation_test_for_metric",
    "cohens_h_for_metric",
    "cohens_d_for_metric",
    "confusion_matrix",
    "compare_reports_with_holm",
]
