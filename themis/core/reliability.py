"""Reliability analysis helpers for rich metric results."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from themis.core.models import MetricResult
from themis.core.workflows import ParsedJudgment


def inter_judge_agreement(
    metric_id: str, judgments: Iterable[ParsedJudgment]
) -> MetricResult:
    """Return the majority-label share across parsed judge judgments."""

    labels = [judgment.label for judgment in judgments]
    if not labels:
        return MetricResult(
            metric_id=metric_id,
            result_type="agreement",
            value=None,
            dimensions={"judge_count": 0.0, "distinct_labels": 0.0},
        )
    counts = Counter(labels)
    majority_label, majority_count = counts.most_common(1)[0]
    return MetricResult(
        metric_id=metric_id,
        result_type="agreement",
        value=majority_count / len(labels),
        dimensions={
            "judge_count": float(len(labels)),
            "distinct_labels": float(len(counts)),
        },
        labels={"majority_label": majority_label},
        metadata={"label_counts": dict(counts)},
    )


def calibration_error(metric_id: str, results: Iterable[MetricResult]) -> MetricResult:
    """Return mean absolute gap between confidence and binary correctness."""

    gaps = [
        abs(float(result.confidence) - float(result.value or 0.0))
        for result in results
        if result.confidence is not None and result.value is not None
    ]
    if not gaps:
        return MetricResult(
            metric_id=metric_id,
            result_type="calibration",
            value=None,
            dimensions={"sample_count": 0.0},
        )
    return MetricResult(
        metric_id=metric_id,
        result_type="calibration",
        value=sum(gaps) / len(gaps),
        dimensions={"sample_count": float(len(gaps))},
    )


def sensitivity_summary(
    metric_id: str, results: Iterable[MetricResult]
) -> MetricResult:
    """Return the range of numeric metric values across variants."""

    values = [float(result.value) for result in results if result.value is not None]
    if not values:
        return MetricResult(
            metric_id=metric_id,
            result_type="scalar",
            value=None,
            dimensions={"sample_count": 0.0},
        )
    minimum = min(values)
    maximum = max(values)
    return MetricResult(
        metric_id=metric_id,
        result_type="scalar",
        value=round(maximum - minimum, 12),
        dimensions={
            "min": minimum,
            "max": maximum,
            "sample_count": float(len(values)),
        },
    )
