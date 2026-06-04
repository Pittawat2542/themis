from __future__ import annotations

from themis.core.models import MetricResult
from themis.core.reliability import (
    calibration_error,
    inter_judge_agreement,
    sensitivity_summary,
)
from themis.core.workflows import ParsedJudgment


def test_inter_judge_agreement_counts_majority_share() -> None:
    judgments = [
        ParsedJudgment(label="pass", score=1.0),
        ParsedJudgment(label="pass", score=1.0),
        ParsedJudgment(label="fail", score=0.0),
    ]

    result = inter_judge_agreement("metric/agreement", judgments)

    assert result.result_type == "agreement"
    assert result.value == 2 / 3
    assert result.labels == {"majority_label": "pass"}
    assert result.dimensions == {"judge_count": 3.0, "distinct_labels": 2.0}


def test_calibration_error_uses_confidence_and_correctness_gap() -> None:
    results = [
        MetricResult(metric_id="metric/a", value=1.0, confidence=0.9),
        MetricResult(metric_id="metric/a", value=0.0, confidence=0.8),
    ]

    result = calibration_error("metric/calibration", results)

    assert result.result_type == "calibration"
    assert result.value == 0.45
    assert result.dimensions == {"sample_count": 2.0}


def test_sensitivity_summary_reports_value_range() -> None:
    results = [
        MetricResult(metric_id="metric/a", value=0.2),
        MetricResult(metric_id="metric/a", value=0.8),
        MetricResult(metric_id="metric/a", value=None),
    ]

    result = sensitivity_summary("metric/sensitivity", results)

    assert result.value == 0.6
    assert result.dimensions == {"min": 0.2, "max": 0.8, "sample_count": 2.0}
