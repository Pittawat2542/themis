import pytest

from themis.evaluation import metrics


def test_exact_match_normalizes_whitespace_and_case():
    metric = metrics.ExactMatch(case_sensitive=False, strip_whitespace=True)

    score = metric.compute(prediction="  Paris ", references=["paris", "london"])

    assert score.value == 1.0
    assert score.details["matched_reference"] == "paris"


def test_exact_match_handles_case_sensitive_mode():
    metric = metrics.ExactMatch(case_sensitive=True)

    score = metric.compute(prediction="Paris", references=["paris"])

    assert score.value == 0.0
    assert score.details["matched_reference"] is None


def test_metric_result_supports_metadata_passthrough():
    metric = metrics.ExactMatch(case_sensitive=False, strip_whitespace=True)

    score = metric.compute(
        prediction="Bangkok",
        references=["Bangkok"],
        metadata={"sample_id": "sample-1"},
    )

    assert score.metadata["sample_id"] == "sample-1"


def test_composite_metric_combines_children_scores():
    exact = metrics.ExactMatch()
    length = metrics.LengthDifferenceTolerance(max_delta=2)
    composite = metrics.CompositeMetric(children=[exact, length])

    score = composite.compute(
        prediction="Paris",
        references=["Paris"],
        metadata={"sample_id": "sample-42"},
    )

    assert score.value == pytest.approx(1.0)
    assert "ExactMatch" in score.details
    assert "LengthDifferenceTolerance" in score.details
    assert score.metadata["sample_id"] == "sample-42"
