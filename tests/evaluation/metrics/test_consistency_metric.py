from themis.evaluation import metrics


def test_consistency_metric_with_reference_and_without_reference():
    metric = metrics.ConsistencyMetric()

    score_with_ref = metric.compute(
        prediction=["Paris", "Paris", "London"], references=["Paris"]
    )
    assert score_with_ref.value == 2 / 3
    assert score_with_ref.details["agreement"] == 2 / 3

    score_no_ref = metric.compute(prediction=["A", "B", "A", "C"], references=[])
    assert score_no_ref.value == 0.5
    assert 0.0 <= score_no_ref.details["flip_rate"] <= 1.0
