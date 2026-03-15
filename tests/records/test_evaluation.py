from themis.records.evaluation import MetricScore, EvaluationRecord
from themis.types.enums import RecordStatus


def test_evaluation_record():
    score1 = MetricScore(
        metric_id="exact_match", value=1.0, details={"case_sensitive": False}
    )
    score2 = MetricScore(metric_id="bleu", value=0.85)

    record = EvaluationRecord(
        spec_hash="eval1",
        metric_scores=[score1, score2],
        aggregate_scores={"exact_match": 1.0, "bleu": 0.85},
    )

    assert record.status == RecordStatus.OK
    assert len(record.metric_scores) == 2
    assert record.aggregate_scores["exact_match"] == 1.0
