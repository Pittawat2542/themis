from datetime import datetime, timezone

from themis.records.candidate import CandidateRecord
from themis.records.conversation import Conversation, MessageEvent, MessagePayload
from themis.records.evaluation import EvaluationRecord, MetricScore
from themis.records.extraction import ExtractionRecord
from themis.records.inference import InferenceRecord
from themis.records.timeline import RecordTimeline, TimelineStageRecord
from themis.records.trial import TrialRecord
from themis.types.enums import RecordStatus


def test_candidate_record_aggregate():
    inf = InferenceRecord(spec_hash="inf1", raw_text="42")
    ext = ExtractionRecord(
        spec_hash="ext1", extractor_id="json", success=True, parsed_answer=42
    )
    eval_rec = EvaluationRecord(
        spec_hash="ev1", metric_scores=[MetricScore(metric_id="exact_match", value=1.0)]
    )

    cand = CandidateRecord(
        spec_hash="cand1", inference=inf, extractions=[ext], evaluation=eval_rec
    )

    assert cand.status == RecordStatus.OK
    assert cand.extractions[0].parsed_answer == 42
    assert cand.evaluation.aggregate_scores["exact_match"] == 1.0


def test_trial_record_aggregate():
    cand1 = CandidateRecord(
        spec_hash="cand1", inference=InferenceRecord(spec_hash="inf1", raw_text="42")
    )
    cand2 = CandidateRecord(
        spec_hash="cand2", inference=InferenceRecord(spec_hash="inf2", raw_text="24")
    )

    trial = TrialRecord(spec_hash="trial1", candidates=[cand1, cand2])

    assert trial.status == RecordStatus.OK
    assert len(trial.candidates) == 2


def test_candidate_record_exposes_v2_public_fields():
    conversation = Conversation(
        events=[
            MessageEvent(
                role="assistant",
                payload=MessagePayload(content="42"),
                event_index=0,
            )
        ]
    )
    timeline = RecordTimeline(
        record_id="cand1",
        record_type="candidate",
        trial_hash="trial1",
        candidate_id="cand1",
        item_id="item1",
        stages=[
            TimelineStageRecord(
                stage="inference",
                status=RecordStatus.OK,
                started_at=datetime.now(timezone.utc),
                ended_at=datetime.now(timezone.utc),
                duration_ms=1,
            )
        ],
    )

    cand = CandidateRecord(
        spec_hash="cand1",
        candidate_id="cand1",
        sample_index=2,
        conversation=conversation,
        timeline=timeline,
    )

    assert cand.candidate_id == "cand1"
    assert cand.sample_index == 2
    assert cand.conversation == conversation
    assert cand.timeline == timeline
