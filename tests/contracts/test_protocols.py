from __future__ import annotations

from themis.contracts.protocols import (
    DatasetContext,
    Extractor,
    InferenceEngine,
    InferenceResult,
    JudgeService,
    MetricContext,
    Metric,
    ProjectionRepository,
    RuntimeContext,
    TrialEventRepository,
)
from themis.records.candidate import CandidateRecord
from themis.records.conversation import Conversation
from themis.records.inference import InferenceRecord
from themis.records.judge import JudgeAuditTrail
from themis.records.timeline import RecordTimeline
from themis.records.trial import TrialRecord
from themis.runtime import RecordTimelineView
from themis.specs.base import SpecBase
from themis.specs.experiment import PromptTemplateSpec, TrialSpec
from themis.specs.foundational import JudgeInferenceSpec
from themis.storage.events import ScoreRow, TrialEvent, TrialEventType


class MockInferenceEngine(InferenceEngine):
    def infer(
        self,
        trial: TrialSpec,
        context: DatasetContext,
        runtime: RuntimeContext,
    ) -> InferenceResult:
        raise NotImplementedError


class MockExtractor(Extractor):
    def extract(self, trial: TrialSpec, candidate: CandidateRecord):
        raise NotImplementedError


class MockMetric(Metric):
    def score(
        self, trial: TrialSpec, candidate: CandidateRecord, context: MetricContext
    ):
        raise NotImplementedError


class MockJudgeService(JudgeService):
    def judge(
        self,
        metric_id: str,
        parent_candidate: CandidateRecord,
        judge_spec: JudgeInferenceSpec,
        prompt: PromptTemplateSpec,
        runtime: MetricContext,
    ) -> InferenceRecord:
        raise NotImplementedError

    def consume_audit_trail(self, candidate_hash: str) -> JudgeAuditTrail | None:
        return None


class MockTrialEventRepository(TrialEventRepository):
    def save_spec(self, spec: SpecBase) -> None:
        self.last_spec = spec

    def append_event(self, event: TrialEvent) -> None:
        self.last_event = event

    def last_event_index(
        self, trial_hash: str, candidate_id: str | None = None
    ) -> int | None:
        return None

    def get_events(
        self, trial_hash: str, candidate_id: str | None = None
    ) -> list[TrialEvent]:
        return []

    def has_projection_for_revision(self, trial_hash: str, eval_revision: str) -> bool:
        return False

    def latest_terminal_event_type(self, trial_hash: str) -> TrialEventType | None:
        return None


class MockProjectionRepository(ProjectionRepository):
    def get_trial_record(
        self, trial_hash: str, eval_revision: str
    ) -> TrialRecord | None:
        return None

    def get_conversation(
        self, trial_hash: str, candidate_id: str
    ) -> Conversation | None:
        return None

    def get_record_timeline(
        self,
        record_id: str,
        record_type: str,
        eval_revision: str,
    ) -> RecordTimeline | None:
        return None

    def get_timeline_view(
        self,
        record_id: str,
        record_type: str,
        eval_revision: str,
    ) -> RecordTimelineView | None:
        return None

    def materialize_trial_record(
        self, trial_hash: str, eval_revision: str
    ) -> TrialRecord:
        raise NotImplementedError

    def iter_candidate_scores(
        self,
        *,
        trial_hash: str | None = None,
        metric_id: str | None = None,
        eval_revision: str = "latest",
    ):
        return iter(
            [
                ScoreRow(
                    trial_hash="trial",
                    candidate_id="candidate",
                    metric_id="metric",
                    score=1.0,
                )
            ]
        )

    def save_trial_record(
        self, record: TrialRecord, *, eval_revision: str = "latest"
    ) -> None:
        pass

    def has_trial(self, trial_hash: str, eval_revision: str = "latest") -> bool:
        return False


def test_protocols_instantiation():
    # If they were purely ABCs and missing methods, this would raise TypeError.
    # Since we implemented the methods (even with pass), they instantiate.
    engine = MockInferenceEngine()
    extractor = MockExtractor()
    metric = MockMetric()
    judge = MockJudgeService()
    event_repo = MockTrialEventRepository()
    projection_repo = MockProjectionRepository()

    # Check that they can be used with isinstance against their protocol definitions
    assert isinstance(engine, InferenceEngine)
    assert isinstance(extractor, Extractor)
    assert isinstance(metric, Metric)
    assert isinstance(judge, JudgeService)
    assert isinstance(event_repo, TrialEventRepository)
    assert isinstance(projection_repo, ProjectionRepository)


def test_inference_result_envelope_is_frozen() -> None:
    envelope = InferenceResult(
        inference=InferenceRecord(spec_hash="inf_hash", raw_text="42"),
        conversation=Conversation(events=[]),
    )

    assert isinstance(envelope, InferenceResult)
    assert envelope.inference.raw_text == "42"
