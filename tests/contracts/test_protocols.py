from __future__ import annotations

from collections.abc import Mapping
import inspect

from themis.contracts.protocols import (
    BlobStore,
    DatasetContext,
    DatasetLoader,
    Extractor,
    InferenceEngine,
    InferenceResult,
    JudgeService,
    MetricContext,
    Metric,
    ObservabilityStore,
    ProjectionRepository,
    ProjectionRefreshRepository,
    RuntimeContext,
    TrialEventRepository,
)
from themis.records.candidate import CandidateRecord
from themis.records.conversation import Conversation
from themis.records.extraction import ExtractionRecord
from themis.records.inference import InferenceRecord
from themis.records.judge import JudgeAuditTrail
from themis.records.observability import ObservabilityLink, ObservabilitySnapshot
from themis.records.timeline import RecordTimeline
from themis.records.trial import TrialRecord
from themis.runtime import RecordTimelineView
from themis.specs.base import SpecBase
from themis.specs.experiment import PromptTemplateSpec, TrialSpec
from themis.specs.foundational import JudgeInferenceSpec, TaskSpec
from themis.types.events import ScoreRow, TrialEvent, TrialEventType
from themis.types.json_types import JSONValueType


class MockInferenceEngine(InferenceEngine):
    def infer(
        self,
        trial: TrialSpec,
        context: DatasetContext,
        runtime: RuntimeContext,
    ) -> InferenceResult:
        raise NotImplementedError


class MockExtractor(Extractor):
    def extract(
        self,
        trial: TrialSpec,
        candidate: CandidateRecord,
        config: Mapping[str, JSONValueType] | None = None,
    ) -> ExtractionRecord:
        del trial, candidate, config
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

    def append_event(self, event: TrialEvent, conn=None) -> None:
        del conn
        self.last_event = event

    def last_event_index(
        self, trial_hash: str, candidate_id: str | None = None
    ) -> int | None:
        return None

    def get_events(
        self, trial_hash: str, candidate_id: str | None = None
    ) -> list[TrialEvent]:
        del trial_hash, candidate_id
        return []

    def has_projection_for_overlay(
        self,
        trial_hash: str,
        *,
        transform_hash: str | None = None,
        evaluation_hash: str | None = None,
    ) -> bool:
        del trial_hash, transform_hash, evaluation_hash
        return False

    def latest_terminal_event_type(self, trial_hash: str) -> TrialEventType | None:
        del trial_hash
        return None


class MockDatasetLoader(DatasetLoader):
    def load_task_items(self, task: TaskSpec):
        del task
        return []


class MockBlobStore(BlobStore):
    def put_blob(self, blob: bytes, media_type: str) -> str:
        del blob, media_type
        return "sha256:blob"

    def get_blob(self, ref: str) -> bytes:
        del ref
        return b"{}"

    def write_json(self, data: JSONValueType) -> tuple[str, str]:
        del data
        return ("file:///tmp/blob", "sha256:blob")

    def read_json(self, sha256_hash: str) -> JSONValueType:
        del sha256_hash
        return {}

    def exists(self, sha256_hash: str) -> bool:
        del sha256_hash
        return True


class MockObservabilityStore(ObservabilityStore):
    def save_snapshot(
        self,
        trial_hash: str,
        candidate_id: str | None,
        overlay_key: str,
        snapshot: ObservabilitySnapshot,
    ) -> None:
        del trial_hash, candidate_id, overlay_key, snapshot

    def save_link(
        self,
        trial_hash: str,
        candidate_id: str | None,
        overlay_key: str,
        link: ObservabilityLink,
    ) -> None:
        del trial_hash, candidate_id, overlay_key, link

    def get_snapshot(
        self,
        trial_hash: str,
        candidate_id: str | None,
        overlay_key: str,
    ) -> ObservabilitySnapshot | None:
        del trial_hash, candidate_id, overlay_key
        return None


class MockProjectionRepository(ProjectionRepository):
    def get_trial_record(
        self,
        trial_hash: str,
        *,
        transform_hash: str | None = None,
        evaluation_hash: str | None = None,
    ) -> TrialRecord | None:
        del trial_hash, transform_hash, evaluation_hash
        return None

    def get_conversation(
        self, trial_hash: str, candidate_id: str
    ) -> Conversation | None:
        return None

    def get_record_timeline(
        self,
        record_id: str,
        record_type: str,
        *,
        transform_hash: str | None = None,
        evaluation_hash: str | None = None,
    ) -> RecordTimeline | None:
        del record_id, record_type, transform_hash, evaluation_hash
        return None

    def get_timeline_view(
        self,
        record_id: str,
        record_type: str,
        *,
        transform_hash: str | None = None,
        evaluation_hash: str | None = None,
    ) -> RecordTimelineView | None:
        del record_id, record_type, transform_hash, evaluation_hash
        return None

    def materialize_trial_record(
        self,
        trial_hash: str,
        *,
        transform_hash: str | None = None,
        evaluation_hash: str | None = None,
        extra_events: list[TrialEvent] | None = None,
        conn=None,
    ) -> TrialRecord:
        del trial_hash, transform_hash, evaluation_hash, extra_events, conn
        raise NotImplementedError

    def iter_candidate_scores(
        self,
        *,
        trial_hash: str | None = None,
        metric_id: str | None = None,
        evaluation_hash: str | None = None,
    ):
        del trial_hash, metric_id, evaluation_hash
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
        self,
        record: TrialRecord,
        *,
        transform_hash: str | None = None,
        evaluation_hash: str | None = None,
    ) -> None:
        del record, transform_hash, evaluation_hash

    def has_trial(
        self,
        trial_hash: str,
        *,
        transform_hash: str | None = None,
        evaluation_hash: str | None = None,
    ) -> bool:
        del trial_hash, transform_hash, evaluation_hash
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
    dataset_loader = MockDatasetLoader()
    blob_store = MockBlobStore()
    observability_store = MockObservabilityStore()

    # Check that they can be used with isinstance against their protocol definitions
    assert isinstance(engine, InferenceEngine)
    assert isinstance(extractor, Extractor)
    assert isinstance(metric, Metric)
    assert isinstance(judge, JudgeService)
    assert isinstance(event_repo, TrialEventRepository)
    assert isinstance(projection_repo, ProjectionRepository)
    assert isinstance(projection_repo, ProjectionRefreshRepository)
    assert isinstance(dataset_loader, DatasetLoader)
    assert isinstance(blob_store, BlobStore)
    assert isinstance(observability_store, ObservabilityStore)


def test_storage_protocols_do_not_expose_sqlite_connections() -> None:
    assert "conn" not in inspect.signature(TrialEventRepository.append_event).parameters
    assert (
        "conn"
        not in inspect.signature(
            ProjectionRepository.materialize_trial_record
        ).parameters
    )
    assert (
        "conn"
        not in inspect.signature(
            ProjectionRefreshRepository.materialize_trial_record
        ).parameters
    )


def test_inference_result_envelope_is_frozen() -> None:
    envelope = InferenceResult(
        inference=InferenceRecord(spec_hash="inf_hash", raw_text="42"),
        conversation=Conversation(events=[]),
    )

    assert isinstance(envelope, InferenceResult)
    assert envelope.inference.raw_text == "42"
