from themis.types.enums import PromptRole, RunStage
import asyncio

import pytest

from themis.contracts.protocols import (
    DatasetContext,
    Extractor,
    InferenceEngine,
    InferenceResult,
    Metric,
    RuntimeContext,
    TrialEventRepository,
)
from themis.errors import (
    ExtractionError,
    InferenceError,
    MetricError,
    SpecValidationError,
)
from themis.orchestration.trial_runner import TrialRunner
from themis.orchestration.task_resolution import resolve_task_stages
from themis.records.candidate import CandidateRecord
from themis.records.conversation import Conversation, MessageEvent, MessagePayload
from themis.records.evaluation import MetricScore
from themis.records.extraction import ExtractionRecord
from themis.records.inference import InferenceRecord, TokenUsage
from themis.registry.plugin_registry import PluginRegistry
from themis.specs.foundational import (
    DatasetSpec,
    EvaluationSpec,
    ExtractorChainSpec,
    GenerationSpec,
    JudgeInferenceSpec,
    ModelSpec,
    OutputTransformSpec,
    TaskSpec,
)
from themis.specs.experiment import (
    InferenceParamsSpec,
    PromptMessage,
    PromptTemplateSpec,
    TrialSpec,
)
from themis.telemetry.bus import TelemetryBus, TelemetryEventName
from themis.storage.sqlite_schema import DatabaseManager
from themis.storage.artifact_store import ArtifactStore
from themis.storage.event_repo import SqliteEventRepository
from themis.storage.projection_repo import SqliteProjectionRepository
from themis.types.enums import ErrorCode, RecordStatus as EventStatus, DatasetSource
from themis.types.events import (
    EvaluationCompletedEventMetadata,
    ExtractionCompletedEventMetadata,
    InferenceCompletedEventMetadata,
    ItemLoadedEventMetadata,
    ProjectionCompletedEventMetadata,
    PromptRenderedEventMetadata,
    TimelineStage,
    TrialEvent,
    TrialEventType,
)
from themis.types.enums import RecordStatus


class MockEventRepo(TrialEventRepository):
    def __init__(self):
        self.events = []
        self.saved_specs = []

    def save_spec(self, spec) -> None:
        self.saved_specs.append(spec)

    def append_event(self, event: TrialEvent, conn=None):
        del conn
        self.events.append(event)

    def last_event_index(self, trial_hash, candidate_id=None):
        matching = [
            event.event_seq
            for event in self.events
            if event.trial_hash == trial_hash
            and (candidate_id is None or event.candidate_id == candidate_id)
        ]
        return max(matching) if matching else None

    def get_events(self, trial_hash, candidate_id=None):
        return [
            event
            for event in self.events
            if event.trial_hash == trial_hash
            and (candidate_id is None or event.candidate_id == candidate_id)
        ]

    def has_projection_for_overlay(
        self,
        trial_hash,
        *,
        transform_hash=None,
        evaluation_hash=None,
    ):
        return any(
            event.trial_hash == trial_hash
            and event.event_type == TrialEventType.PROJECTION_COMPLETED
            and isinstance(event.metadata, ProjectionCompletedEventMetadata)
            and event.metadata.transform_hash == transform_hash
            and event.metadata.evaluation_hash == evaluation_hash
            for event in self.events
        )

    def latest_terminal_event_type(self, trial_hash):
        for event in reversed(self.events):
            if event.trial_hash != trial_hash:
                continue
            if event.event_type in {
                TrialEventType.TRIAL_COMPLETED,
                TrialEventType.TRIAL_FAILED,
            }:
                return event.event_type
        return None


class MockInferenceEngine(InferenceEngine):
    def __init__(self, fail_count=0):
        self.fail_count = fail_count
        self.attempts = 0

    def infer(self, trial, context: DatasetContext, runtime):
        self.attempts += 1
        if self.attempts <= self.fail_count:
            raise InferenceError(
                code=ErrorCode.PROVIDER_RATE_LIMIT, message="Rate limit", details={}
            )
        return InferenceResult(
            inference=InferenceRecord(
                spec_hash="inf_hash",
                raw_text="The answer is 42.",
                latency_ms=42,
                provider_request_id="req_123",
                token_usage=TokenUsage(
                    prompt_tokens=10, completion_tokens=3, total_tokens=13
                ),
            ),
            conversation=Conversation(
                events=[
                    MessageEvent(
                        role=PromptRole.ASSISTANT,
                        payload=MessagePayload(content="The answer is 42."),
                        event_index=0,
                    )
                ]
            ),
        )


class MockExtractor(Extractor):
    def extract(self, trial, candidate, config=None):
        return ExtractionRecord(
            spec_hash="ext_hash", extractor_id="mock", success=True, parsed_answer="42"
        )


class MockMetric(Metric):
    def score(self, trial, candidate, context):
        return MetricScore(metric_id="em", value=1.0, details={"matched": True})


class JudgeAwareMetric(Metric):
    def score(self, trial, candidate, context):
        judge_service = context["judge_service"]
        judge_inference = judge_service.judge(
            metric_id="judge_metric",
            parent_candidate=candidate,
            judge_spec=JudgeInferenceSpec(
                model=ModelSpec(model_id="judge-model", provider="judge_provider")
            ),
            prompt=PromptTemplateSpec(
                id="judge_prompt",
                messages=[{"role": "user", "content": "Rate this answer."}],
            ),
            runtime={
                "task_spec": trial.task,
                "dataset_context": context,
            },
        )
        return MetricScore(
            metric_id="judge_metric",
            value=1.0,
            details={"judge_raw_text": judge_inference.raw_text},
        )


class MockJudgeInferenceEngine(InferenceEngine):
    def infer(self, trial, context: DatasetContext, runtime):
        assert isinstance(runtime, RuntimeContext)
        return InferenceResult(
            inference=InferenceRecord(
                spec_hash="judge_inf_hash",
                raw_text="SCORE: 5/5",
                provider_request_id="judge_req_123",
            )
        )


@pytest.fixture
def trial_spec():
    return TrialSpec(
        trial_id="test_trial",
        model=ModelSpec(model_id="gpt-4", provider="openai"),
        task=TaskSpec(
            task_id="t1",
            dataset=DatasetSpec(source=DatasetSource.MEMORY),
            generation=GenerationSpec(),
            output_transforms=[
                OutputTransformSpec(
                    name="parsed",
                    extractor_chain=ExtractorChainSpec(extractors=["mock"]),
                )
            ],
            evaluations=[
                EvaluationSpec(name="score", transform="parsed", metrics=["em"])
            ],
        ),
        item_id="item1",
        prompt=PromptTemplateSpec(messages=[]),
        params=InferenceParamsSpec(),
        candidate_count=2,
    )


def test_trial_runner_success(trial_spec):
    registry = PluginRegistry()
    registry.register_inference_engine("openai", MockInferenceEngine())
    registry.register_extractor("mock", MockExtractor())
    registry.register_metric("em", MockMetric())

    repo = MockEventRepo()
    runner = TrialRunner(registry, event_repo=repo)

    trial_record = runner.run_trial(trial_spec, {}, RuntimeContext())

    assert trial_record.status == RecordStatus.OK
    assert len(trial_record.candidates) == 2
    assert trial_record.candidates[0].status == RecordStatus.OK
    assert trial_record.candidates[1].status == RecordStatus.OK

    # Event verification
    event_types = [event.event_type for event in repo.events]
    assert event_types[0] == "trial_started"
    assert "item_loaded" in event_types
    assert "prompt_rendered" in event_types
    assert event_types.count("candidate_started") == 2
    assert event_types.count("inference_completed") == 2
    assert event_types.count("extraction_completed") == 2
    assert event_types.count("evaluation_completed") == 2
    assert event_types.count("candidate_completed") == 2
    assert event_types[-1] == "trial_completed"
    assert repo.last_event_index(trial_record.spec_hash) == len(repo.events)


def test_trial_runner_continues_when_telemetry_subscriber_fails(trial_spec, caplog):
    registry = PluginRegistry()
    registry.register_inference_engine("openai", MockInferenceEngine())
    registry.register_extractor("mock", MockExtractor())
    registry.register_metric("em", MockMetric())

    repo = MockEventRepo()
    bus = TelemetryBus()
    seen = []

    def explode(event):
        raise RuntimeError("telemetry boom")

    bus.subscribe(explode)
    bus.subscribe(seen.append)
    runner = TrialRunner(
        registry,
        event_repo=repo,
        telemetry_bus=bus,
        parallel_candidates=1,
    )

    with caplog.at_level("ERROR"):
        record = runner.run_trial(
            trial_spec.model_copy(update={"candidate_count": 1}),
            {},
            RuntimeContext(),
        )

    assert record.status == RecordStatus.OK
    assert TelemetryEventName.TRIAL_START in [event.name for event in seen]
    assert TelemetryEventName.METRIC_END in [event.name for event in seen]
    assert TelemetryEventName.TRIAL_END in [event.name for event in seen]
    assert "Telemetry subscriber failed for" in caplog.text


def test_trial_runner_prepares_full_resolved_stage_runtime(trial_spec):
    registry = PluginRegistry()
    registry.register_inference_engine("openai", MockInferenceEngine())
    registry.register_extractor("mock", MockExtractor())
    registry.register_metric("em", MockMetric())

    runner = TrialRunner(registry, event_repo=MockEventRepo())
    session = runner.prepare_trial_session(trial_spec, {}, RuntimeContext())

    assert session.resolved_plugins is not None
    assert session.resolved_plugins.generation is not None
    assert len(session.resolved_plugins.output_transforms) == len(
        session.resolved_stages.output_transforms
    )
    assert len(session.resolved_plugins.evaluations) == len(
        session.resolved_stages.evaluations
    )


def test_trial_runner_can_prepare_generation_only_runtime(trial_spec):
    registry = PluginRegistry()
    registry.register_inference_engine("openai", MockInferenceEngine())
    registry.register_extractor("mock", MockExtractor())
    registry.register_metric("em", MockMetric())

    runner = TrialRunner(registry, event_repo=MockEventRepo())
    session = runner.prepare_trial_session(
        trial_spec,
        {},
        RuntimeContext(),
        required_stages={RunStage.GENERATION},
    )

    assert session.resolved_plugins is not None
    assert session.resolved_plugins.generation is not None
    assert session.resolved_plugins.output_transforms == ()
    assert session.resolved_plugins.evaluations == ()


def test_trial_runner_emits_single_candidate_events_in_exact_stage_order(trial_spec):
    registry = PluginRegistry()
    registry.register_inference_engine("openai", MockInferenceEngine())
    registry.register_extractor("mock", MockExtractor())
    registry.register_metric("em", MockMetric())

    repo = MockEventRepo()
    runner = TrialRunner(registry, event_repo=repo, parallel_candidates=1)
    executed_trial = trial_spec.model_copy(update={"candidate_count": 1})

    trial_record = runner.run_trial(
        executed_trial,
        {"question": "6 * 7", "answer": "42"},
        RuntimeContext(),
    )

    candidate_id = trial_record.candidates[0].candidate_id
    assert candidate_id is not None

    events = repo.get_events(executed_trial.spec_hash)
    assert [event.event_seq for event in events] == list(range(1, len(events) + 1))
    assert [event.event_type for event in events] == [
        TrialEventType.TRIAL_STARTED,
        TrialEventType.ITEM_LOADED,
        TrialEventType.PROMPT_RENDERED,
        TrialEventType.CANDIDATE_STARTED,
        TrialEventType.PROMPT_RENDERED,
        TrialEventType.INFERENCE_COMPLETED,
        TrialEventType.CONVERSATION_EVENT,
        TrialEventType.CANDIDATE_COMPLETED,
        TrialEventType.EXTRACTION_COMPLETED,
        TrialEventType.EVALUATION_COMPLETED,
        TrialEventType.TRIAL_COMPLETED,
    ]
    assert [
        event.event_type
        for event in repo.get_events(executed_trial.spec_hash, candidate_id)
    ] == [
        TrialEventType.CANDIDATE_STARTED,
        TrialEventType.PROMPT_RENDERED,
        TrialEventType.INFERENCE_COMPLETED,
        TrialEventType.CONVERSATION_EVENT,
        TrialEventType.CANDIDATE_COMPLETED,
        TrialEventType.EXTRACTION_COMPLETED,
        TrialEventType.EVALUATION_COMPLETED,
    ]


def test_trial_runner_retry(trial_spec):
    registry = PluginRegistry()
    engine = MockInferenceEngine(fail_count=1)
    registry.register_inference_engine("openai", engine)
    registry.register_extractor("mock", MockExtractor())
    registry.register_metric("em", MockMetric())

    repo = MockEventRepo()
    runner = TrialRunner(registry, event_repo=repo, max_retries=2)

    trial_record = runner.run_trial(trial_spec, {}, RuntimeContext())

    assert trial_record.status == RecordStatus.OK
    assert engine.attempts == 3  # 1 fail + 2 successful parallel candidates

    events = [event.event_type for event in repo.events]
    assert "trial_retry" in events


@pytest.mark.parametrize(
    ("stage_name", "retryable_codes"),
    [
        ("transform", [ErrorCode.PARSE_ERROR.value]),
        ("evaluation", [ErrorCode.METRIC_COMPUTATION.value]),
    ],
)
def test_trial_runner_retries_overlay_stages_for_retryable_errors(
    trial_spec,
    stage_name: str,
    retryable_codes: list[str],
):
    registry = PluginRegistry()
    registry.register_inference_engine("openai", MockInferenceEngine())

    class FlakyExtractor(Extractor):
        def __init__(self) -> None:
            self.calls = 0

        def extract(self, trial, candidate, config=None):
            del trial, candidate, config
            self.calls += 1
            if stage_name == "transform" and self.calls == 1:
                raise ExtractionError(
                    code=ErrorCode.PARSE_ERROR,
                    message="try transform again",
                )
            return ExtractionRecord(
                spec_hash="ext_hash",
                extractor_id="mock",
                success=True,
                parsed_answer="42",
            )

    class FlakyMetric(Metric):
        def __init__(self) -> None:
            self.calls = 0

        def score(self, trial, candidate, context):
            del trial, candidate, context
            self.calls += 1
            if stage_name == "evaluation" and self.calls == 1:
                raise MetricError(
                    code=ErrorCode.METRIC_COMPUTATION,
                    message="try evaluation again",
                )
            return MetricScore(metric_id="em", value=1.0, details={"matched": True})

    extractor = FlakyExtractor()
    metric = FlakyMetric()
    registry.register_extractor("mock", extractor)
    registry.register_metric("em", metric)

    repo = MockEventRepo()
    runner = TrialRunner(
        registry,
        event_repo=repo,
        max_retries=2,
        retryable_error_codes=retryable_codes,
    )

    trial_record = runner.run_trial(
        trial_spec.model_copy(update={"candidate_count": 1}),
        {},
        RuntimeContext(),
    )

    assert trial_record.status == RecordStatus.OK
    assert extractor.calls == (2 if stage_name == "transform" else 1)
    assert metric.calls == (2 if stage_name == "evaluation" else 1)
    retry_events = [event for event in repo.events if event.event_type == "trial_retry"]
    assert len(retry_events) == 1
    assert retry_events[0].stage == (
        TimelineStage.EXTRACTION
        if stage_name == "transform"
        else TimelineStage.EVALUATION
    )


def test_trial_runner_rejects_unknown_retryable_error_codes() -> None:
    with pytest.raises(SpecValidationError) as exc_info:
        TrialRunner(
            PluginRegistry(),
            event_repo=MockEventRepo(),
            retryable_error_codes=[
                ErrorCode.PROVIDER_TIMEOUT.value,
                "not_a_real_error_code",
            ],
        )

    assert exc_info.value.code is ErrorCode.SCHEMA_MISMATCH
    assert "not_a_real_error_code" in exc_info.value.message


def test_trial_runner_replays_artifact_backed_stage_payloads_without_inline_json(
    trial_spec, tmp_path
):
    registry = PluginRegistry()
    registry.register_inference_engine("openai", MockInferenceEngine())
    registry.register_extractor("mock", MockExtractor())
    registry.register_metric("em", MockMetric())

    manager = DatabaseManager(f"sqlite:///{tmp_path}/artifact-backed-events.db")
    manager.initialize()
    artifact_store = ArtifactStore(tmp_path / "artifacts", manager=manager)
    event_repo = SqliteEventRepository(manager)
    projection_repo = SqliteProjectionRepository(manager, artifact_store=artifact_store)
    runner = TrialRunner(
        registry,
        event_repo=event_repo,
        artifact_store=artifact_store,
    )

    executed_trial = trial_spec.model_copy(update={"candidate_count": 1})
    runner.run_trial(
        executed_trial,
        {"question": "6 * 7", "answer": "42"},
        RuntimeContext(),
    )

    resolved = resolve_task_stages(executed_trial.task)
    record = projection_repo.materialize_trial_record(
        executed_trial.spec_hash,
        evaluation_hash=resolved.evaluations[0].evaluation_hash,
    )

    assert record.status == RecordStatus.OK
    assert record.candidates[0].evaluation is not None
    assert record.candidates[0].evaluation.aggregate_scores["em"] == 1.0

    with manager.get_connection() as conn:
        rows = conn.execute(
            """
            SELECT stage, payload_json, artifact_refs_json
            FROM trial_events
            WHERE trial_hash = ? AND stage IS NOT NULL
            ORDER BY event_seq ASC
            """,
            (executed_trial.spec_hash,),
        ).fetchall()

    by_stage = {
        row["stage"]: row
        for row in rows
        if row["stage"]
        in {
            TimelineStage.ITEM_LOAD,
            TimelineStage.INFERENCE,
            TimelineStage.EXTRACTION,
            TimelineStage.EVALUATION,
        }
    }
    assert by_stage[TimelineStage.ITEM_LOAD]["payload_json"] is None
    assert by_stage[TimelineStage.ITEM_LOAD]["artifact_refs_json"] is not None
    assert by_stage[TimelineStage.INFERENCE]["payload_json"] is None
    assert by_stage[TimelineStage.INFERENCE]["artifact_refs_json"] is not None
    assert by_stage[TimelineStage.EXTRACTION]["payload_json"] is None
    assert by_stage[TimelineStage.EXTRACTION]["artifact_refs_json"] is not None
    assert by_stage[TimelineStage.EVALUATION]["payload_json"] is None
    assert by_stage[TimelineStage.EVALUATION]["artifact_refs_json"] is not None


def test_trial_runner_delegates_generation_candidate_execution(trial_spec):
    registry = PluginRegistry()
    registry.register_inference_engine("openai", MockInferenceEngine())
    registry.register_extractor("mock", MockExtractor())
    registry.register_metric("em", MockMetric())
    repo = MockEventRepo()
    runner = TrialRunner(registry, event_repo=repo)
    session = runner.prepare_trial_session(trial_spec, {}, RuntimeContext())
    expected = CandidateRecord(
        spec_hash="candidate_hash",
        candidate_id="candidate_hash",
        sample_index=7,
        status=RecordStatus.OK,
    )

    class StubGenerationStage:
        def __init__(self) -> None:
            self.calls: list[tuple[object, int]] = []

        def run_candidate(self, passed_session, cand_index: int) -> CandidateRecord:
            self.calls.append((passed_session, cand_index))
            return expected

    stub = StubGenerationStage()
    runner.generation_stage = stub

    result = runner.run_generation_candidate(session, 7)

    assert result is expected
    assert stub.calls == [(session, 7)]


def test_trial_runner_delegates_overlay_stage_execution(trial_spec):
    registry = PluginRegistry()
    registry.register_inference_engine("openai", MockInferenceEngine())
    registry.register_extractor("mock", MockExtractor())
    registry.register_metric("em", MockMetric())
    repo = MockEventRepo()
    runner = TrialRunner(registry, event_repo=repo)
    session = runner.prepare_trial_session(trial_spec, {}, RuntimeContext())
    candidate = CandidateRecord(
        spec_hash="candidate_hash",
        candidate_id="candidate_hash",
        sample_index=0,
        status=RecordStatus.OK,
    )
    resolved = resolve_task_stages(trial_spec.task)
    transform = resolved.output_transforms[0]
    evaluation = resolved.evaluations[0]
    transformed = candidate.model_copy(update={"status": RecordStatus.OK})
    evaluated = candidate.model_copy(update={"status": RecordStatus.OK})

    class StubOverlayStage:
        def __init__(self) -> None:
            self.transform_calls: list[tuple[object, CandidateRecord, object]] = []
            self.evaluation_calls: list[tuple[object, CandidateRecord, object]] = []

        def run_output_transform(
            self,
            passed_session,
            passed_candidate: CandidateRecord,
            passed_transform,
        ) -> CandidateRecord:
            self.transform_calls.append(
                (passed_session, passed_candidate, passed_transform)
            )
            return transformed

        def run_evaluation_candidate(
            self,
            passed_session,
            passed_candidate: CandidateRecord,
            passed_evaluation,
        ) -> CandidateRecord:
            self.evaluation_calls.append(
                (passed_session, passed_candidate, passed_evaluation)
            )
            return evaluated

    stub = StubOverlayStage()
    runner.overlay_stage = stub

    transform_result = runner.run_output_transform(session, candidate, transform)
    evaluation_result = runner.run_evaluation_candidate(session, candidate, evaluation)

    assert transform_result is transformed
    assert evaluation_result is evaluated
    assert stub.transform_calls == [(session, candidate, transform)]
    assert stub.evaluation_calls == [(session, candidate, evaluation)]


def test_trial_runner_can_run_inside_running_event_loop(trial_spec):
    registry = PluginRegistry()
    registry.register_inference_engine("openai", MockInferenceEngine())
    registry.register_extractor("mock", MockExtractor())
    registry.register_metric("em", MockMetric())

    repo = MockEventRepo()
    runner = TrialRunner(registry, event_repo=repo, parallel_candidates=1)

    async def run_in_loop():
        return runner.run_trial(
            trial_spec.model_copy(update={"candidate_count": 1}),
            {},
            RuntimeContext(),
        )

    record = asyncio.run(run_in_loop())

    assert record.status == RecordStatus.OK
    assert len(record.candidates) == 1


def test_trial_runner_skips_completed_candidates_when_resuming(tmp_path, trial_spec):
    registry = PluginRegistry()
    engine = MockInferenceEngine()
    registry.register_inference_engine("openai", engine)
    registry.register_extractor("mock", MockExtractor())
    registry.register_metric("em", MockMetric())

    manager = DatabaseManager(f"sqlite:///{tmp_path}/resume_skip.db")
    manager.initialize()
    event_repo = SqliteEventRepository(manager)
    runner = TrialRunner(
        registry,
        event_repo=event_repo,
        parallel_candidates=1,
    )
    trial = trial_spec.model_copy(update={"candidate_count": 1})

    first_run = runner.run_trial(trial, {}, RuntimeContext())
    assert engine.attempts == 1

    with manager.get_connection() as conn:
        with conn:
            conn.execute(
                """
                DELETE FROM trial_events
                WHERE trial_hash = ? AND event_type = 'trial_completed'
                """,
                (trial.spec_hash,),
            )

    resumed_run = runner.run_trial(trial, {}, RuntimeContext())
    events = event_repo.get_events(trial.spec_hash)

    assert engine.attempts == 1
    assert (
        resumed_run.candidates[0].candidate_id == first_run.candidates[0].candidate_id
    )
    assert [event.event_type for event in events].count("inference_completed") == 1
    assert [event.event_type for event in events].count("candidate_completed") == 1
    assert [event.event_type for event in events].count("trial_completed") == 1


def test_trial_runner_emits_required_stage_metadata_and_artifact_refs(trial_spec):
    registry = PluginRegistry()
    registry.register_inference_engine("openai", MockInferenceEngine())
    registry.register_extractor("mock", MockExtractor())
    registry.register_metric("em", MockMetric())

    repo = MockEventRepo()
    runner = TrialRunner(registry, event_repo=repo, parallel_candidates=1)

    runner.run_trial(
        trial_spec.model_copy(
            update={
                "candidate_count": 1,
                "prompt": PromptTemplateSpec(id="baseline", messages=[]),
            }
        ),
        {"question": "6 * 7", "answer": "42"},
        RuntimeContext(environment={"suite": "tests"}),
    )

    item_event = next(
        event for event in repo.events if event.stage == TimelineStage.ITEM_LOAD
    )
    assert item_event.status == EventStatus.OK
    assert item_event.payload == {"question": "6 * 7", "answer": "42"}
    assert item_event.metadata.item_payload_hash is not None
    assert item_event.artifact_refs[0].role == "item_payload"
    assert isinstance(item_event.metadata, ItemLoadedEventMetadata)

    prompt_event = next(
        event
        for event in repo.events
        if event.stage == TimelineStage.PROMPT_RENDER and event.candidate_id is None
    )
    assert prompt_event.status == EventStatus.OK
    assert prompt_event.metadata.rendered_prompt_hash is not None
    assert prompt_event.metadata.input_field_map == ["answer", "question"]
    assert prompt_event.artifact_refs[0].role == "rendered_prompt"
    assert isinstance(prompt_event.metadata, PromptRenderedEventMetadata)

    inference_event = next(
        event for event in repo.events if event.stage == TimelineStage.INFERENCE
    )
    assert inference_event.status == EventStatus.OK
    assert inference_event.metadata.provider_request_id == "req_123"
    assert inference_event.metadata.token_usage["total_tokens"] == 13
    assert inference_event.artifact_refs[0].role == "inference_output"
    assert isinstance(inference_event.metadata, InferenceCompletedEventMetadata)

    extraction_event = next(
        event for event in repo.events if event.stage == TimelineStage.EXTRACTION
    )
    assert extraction_event.status == EventStatus.OK
    assert extraction_event.metadata.failure_reason is None

    evaluation_event = next(
        event for event in repo.events if event.stage == TimelineStage.EVALUATION
    )
    assert evaluation_event.status == EventStatus.OK
    assert evaluation_event.metadata.details_hash is not None
    assert evaluation_event.artifact_refs[0].role == "metric_details"
    assert isinstance(evaluation_event.metadata, EvaluationCompletedEventMetadata)


def test_trial_runner_emits_overlay_hashes_for_stage_events(trial_spec):
    registry = PluginRegistry()
    registry.register_inference_engine("openai", MockInferenceEngine())
    registry.register_extractor("mock", MockExtractor())
    registry.register_metric("em", MockMetric())

    repo = MockEventRepo()
    runner = TrialRunner(registry, event_repo=repo, parallel_candidates=1)
    resolved = resolve_task_stages(trial_spec.task)

    runner.run_trial(
        trial_spec.model_copy(update={"candidate_count": 1}),
        {},
        RuntimeContext(environment={"suite": "tests"}),
    )

    extraction_event = next(
        event for event in repo.events if event.stage == TimelineStage.EXTRACTION
    )
    evaluation_event = next(
        event for event in repo.events if event.stage == TimelineStage.EVALUATION
    )

    assert isinstance(extraction_event.metadata, ExtractionCompletedEventMetadata)
    assert (
        extraction_event.metadata.transform_hash
        == resolved.output_transforms[0].transform_hash
    )
    assert isinstance(evaluation_event.metadata, EvaluationCompletedEventMetadata)
    assert (
        evaluation_event.metadata.transform_hash
        == resolved.output_transforms[0].transform_hash
    )
    assert (
        evaluation_event.metadata.evaluation_hash
        == resolved.evaluations[0].evaluation_hash
    )


def test_trial_runner_emits_all_transform_and_evaluation_overlays():
    class ExtractorA(Extractor):
        def extract(self, trial, candidate, config=None):
            return ExtractionRecord(
                spec_hash="ext_a",
                extractor_id="extractor_a",
                success=True,
                parsed_answer="A",
            )

    class ExtractorB(Extractor):
        def extract(self, trial, candidate, config=None):
            return ExtractionRecord(
                spec_hash="ext_b",
                extractor_id="extractor_b",
                success=True,
                parsed_answer="B",
            )

    class MetricA(Metric):
        def score(self, trial, candidate, context):
            return MetricScore(metric_id="metric_a", value=1.0)

    class MetricB(Metric):
        def score(self, trial, candidate, context):
            return MetricScore(metric_id="metric_b", value=0.0)

    trial = TrialSpec(
        trial_id="multi_overlay_trial",
        model=ModelSpec(model_id="gpt-4", provider="openai"),
        task=TaskSpec(
            task_id="multi_overlay_task",
            dataset=DatasetSpec(source=DatasetSource.MEMORY),
            generation=GenerationSpec(),
            output_transforms=[
                OutputTransformSpec(
                    name="first",
                    extractor_chain=ExtractorChainSpec(extractors=["extractor_a"]),
                ),
                OutputTransformSpec(
                    name="second",
                    extractor_chain=ExtractorChainSpec(extractors=["extractor_b"]),
                ),
            ],
            evaluations=[
                EvaluationSpec(name="score_a", transform="first", metrics=["metric_a"]),
                EvaluationSpec(
                    name="score_b", transform="second", metrics=["metric_b"]
                ),
            ],
        ),
        item_id="item1",
        prompt=PromptTemplateSpec(id="baseline", messages=[]),
        params=InferenceParamsSpec(),
        candidate_count=1,
    )
    resolved = resolve_task_stages(trial.task)

    registry = PluginRegistry()
    registry.register_inference_engine("openai", MockInferenceEngine())
    registry.register_extractor("extractor_a", ExtractorA())
    registry.register_extractor("extractor_b", ExtractorB())
    registry.register_metric("metric_a", MetricA())
    registry.register_metric("metric_b", MetricB())

    repo = MockEventRepo()
    runner = TrialRunner(registry, event_repo=repo, parallel_candidates=1)

    runner.run_trial(trial, {}, RuntimeContext())

    extraction_events = [
        event for event in repo.events if event.stage == TimelineStage.EXTRACTION
    ]
    evaluation_events = [
        event for event in repo.events if event.stage == TimelineStage.EVALUATION
    ]

    assert len(extraction_events) == 2
    assert {event.metadata.transform_hash for event in extraction_events} == {
        transform.transform_hash for transform in resolved.output_transforms
    }
    assert len(evaluation_events) == 2
    assert {event.metadata.evaluation_hash for event in evaluation_events} == {
        evaluation.evaluation_hash for evaluation in resolved.evaluations
    }


def test_projection_repo_materializes_overlay_from_runner_events(tmp_path, trial_spec):
    registry = PluginRegistry()
    registry.register_inference_engine("openai", MockInferenceEngine())
    registry.register_extractor("mock", MockExtractor())
    registry.register_metric("em", MockMetric())

    manager = DatabaseManager(f"sqlite:///{tmp_path}/runner_projection.db")
    manager.initialize()
    event_repo = SqliteEventRepository(manager)
    projection_repo = SqliteProjectionRepository(manager)
    runner = TrialRunner(registry, event_repo=event_repo, parallel_candidates=1)
    trial = trial_spec.model_copy(update={"candidate_count": 1})
    resolved = resolve_task_stages(trial.task)
    transform_hash = resolved.output_transforms[0].transform_hash
    evaluation_hash = resolved.evaluations[0].evaluation_hash

    runner.run_trial(
        trial,
        {"question": "6 * 7", "answer": "42"},
        RuntimeContext(environment={"suite": "tests"}),
    )

    transform_record = projection_repo.materialize_trial_record(
        trial.spec_hash,
        transform_hash=transform_hash,
    )
    evaluation_record = projection_repo.materialize_trial_record(
        trial.spec_hash,
        transform_hash=transform_hash,
        evaluation_hash=evaluation_hash,
    )

    assert transform_record.candidates[0].extractions[0].parsed_answer == "42"
    assert evaluation_record.candidates[0].extractions[0].parsed_answer == "42"
    assert evaluation_record.candidates[0].evaluation is not None
    assert evaluation_record.candidates[0].evaluation.aggregate_scores["em"] == 1.0


def test_trial_runner_persists_emitted_artifacts_in_blob_store_and_index(
    tmp_path, trial_spec
):
    registry = PluginRegistry()
    registry.register_inference_engine("openai", MockInferenceEngine())
    registry.register_extractor("mock", MockExtractor())
    registry.register_metric("em", MockMetric())

    manager = DatabaseManager(f"sqlite:///{tmp_path}/artifacts.db")
    manager.initialize()
    artifact_store = ArtifactStore(base_path=tmp_path / "artifacts", manager=manager)
    event_repo = SqliteEventRepository(manager)
    runner = TrialRunner(
        registry,
        event_repo=event_repo,
        artifact_store=artifact_store,
        parallel_candidates=1,
    )

    executed_trial = trial_spec.model_copy(
        update={
            "candidate_count": 1,
            "prompt": PromptTemplateSpec(id="baseline", messages=[]),
        }
    )
    runner.run_trial(
        executed_trial,
        {"question": "6 * 7", "answer": "42"},
        RuntimeContext(environment={"suite": "tests"}),
    )

    events = event_repo.get_events(executed_trial.spec_hash)
    artifact_hashes = [
        artifact.artifact_hash for event in events for artifact in event.artifact_refs
    ]
    assert artifact_hashes
    with manager.get_connection() as conn:
        rows = conn.execute(
            "SELECT artifact_hash FROM artifacts ORDER BY artifact_hash ASC"
        ).fetchall()
    persisted_hashes = [row["artifact_hash"] for row in rows]
    assert set(artifact_hashes).issubset(set(persisted_hashes))


def test_trial_runner_persists_judge_audit_artifacts_and_projection_hydrates_them(
    tmp_path,
):
    registry = PluginRegistry()
    registry.register_inference_engine("openai", MockInferenceEngine())
    registry.register_inference_engine("judge_provider", MockJudgeInferenceEngine())
    registry.register_metric("judge_metric", JudgeAwareMetric())

    manager = DatabaseManager(f"sqlite:///{tmp_path}/judge_audit.db")
    manager.initialize()
    artifact_store = ArtifactStore(base_path=tmp_path / "artifacts")
    event_repo = SqliteEventRepository(manager)
    projection_repo = SqliteProjectionRepository(manager, artifact_store=artifact_store)
    runner = TrialRunner(
        registry,
        event_repo=event_repo,
        artifact_store=artifact_store,
        parallel_candidates=1,
    )

    trial = TrialSpec(
        trial_id="judge_trial",
        model=ModelSpec(model_id="gpt-4", provider="openai"),
        task=TaskSpec(
            task_id="judge_task",
            dataset=DatasetSpec(source=DatasetSource.MEMORY),
            generation=GenerationSpec(),
            evaluations=[EvaluationSpec(name="judge", metrics=["judge_metric"])],
        ),
        item_id="item1",
        prompt=PromptTemplateSpec(id="baseline", messages=[]),
        params=InferenceParamsSpec(),
        candidate_count=1,
    )

    result = runner.run_trial(trial, {"question": "6 * 7"}, RuntimeContext())
    resolved = resolve_task_stages(trial.task)
    evaluation_hash = resolved.evaluations[0].evaluation_hash
    projection_repo.materialize_trial_record(
        trial.spec_hash,
        evaluation_hash=evaluation_hash,
    )

    evaluation_event = next(
        event
        for event in event_repo.get_events(trial.spec_hash)
        if event.stage == TimelineStage.EVALUATION
    )
    judge_artifact = next(
        artifact
        for artifact in evaluation_event.artifact_refs
        if artifact.role == "judge_audit"
    )
    assert artifact_store.exists(judge_artifact.artifact_hash)

    candidate_id = result.candidates[0].candidate_id
    timeline_view = projection_repo.get_timeline_view(
        candidate_id,
        "candidate",
        evaluation_hash=evaluation_hash,
    )
    assert timeline_view is not None
    assert timeline_view.judge_audit is not None
    assert timeline_view.judge_audit.candidate_hash == candidate_id
    assert timeline_view.judge_audit.judge_calls[0].metric_id == "judge_metric"
    assert timeline_view.judge_audit.judge_calls[0].rendered_prompt == [
        PromptMessage(role=PromptRole.USER, content="Rate this answer.")
    ]
    assert timeline_view.judge_audit.judge_calls[0].inference.raw_text == "SCORE: 5/5"


def test_trial_runner_maps_provider_failures_with_context() -> None:
    class ExplodingInferenceEngine(InferenceEngine):
        def infer(self, trial, context: DatasetContext, runtime: RuntimeContext):
            try:
                raise TimeoutError("socket timed out")
            except TimeoutError as cause:
                raise InferenceError(
                    code=ErrorCode.PROVIDER_TIMEOUT,
                    message="provider timeout",
                    details={"endpoint": "responses"},
                ) from cause

    registry = PluginRegistry()
    registry.register_inference_engine("openai", ExplodingInferenceEngine())
    registry.register_metric("em", MockMetric())

    repo = MockEventRepo()
    runner = TrialRunner(registry, event_repo=repo, parallel_candidates=1)
    trial = TrialSpec(
        trial_id="failing_trial",
        model=ModelSpec(model_id="gpt-4", provider="openai"),
        task=TaskSpec(
            task_id="t1",
            dataset=DatasetSpec(source=DatasetSource.MEMORY),
            generation=GenerationSpec(),
            evaluations=[EvaluationSpec(name="score", metrics=["em"])],
        ),
        item_id="item1",
        prompt=PromptTemplateSpec(messages=[]),
        params=InferenceParamsSpec(),
        candidate_count=1,
    )

    record = runner.run_trial(trial, {}, RuntimeContext())

    assert record.status == RecordStatus.ERROR
    assert record.error is not None
    assert record.error.details["provider"] == "openai"
    assert record.error.details["model_id"] == "gpt-4"
    assert record.error.details["candidate_id"] == record.candidates[0].candidate_id
    assert record.error.details["attempt"] == 1
    assert record.error.cause_chain[0].message == "TimeoutError: socket timed out"


def test_trial_runner_rejects_non_json_safe_dataset_payloads(trial_spec):
    registry = PluginRegistry()
    registry.register_inference_engine("openai", MockInferenceEngine())
    registry.register_extractor("mock", MockExtractor())
    registry.register_metric("em", MockMetric())

    repo = MockEventRepo()
    runner = TrialRunner(registry, event_repo=repo, parallel_candidates=1)

    with pytest.raises(SpecValidationError, match="dataset payload"):
        runner.run_trial(
            trial_spec.model_copy(update={"candidate_count": 1}),
            {"question": object()},
            RuntimeContext(),
        )
