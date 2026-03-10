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
from themis.errors.exceptions import InferenceError, SpecValidationError
from themis.orchestration.trial_runner import TrialRunner
from themis.records.conversation import Conversation, MessageEvent, MessagePayload
from themis.records.evaluation import MetricScore
from themis.records.extraction import ExtractionRecord
from themis.records.inference import InferenceRecord, TokenUsage
from themis.registry.plugin_registry import PluginRegistry
from themis.specs.foundational import (
    DatasetSpec,
    ExtractorChainSpec,
    JudgeInferenceSpec,
    ModelSpec,
    TaskSpec,
)
from themis.specs.experiment import (
    InferenceParamsSpec,
    PromptMessage,
    PromptTemplateSpec,
    TrialSpec,
)
from themis.storage.artifact_store import ArtifactStore
from themis.storage.event_repo import SqliteEventRepository
from themis.storage.events import TrialEvent, TrialEventType
from themis.types.enums import ErrorCode, RecordStatus as EventStatus
from themis.storage.sqlite_schema import DatabaseManager
from themis.storage.projection_repo import SqliteProjectionRepository
from themis.types.enums import RecordStatus


class MockEventRepo(TrialEventRepository):
    def __init__(self):
        self.events = []
        self.saved_specs = []

    def save_spec(self, spec) -> None:
        self.saved_specs.append(spec)

    def append_event(self, event: TrialEvent):
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

    def has_projection_for_revision(self, trial_hash, eval_revision):
        return any(
            event.trial_hash == trial_hash
            and event.event_type == TrialEventType.PROJECTION_COMPLETED
            and event.metadata.get("eval_revision") == eval_revision
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
                        role="assistant",
                        payload=MessagePayload(content="The answer is 42."),
                        event_index=0,
                    )
                ]
            ),
        )


class MockExtractor(Extractor):
    def extract(self, trial, candidate):
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
            dataset=DatasetSpec(source="memory"),
            default_extractor_chain=ExtractorChainSpec(extractors=["mock"]),
            default_metrics=["em"],
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

    item_event = next(event for event in repo.events if event.stage == "item_load")
    assert item_event.status == EventStatus.OK
    assert item_event.payload == {"question": "6 * 7", "answer": "42"}
    assert "item_payload_hash" in item_event.metadata
    assert item_event.artifact_refs[0].role == "item_payload"

    prompt_event = next(
        event
        for event in repo.events
        if event.stage == "prompt_render" and event.candidate_id is None
    )
    assert prompt_event.status == EventStatus.OK
    assert "rendered_prompt_hash" in prompt_event.metadata
    assert prompt_event.metadata["input_field_map"] == ["answer", "question"]
    assert prompt_event.artifact_refs[0].role == "rendered_prompt"

    inference_event = next(event for event in repo.events if event.stage == "inference")
    assert inference_event.status == EventStatus.OK
    assert inference_event.metadata["provider_request_id"] == "req_123"
    assert inference_event.metadata["token_usage"]["total_tokens"] == 13
    assert inference_event.artifact_refs[0].role == "inference_output"

    extraction_event = next(
        event for event in repo.events if event.stage == "extraction"
    )
    assert extraction_event.status == EventStatus.OK
    assert extraction_event.metadata["failure_reason"] is None

    evaluation_event = next(
        event for event in repo.events if event.stage == "evaluation"
    )
    assert evaluation_event.status == EventStatus.OK
    assert evaluation_event.metadata["details_hash"] is not None
    assert evaluation_event.artifact_refs[0].role == "metric_details"


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
            dataset=DatasetSpec(source="memory"),
            default_metrics=["judge_metric"],
        ),
        item_id="item1",
        prompt=PromptTemplateSpec(id="baseline", messages=[]),
        params=InferenceParamsSpec(),
        candidate_count=1,
    )

    result = runner.run_trial(trial, {"question": "6 * 7"}, RuntimeContext())
    projection_repo.materialize_trial_record(trial.spec_hash, "latest")

    evaluation_event = next(
        event
        for event in event_repo.get_events(trial.spec_hash)
        if event.stage == "evaluation"
    )
    judge_artifact = next(
        artifact
        for artifact in evaluation_event.artifact_refs
        if artifact.role == "judge_audit"
    )
    assert artifact_store.exists(judge_artifact.artifact_hash)

    candidate_id = result.candidates[0].candidate_id
    timeline_view = projection_repo.get_timeline_view(
        candidate_id, "candidate", "latest"
    )
    assert timeline_view is not None
    assert timeline_view.judge_audit is not None
    assert timeline_view.judge_audit.candidate_hash == candidate_id
    assert timeline_view.judge_audit.judge_calls[0].metric_id == "judge_metric"
    assert timeline_view.judge_audit.judge_calls[0].rendered_prompt == [
        PromptMessage(role="user", content="Rate this answer.")
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
            task_id="t1", dataset=DatasetSpec(source="memory"), default_metrics=["em"]
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
