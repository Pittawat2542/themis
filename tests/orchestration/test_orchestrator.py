import textwrap

import pytest

from themis.contracts.protocols import InferenceResult
from themis.errors.exceptions import SpecValidationError
from themis.orchestration.orchestrator import Orchestrator
from themis.registry.plugin_registry import PluginRegistry
from themis.records.conversation import (
    Conversation,
    MessageEvent,
    MessagePayload,
    ToolCallEvent,
    ToolCallPayload,
    ToolResultEvent,
    ToolResultPayload,
)
from themis.records.extraction import ExtractionRecord
from themis.records.evaluation import MetricScore
from themis.records.inference import InferenceRecord, TokenUsage
from themis.runtime import ExperimentResult
from themis.storage.sqlite_schema import DatabaseManager
from themis.specs.experiment import (
    ExecutionPolicySpec,
    ExperimentSpec,
    InferenceGridSpec,
    InferenceParamsSpec,
    ProjectSpec,
    PromptTemplateSpec,
    RuntimeContext,
    StorageSpec,
)
from themis.specs.foundational import (
    DatasetSpec,
    ExtractorChainSpec,
    ModelSpec,
    TaskSpec,
)
from themis.telemetry.bus import TelemetryBus


class MockDatasetLoader:
    def load_task_items(self, task):
        return [
            {"item_id": "item1", "question": "6 * 7"},
            {"item_id": "item2", "question": "8 * 8"},
        ]


class DummyEngine:
    pass


class DummyMetric:
    pass


class RunnableEngine:
    def __init__(self) -> None:
        self.seen_questions = []
        self.seen_seeds = []

    def infer(self, trial, context, runtime):
        self.seen_questions.append((trial.item_id, context["question"]))
        self.seen_seeds.append(runtime.candidate_seed)
        return InferenceResult(
            inference=InferenceRecord(
                spec_hash=f"inference_{trial.item_id}",
                raw_text="42",
                latency_ms=12,
                provider_request_id=f"req_{trial.item_id}",
                token_usage=TokenUsage(
                    prompt_tokens=5, completion_tokens=2, total_tokens=7
                ),
            )
        )


class RunnableMetric:
    def score(self, trial, candidate, context):
        return MetricScore(metric_id="em", value=1.0, details={"matched": True})


class ConversationalEngine(RunnableEngine):
    def infer(self, trial, context, runtime):
        self.seen_questions.append((trial.item_id, context["question"]))
        self.seen_seeds.append(runtime.candidate_seed)
        return InferenceResult(
            inference=InferenceRecord(
                spec_hash=f"inference_{trial.item_id}",
                raw_text="42",
                latency_ms=12,
                provider_request_id=f"req_{trial.item_id}",
                token_usage=TokenUsage(
                    prompt_tokens=5, completion_tokens=2, total_tokens=7
                ),
            ),
            conversation=Conversation(
                events=[
                    MessageEvent(
                        role="assistant",
                        event_index=1,
                        payload=MessagePayload(content="Working..."),
                    ),
                    ToolCallEvent(
                        role="assistant",
                        event_index=2,
                        payload=ToolCallPayload(
                            tool_name="calc",
                            tool_arguments={"expression": "6*7"},
                            call_id="call-1",
                        ),
                    ),
                    ToolResultEvent(
                        role="tool",
                        event_index=3,
                        payload=ToolResultPayload(call_id="call-1", result="42"),
                    ),
                ]
            ),
        )


class RunnableExtractor:
    def extract(self, trial, candidate, config=None):
        return ExtractionRecord(
            spec_hash="extract_1", extractor_id="mock", success=True, parsed_answer="42"
        )


def test_orchestrator_initialization():
    registry = PluginRegistry()
    manager = DatabaseManager("sqlite:///:memory:")
    manager.initialize()

    orchestrator = Orchestrator(registry, manager, dataset_loader=MockDatasetLoader())
    assert orchestrator.planner is not None
    assert orchestrator.executor is not None


def test_orchestrator_run_experiment():
    registry = PluginRegistry()
    registry.register_inference_engine("p", DummyEngine())
    registry.register_metric("em", DummyMetric())
    manager = DatabaseManager("sqlite:///:memory:")
    manager.initialize()

    orchestrator = Orchestrator(registry, manager, dataset_loader=MockDatasetLoader())

    # Mocking out the inner executor loop so we don't need real inference plugins
    executed_trials = []

    def mock_execute(trials, runtime, **kwargs):
        executed_trials.extend(trials)

    orchestrator.executor.execute_trials = mock_execute

    experiment = ExperimentSpec(
        models=[ModelSpec(model_id="m", provider="p")],
        tasks=[
            TaskSpec(
                task_id="t",
                dataset=DatasetSpec(source="memory"),
                default_metrics=["em"],
            )
        ],
        prompt_templates=[PromptTemplateSpec(messages=[])],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec()]),
        num_samples=1,
    )

    result = orchestrator.run(experiment, runtime=RuntimeContext())
    assert len(executed_trials) == 2  # 2 items from mock loader
    assert isinstance(result, ExperimentResult)
    assert sorted(result.trial_hashes) == sorted(
        planned_trial.trial_spec.spec_hash for planned_trial in executed_trials
    )


def test_orchestrator_run_executes_real_trial_and_materializes_projection():
    registry = PluginRegistry()
    engine = RunnableEngine()
    registry.register_inference_engine("p", engine)
    registry.register_metric("em", RunnableMetric())
    manager = DatabaseManager("sqlite:///:memory:")
    manager.initialize()

    orchestrator = Orchestrator(registry, manager, dataset_loader=MockDatasetLoader())
    experiment = ExperimentSpec(
        models=[ModelSpec(model_id="m", provider="p")],
        tasks=[
            TaskSpec(
                task_id="t",
                dataset=DatasetSpec(source="memory"),
                default_metrics=["em"],
            )
        ],
        prompt_templates=[PromptTemplateSpec(id="baseline", messages=[])],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec()]),
        num_samples=1,
    )

    result = orchestrator.run(
        experiment, runtime=RuntimeContext(environment={"suite": "tests"})
    )

    assert isinstance(result, ExperimentResult)
    assert len(result.trial_hashes) == 2
    assert sorted(engine.seen_questions) == [("item1", "6 * 7"), ("item2", "8 * 8")]

    trial = result.get_trial(result.trial_hashes[0])
    assert trial is not None
    assert trial.trial_spec is not None
    assert trial.candidates[0].evaluation is not None
    assert trial.candidates[0].evaluation.aggregate_scores["em"] == 1.0

    timeline_payloads = {
        result.get_trial(trial_hash).trial_spec.item_id: result.view_timeline(
            trial_hash,
            record_type="trial",
        ).item_payload
        for trial_hash in result.trial_hashes
    }
    assert timeline_payloads == {
        "item1": {"item_id": "item1", "question": "6 * 7"},
        "item2": {"item_id": "item2", "question": "8 * 8"},
    }


def test_orchestrator_from_project_spec_uses_project_storage_and_policy(tmp_path):
    registry = PluginRegistry()
    registry.register_inference_engine("p", RunnableEngine())
    registry.register_metric("em", RunnableMetric())

    project = ProjectSpec(
        project_name="lab-project",
        researcher_id="researcher-1",
        global_seed=7,
        storage=StorageSpec(root_dir=str(tmp_path / "runs")),
        execution_policy=ExecutionPolicySpec(
            max_retries=7, circuit_breaker_threshold=3
        ),
    )

    orchestrator = Orchestrator.from_project_spec(
        project,
        registry=registry,
        dataset_loader=MockDatasetLoader(),
    )

    assert orchestrator.execution_policy.max_retries == 7
    assert orchestrator.execution_policy.circuit_breaker_threshold == 3
    assert orchestrator.db_manager.db_path == str(tmp_path / "runs" / "themis.sqlite3")


def test_orchestrator_rejects_non_positive_parallel_candidates(tmp_path):
    registry = PluginRegistry()
    registry.register_inference_engine("p", RunnableEngine())
    registry.register_metric("em", RunnableMetric())

    project = ProjectSpec(
        project_name="lab-project",
        researcher_id="researcher-1",
        global_seed=7,
        storage=StorageSpec(root_dir=str(tmp_path / "runs")),
        execution_policy=ExecutionPolicySpec(),
    )

    with pytest.raises(SpecValidationError, match="parallel_candidates"):
        Orchestrator.from_project_spec(
            project,
            registry=registry,
            dataset_loader=MockDatasetLoader(),
            parallel_candidates=0,
        )

    manager = DatabaseManager("sqlite:///:memory:")
    manager.initialize()

    with pytest.raises(SpecValidationError, match="parallel_candidates"):
        Orchestrator(
            registry,
            manager,
            dataset_loader=MockDatasetLoader(),
            parallel_candidates=-1,
        )


def test_orchestrator_no_longer_accepts_parallel_trials(tmp_path):
    registry = PluginRegistry()
    registry.register_inference_engine("p", RunnableEngine())
    registry.register_metric("em", RunnableMetric())

    project = ProjectSpec(
        project_name="lab-project",
        researcher_id="researcher-1",
        global_seed=7,
        storage=StorageSpec(root_dir=str(tmp_path / "runs")),
        execution_policy=ExecutionPolicySpec(),
    )

    with pytest.raises(TypeError):
        Orchestrator.from_project_spec(
            project,
            registry=registry,
            dataset_loader=MockDatasetLoader(),
            parallel_trials=2,
        )

    manager = DatabaseManager("sqlite:///:memory:")
    manager.initialize()

    with pytest.raises(TypeError):
        Orchestrator(
            registry,
            manager,
            dataset_loader=MockDatasetLoader(),
            parallel_trials=2,
        )


def test_orchestrator_respects_store_item_payloads_false(tmp_path):
    registry = PluginRegistry()
    registry.register_inference_engine("p", RunnableEngine())
    registry.register_metric("em", RunnableMetric())

    project = ProjectSpec(
        project_name="lab-project",
        researcher_id="researcher-1",
        global_seed=7,
        storage=StorageSpec(root_dir=str(tmp_path / "runs"), store_item_payloads=False),
        execution_policy=ExecutionPolicySpec(),
    )

    orchestrator = Orchestrator.from_project_spec(
        project,
        registry=registry,
        dataset_loader=MockDatasetLoader(),
    )
    experiment = ExperimentSpec(
        models=[ModelSpec(model_id="m", provider="p")],
        tasks=[
            TaskSpec(
                task_id="t",
                dataset=DatasetSpec(source="memory"),
                default_metrics=["em"],
            )
        ],
        prompt_templates=[PromptTemplateSpec(id="baseline", messages=[])],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec()]),
        num_samples=1,
    )

    result = orchestrator.run(
        experiment, runtime=RuntimeContext(environment={"suite": "tests"})
    )
    view = result.view_timeline(result.trial_hashes[0], record_type="trial")

    assert view is not None
    assert view.item_payload is None


def test_orchestrator_from_project_spec_derives_deterministic_candidate_seeds(tmp_path):
    registry = PluginRegistry()
    engine = RunnableEngine()
    registry.register_inference_engine("p", engine)
    registry.register_metric("em", RunnableMetric())

    project = ProjectSpec(
        project_name="lab-project",
        researcher_id="researcher-1",
        global_seed=11,
        storage=StorageSpec(root_dir=str(tmp_path / "runs")),
        execution_policy=ExecutionPolicySpec(),
    )

    orchestrator = Orchestrator.from_project_spec(
        project,
        registry=registry,
        dataset_loader=MockDatasetLoader(),
    )
    experiment = ExperimentSpec(
        models=[ModelSpec(model_id="m", provider="p")],
        tasks=[
            TaskSpec(
                task_id="t",
                dataset=DatasetSpec(source="memory"),
                default_metrics=["em"],
            )
        ],
        prompt_templates=[PromptTemplateSpec(id="baseline", messages=[])],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec()]),
        num_samples=1,
    )

    orchestrator.run(experiment, runtime=RuntimeContext(environment={"suite": "tests"}))
    first_run_seeds = list(engine.seen_seeds)

    second_engine = RunnableEngine()
    second_registry = PluginRegistry()
    second_registry.register_inference_engine("p", second_engine)
    second_registry.register_metric("em", RunnableMetric())
    second_project = project.model_copy(
        update={"storage": StorageSpec(root_dir=str(tmp_path / "runs-second"))}
    )
    second_orchestrator = Orchestrator.from_project_spec(
        second_project,
        registry=second_registry,
        dataset_loader=MockDatasetLoader(),
    )
    second_orchestrator.run(
        experiment, runtime=RuntimeContext(environment={"suite": "tests"})
    )

    assert first_run_seeds == second_engine.seen_seeds
    assert len(first_run_seeds) == 2
    assert all(isinstance(seed, int) for seed in first_run_seeds)


def test_orchestrator_from_project_file_loads_toml(tmp_path):
    project_file = tmp_path / "project.toml"
    project_file.write_text(
        textwrap.dedent(
            """
            project_name = "lab-project"
            researcher_id = "researcher-1"
            global_seed = 7

            [storage]
            backend = "sqlite_blob"
            root_dir = "__ROOT__"
            store_item_payloads = true
            compression = "zstd"

            [execution_policy]
            max_retries = 4
            retry_backoff_factor = 2.0
            circuit_breaker_threshold = 6
            """
        ).replace("__ROOT__", str(tmp_path / "runs"))
    )

    orchestrator = Orchestrator.from_project_file(
        str(project_file),
        registry=PluginRegistry(),
        dataset_loader=MockDatasetLoader(),
    )

    assert orchestrator.execution_policy.max_retries == 4
    assert orchestrator.execution_policy.retry_backoff_factor == 2.0
    assert orchestrator.db_manager.db_path == str(tmp_path / "runs" / "themis.sqlite3")


def test_orchestrator_from_project_file_wraps_validation_errors(tmp_path):
    project_file = tmp_path / "project.json"
    project_file.write_text(
        '{"storage": {"root_dir": "/tmp/runs"}, "execution_policy": {}}'
    )

    with pytest.raises(SpecValidationError, match="project.json"):
        Orchestrator.from_project_file(str(project_file))


def test_orchestrator_from_project_file_wraps_json_decode_errors(tmp_path):
    project_file = tmp_path / "project.json"
    project_file.write_text("{not-json}")

    with pytest.raises(SpecValidationError, match="project.json"):
        Orchestrator.from_project_file(str(project_file))


def test_orchestrator_emits_required_telemetry_events_and_exposes_observability_refs(
    tmp_path,
):
    registry = PluginRegistry()
    registry.register_inference_engine("p", ConversationalEngine())
    registry.register_extractor("mock", RunnableExtractor())
    registry.register_metric("em", RunnableMetric())
    manager = DatabaseManager(f"sqlite:///{tmp_path}/telemetry.db")
    manager.initialize()
    bus = TelemetryBus()
    seen = []
    bus.subscribe(seen.append)

    class MockTrace:
        def __init__(self):
            self.id = "trace-1"

    class MockClient:
        def trace(self, **kwargs):
            return MockTrace()

        def span(self, **kwargs):
            return None

    orchestrator = Orchestrator(
        registry,
        manager,
        dataset_loader=MockDatasetLoader(),
        telemetry_bus=bus,
    )
    from themis.telemetry.langfuse_callback import LangfuseCallback

    LangfuseCallback(
        client=MockClient(),
        observability_store=orchestrator.observability_store,
        base_url="https://langfuse.example",
    ).subscribe(bus)

    experiment = ExperimentSpec(
        models=[ModelSpec(model_id="m", provider="p")],
        tasks=[
            TaskSpec(
                task_id="t",
                dataset=DatasetSpec(source="memory"),
                default_extractor_chain=ExtractorChainSpec(extractors=["mock"]),
                default_metrics=["em"],
            )
        ],
        prompt_templates=[PromptTemplateSpec(id="baseline", messages=[])],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec()]),
        num_samples=1,
    )

    result = orchestrator.run(
        experiment, runtime=RuntimeContext(environment={"suite": "tests"})
    )
    trial = result.get_trial(result.trial_hashes[0])
    candidate_id = trial.candidates[0].spec_hash
    candidate_view = result.view_timeline(candidate_id)

    assert {
        "trial_start",
        "trial_end",
        "conversation_event",
        "tool_call",
        "tool_result",
        "extractor_attempt",
        "metric_start",
        "metric_end",
    }.issubset({event.name for event in seen})
    assert candidate_view is not None
    assert candidate_view.observability is not None
    assert candidate_view.observability.langfuse_trace_id == "trace-1"
