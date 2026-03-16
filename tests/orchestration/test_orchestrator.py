from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
import tempfile
import typing

import pytest

from themis.contracts.protocols import InferenceResult
from themis.errors import SpecValidationError
from themis.orchestration.orchestrator import Orchestrator
from themis.orchestration.run_manifest import CostEstimate, RunHandle
from themis.orchestration.task_resolution import resolve_task_stages
from themis.progress import ProgressConfig, ProgressRendererType
from themis.records.candidate import CandidateRecord
from themis.records.evaluation import MetricScore
from themis.records.evaluation import EvaluationRecord
from themis.records.error import ErrorRecord
from themis.records.extraction import ExtractionRecord
from themis.records.inference import InferenceRecord, TokenUsage
from themis.records.trial import TrialRecord
from themis.registry.plugin_registry import PluginRegistry
from themis.runtime import ExperimentResult
from themis.specs.experiment import (
    BatchExecutionBackendSpec,
    DataItemContext,
    ExecutionPolicySpec,
    ExperimentSpec,
    InferenceGridSpec,
    InferenceParamsSpec,
    LocalExecutionBackendSpec,
    ProjectSpec,
    PromptTemplateSpec,
    RuntimeContext,
    StorageSpec,
    TrialSpec,
    WorkerPoolExecutionBackendSpec,
)
from themis.specs.foundational import (
    DatasetSpec,
    EvaluationSpec,
    ExtractorChainSpec,
    ExtractorRefSpec,
    GenerationSpec,
    ModelSpec,
    OutputTransformSpec,
    TaskSpec,
)
from themis.storage.factory import build_storage_bundle
from themis.storage.event_repo import SqliteEventRepository
from themis.orchestration.projection_handler import ProjectionHandler
from themis.storage.sqlite_schema import DatabaseManager
from themis.storage.projection_repo import SqliteProjectionRepository
from themis.storage._protocols import StorageConnectionManager
from themis.types.enums import (
    ErrorCode,
    ErrorWhere,
    RecordStatus,
    DatasetSource,
    RunStage,
)
from themis.types.events import (
    TrialEvent,
    TrialEventType,
    TimelineStage,
    TrialEventMetadata,
)
from typing import cast, Any


class MockDatasetLoader:
    def load_task_items(self, task):
        del task
        return [
            {"item_id": "item1", "question": "6 * 7"},
            {"item_id": "item2", "question": "8 * 8"},
        ]


class RunnableEngine:
    def __init__(self) -> None:
        self.seen_questions: list[tuple[str, str]] = []
        self.seen_seeds: list[int | None] = []

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
                    prompt_tokens=5,
                    completion_tokens=2,
                    total_tokens=7,
                ),
            )
        )


class RunnableExtractor:
    def extract(self, trial, candidate, config=None):
        del trial, config
        raw_text = (
            candidate.inference.raw_text if candidate.inference is not None else ""
        )
        return ExtractionRecord(
            spec_hash="extract_1",
            extractor_id="mock-extractor",
            success=True,
            parsed_answer=raw_text,
        )


class RunnableMetric:
    def score(self, trial, candidate, context):
        del trial, context
        extraction = candidate.best_extraction()
        answer = extraction.parsed_answer if extraction is not None else None
        return MetricScore(metric_id="em", value=1.0 if answer == "42" else 0.0)


class SingleItemDatasetLoader:
    def load_task_items(self, task):
        del task
        return [{"item_id": "item1", "question": "6 * 7"}]


class FailingExtractor:
    def __init__(self) -> None:
        self.calls: int = 0

    def extract(self, trial, candidate, config=None):
        del trial, candidate, config
        self.calls += 1
        raise RuntimeError("bad extraction")


class ImportedExtractor:
    def extract(self, trial, candidate, config=None):
        del trial, config
        raw_text = (
            candidate.inference.raw_text if candidate.inference is not None else ""
        )
        return ExtractionRecord(
            spec_hash="extract_imported",
            extractor_id="imported-extractor",
            success=True,
            parsed_answer=raw_text,
        )


class ImportedMetric:
    def score(self, trial, candidate, context):
        del trial, context
        raw_text = (
            candidate.inference.raw_text if candidate.inference is not None else ""
        )
        return MetricScore(metric_id="em", value=1.0 if raw_text == "42" else 0.0)


def _build_registry() -> tuple[PluginRegistry, RunnableEngine]:
    registry = PluginRegistry()
    engine = RunnableEngine()
    registry.register_inference_engine("mock", engine)
    registry.register_extractor("mock-extractor", RunnableExtractor())
    registry.register_metric("em", RunnableMetric())
    return registry, engine


def _build_experiment() -> ExperimentSpec:
    return ExperimentSpec(
        models=[ModelSpec(model_id="mock-model", provider="mock")],
        tasks=[
            TaskSpec(
                task_id="math",
                dataset=DatasetSpec(source=DatasetSource.MEMORY),
                generation=GenerationSpec(),
                output_transforms=[
                    OutputTransformSpec(
                        name="answer",
                        extractor_chain=ExtractorChainSpec(
                            extractors=[ExtractorRefSpec(id="mock-extractor")]
                        ),
                    )
                ],
                evaluations=[
                    EvaluationSpec(name="default", transform="answer", metrics=["em"])
                ],
            )
        ],
        prompt_templates=[PromptTemplateSpec(id="baseline", messages=[])],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec()]),
        num_samples=2,
    )


def _build_orchestrator() -> tuple[Orchestrator, RunnableEngine]:
    registry, engine = _build_registry()
    project = _build_project_spec(Path(tempfile.mkdtemp()))
    return (
        Orchestrator.from_project_spec(
            project,
            registry=registry,
            dataset_loader=MockDatasetLoader(),
        ),
        engine,
    )


def _build_project_spec(tmp_path: Path) -> ProjectSpec:
    return ProjectSpec(
        project_name="lab-project",
        researcher_id="researcher-1",
        global_seed=7,
        storage=StorageSpec(root_dir=str(tmp_path / "runs")),
        execution_policy=ExecutionPolicySpec(
            max_retries=3,
            circuit_breaker_threshold=3,
            max_in_flight_work_items=9,
        ),
    )


def _build_batch_project_spec(tmp_path: Path) -> ProjectSpec:
    return _build_project_spec(tmp_path).model_copy(
        update={
            "execution_backend": BatchExecutionBackendSpec(
                provider="openai",
                poll_interval_seconds=30,
                max_batch_items=250,
            )
        }
    )


def _planned_trial_id(
    *,
    model: ModelSpec,
    task: TaskSpec,
    prompt: PromptTemplateSpec,
    params: InferenceParamsSpec,
    item: DataItemContext,
) -> str:
    item_str = json.dumps(
        {
            "item_id": item.item_id,
            "payload": item.payload,
            "metadata": item.metadata,
        },
        sort_keys=True,
    )
    composite = "".join(
        [
            model.spec_hash,
            task.spec_hash,
            prompt.spec_hash,
            params.spec_hash,
            item_str,
        ]
    ).encode("utf-8")
    return f"trial_{hashlib.sha256(composite).hexdigest()[:12]}"


def test_orchestrator_from_project_spec_uses_project_storage_and_execution_policy(
    tmp_path,
) -> None:
    registry, _engine = _build_registry()
    project = _build_project_spec(tmp_path).model_copy(
        update={
            "execution_policy": ExecutionPolicySpec(
                max_retries=7,
                circuit_breaker_threshold=3,
                max_in_flight_work_items=9,
            )
        }
    )

    orchestrator = Orchestrator.from_project_spec(
        project,
        registry=registry,
        dataset_loader=MockDatasetLoader(),
    )

    assert orchestrator.execution_policy.max_retries == 7
    assert orchestrator.execution_policy.circuit_breaker_threshold == 3
    from themis.storage.sqlite_schema import DatabaseManager

    assert typing.cast(DatabaseManager, orchestrator.db_manager).db_path == str(
        tmp_path / "runs" / "themis.sqlite3"
    )
    assert isinstance(project.execution_backend, LocalExecutionBackendSpec)


def test_orchestrator_from_project_spec_accepts_prebuilt_storage_bundle(
    tmp_path,
) -> None:
    registry, _engine = _build_registry()
    project = _build_project_spec(tmp_path)
    storage_bundle = build_storage_bundle(project.storage)

    orchestrator = Orchestrator.from_project_spec(
        project,
        registry=registry,
        dataset_loader=MockDatasetLoader(),
        storage_bundle=storage_bundle,
    )

    assert orchestrator.db_manager is storage_bundle.manager


def test_orchestrator_stores_runtime_services_in_private_bundle() -> None:
    orchestrator, _engine = _build_orchestrator()

    assert "_services" in orchestrator.__dict__
    for attr_name in (
        "db_manager",
        "event_repo",
        "observability_store",
        "projection_repo",
        "planner",
        "runner",
        "projection_handler",
        "executor",
    ):
        assert attr_name not in orchestrator.__dict__


@pytest.mark.parametrize(
    "attr_name",
    [
        "event_repo",
        "observability_store",
        "projection_repo",
        "planner",
        "runner",
        "projection_handler",
        "executor",
    ],
)
def test_orchestrator_internal_service_attrs_are_removed(attr_name: str) -> None:
    orchestrator, _engine = _build_orchestrator()

    with pytest.raises(AttributeError, match=attr_name):
        getattr(orchestrator, attr_name)


def test_orchestrator_rejects_direct_runtime_construction() -> None:
    registry, _engine = _build_registry()
    manager = DatabaseManager(
        f"sqlite:///{Path(tempfile.mkdtemp()) / 'themis.sqlite3'}"
    )
    manager.initialize()

    with pytest.raises(
        TypeError,
        match="Orchestrator.from_project_spec",
    ):
        Orchestrator(
            cast(Any, registry),
            manager,  # type: ignore[arg-type]  # Intentional invalid argument type to test init validation
            dataset_loader=cast(Any, MockDatasetLoader()),
        )


def test_orchestrator_rejects_incomplete_internal_construction() -> None:
    with pytest.raises(
        TypeError,
        match="Internal orchestrator construction requires registry",
    ):
        Orchestrator(_allow_runtime_construction=True)


def test_orchestrator_from_project_file_loads_toml(tmp_path) -> None:
    registry, _engine = _build_registry()
    project_path = tmp_path / "project.toml"
    project_path.write_text(
        """
project_name = "lab-project"
researcher_id = "researcher-1"
global_seed = 7

[storage]
backend = "sqlite_blob"
root_dir = "runs"
store_item_payloads = true
compression = "zstd"

[execution_policy]
max_retries = 3
retry_backoff_factor = 1.5
circuit_breaker_threshold = 3
max_in_flight_work_items = 9
retryable_error_codes = []
""".strip()
    )

    orchestrator = Orchestrator.from_project_file(
        str(project_path),
        registry=registry,
        dataset_loader=MockDatasetLoader(),
    )

    assert orchestrator.project_spec is not None
    assert orchestrator.project_spec.project_name == "lab-project"
    assert orchestrator.execution_policy.max_retries == 3


def test_orchestrator_from_project_file_loads_json(tmp_path) -> None:
    registry, _engine = _build_registry()
    project = _build_project_spec(tmp_path)
    project_path = tmp_path / "project.json"
    project_path.write_text(json.dumps(project.model_dump(mode="json")))

    orchestrator = Orchestrator.from_project_file(
        str(project_path),
        registry=registry,
        dataset_loader=MockDatasetLoader(),
    )

    assert orchestrator.project_spec == project
    assert typing.cast(DatabaseManager, orchestrator.db_manager).db_path == str(
        tmp_path / "runs" / "themis.sqlite3"
    )


def test_orchestrator_from_project_file_rejects_unsupported_suffix(tmp_path) -> None:
    project_path = tmp_path / "project.yaml"
    project_path.write_text("project_name: lab-project\n")

    with pytest.raises(ValueError, match=r"\.toml or \.json"):
        Orchestrator.from_project_file(str(project_path))


def test_orchestrator_from_project_file_wraps_toml_parse_errors(tmp_path) -> None:
    project_path = tmp_path / "project.toml"
    project_path.write_text("project_name = 'lab-project'\n[storage\n")

    with pytest.raises(SpecValidationError) as exc_info:
        Orchestrator.from_project_file(str(project_path))

    assert exc_info.value.code is ErrorCode.SCHEMA_MISMATCH
    assert "Failed to parse project config project.toml" in exc_info.value.message


def test_orchestrator_from_project_file_wraps_validation_errors(tmp_path) -> None:
    project_path = tmp_path / "project.json"
    project_path.write_text(json.dumps({"project_name": "lab-project"}))

    with pytest.raises(SpecValidationError) as exc_info:
        Orchestrator.from_project_file(str(project_path))

    assert exc_info.value.code is ErrorCode.SCHEMA_MISMATCH
    assert "Failed to parse project config project.json" in exc_info.value.message


def test_orchestrator_removes_legacy_planning_and_storage_helpers(tmp_path) -> None:
    registry, _engine = _build_registry()
    project = _build_project_spec(tmp_path)
    orchestrator = Orchestrator.from_project_spec(
        project,
        registry=registry,
        dataset_loader=MockDatasetLoader(),
    )

    for attr_name in (
        "_build_manifest",
        "_result_from_manifest",
        "_run_handle_from_manifest",
        "_validate_generation_import",
        "_validate_evaluation_import",
        "_coerce_storage_bundle",
        "_legacy_event_repo",
        "_legacy_projection_repo",
    ):
        assert not hasattr(orchestrator, attr_name)


def test_generate_returns_generation_scoped_result() -> None:
    orchestrator, engine = _build_orchestrator()

    result = orchestrator.generate(_build_experiment(), runtime=RuntimeContext())

    assert isinstance(result, ExperimentResult)
    assert result.transform_hashes == []
    assert result.evaluation_hashes == []
    assert len(result.trial_hashes) == 2
    assert sorted(engine.seen_questions) == [
        ("item1", "6 * 7"),
        ("item1", "6 * 7"),
        ("item2", "8 * 8"),
        ("item2", "8 * 8"),
    ]

    trial = result.get_trial(result.trial_hashes[0])
    assert trial is not None
    assert trial.candidates[0].best_extraction() is None
    assert trial.candidates[0].evaluation is None


def test_transform_returns_transform_scoped_result() -> None:
    orchestrator, _engine = _build_orchestrator()
    experiment = _build_experiment()
    orchestrator.generate(experiment, runtime=RuntimeContext())

    result = orchestrator.transform(experiment, runtime=RuntimeContext())

    assert len(result.transform_hashes) == 1
    assert result.evaluation_hashes == []
    trial = result.get_trial(result.trial_hashes[0])
    assert trial is not None
    extraction = trial.candidates[0].best_extraction()
    assert extraction is not None
    assert extraction.parsed_answer == "42"
    assert trial.candidates[0].evaluation is None


def test_evaluate_materializes_required_transforms_if_missing() -> None:
    orchestrator, _engine = _build_orchestrator()
    experiment = _build_experiment()
    orchestrator.generate(experiment, runtime=RuntimeContext())

    result = orchestrator.evaluate(experiment, runtime=RuntimeContext())

    assert len(result.transform_hashes) == 1
    assert len(result.evaluation_hashes) == 1
    trial = result.get_trial(result.trial_hashes[0])
    assert trial is not None
    assert trial.candidates[0].evaluation is not None
    assert trial.candidates[0].evaluation.aggregate_scores["em"] == 1.0

    transform_view = result.for_transform(result.transform_hashes[0])
    transform_trial = transform_view.get_trial(result.trial_hashes[0])
    assert transform_trial is not None
    extraction = transform_trial.candidates[0].best_extraction()
    assert extraction is not None


def test_run_executes_generation_then_transforms_then_evaluations() -> None:
    orchestrator, engine = _build_orchestrator()

    result = orchestrator.run(_build_experiment(), runtime=RuntimeContext())

    assert isinstance(result, ExperimentResult)
    assert len(result.transform_hashes) == 1
    assert len(result.evaluation_hashes) == 1
    assert sorted(engine.seen_questions) == [
        ("item1", "6 * 7"),
        ("item1", "6 * 7"),
        ("item2", "8 * 8"),
        ("item2", "8 * 8"),
    ]

    trial = result.get_trial(result.trial_hashes[0])
    assert trial is not None
    assert trial.candidates[0].evaluation is not None
    assert trial.candidates[0].evaluation.aggregate_scores["em"] == 1.0


def test_generate_skips_transform_and_metric_validation() -> None:
    class DummyInferenceEngine:
        def infer(self, trial, context, runtime):
            del trial, context, runtime
            return InferenceResult(
                inference=InferenceRecord(spec_hash="inf_hash", raw_text="42")
            )

    registry = PluginRegistry()
    registry.register_inference_engine("openai", DummyInferenceEngine())
    orchestrator = Orchestrator.from_project_spec(
        _build_project_spec(Path(tempfile.mkdtemp())),
        registry=registry,
        dataset_loader=SingleItemDatasetLoader(),
    )
    experiment = ExperimentSpec(
        models=[ModelSpec(model_id="mock-model", provider="openai")],
        tasks=[
            TaskSpec(
                task_id="math",
                dataset=DatasetSpec(source=DatasetSource.MEMORY),
                generation=GenerationSpec(),
                output_transforms=[
                    OutputTransformSpec(
                        name="missing",
                        extractor_chain=ExtractorChainSpec(
                            extractors=[ExtractorRefSpec(id="missing-extractor")]
                        ),
                    )
                ],
                evaluations=[
                    EvaluationSpec(
                        name="missing",
                        transform="missing",
                        metrics=["missing-metric"],
                    )
                ],
            )
        ],
        prompt_templates=[PromptTemplateSpec(id="baseline", messages=[])],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec()]),
        num_samples=1,
    )

    result = orchestrator.generate(experiment, runtime=RuntimeContext())

    assert isinstance(result, ExperimentResult)
    assert len(result.trial_hashes) == 1


def test_run_still_validates_all_declared_stages() -> None:
    class DummyInferenceEngine:
        def infer(self, trial, context, runtime):
            raise NotImplementedError

    registry = PluginRegistry()
    registry.register_inference_engine("openai", DummyInferenceEngine())
    orchestrator = Orchestrator.from_project_spec(
        _build_project_spec(Path(tempfile.mkdtemp())),
        registry=registry,
        dataset_loader=SingleItemDatasetLoader(),
    )
    experiment = ExperimentSpec(
        models=[ModelSpec(model_id="mock-model", provider="openai")],
        tasks=[
            TaskSpec(
                task_id="math",
                dataset=DatasetSpec(source=DatasetSource.MEMORY),
                generation=GenerationSpec(),
                output_transforms=[
                    OutputTransformSpec(
                        name="missing",
                        extractor_chain=ExtractorChainSpec(
                            extractors=[ExtractorRefSpec(id="missing-extractor")]
                        ),
                    )
                ],
            )
        ],
        prompt_templates=[PromptTemplateSpec(id="baseline", messages=[])],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec()]),
        num_samples=1,
    )

    with pytest.raises(Exception, match="missing-extractor"):
        orchestrator.run(experiment, runtime=RuntimeContext())


def test_transform_can_run_without_inference_engine_for_transform_only_task() -> None:
    registry = PluginRegistry()
    registry.register_extractor("imported-extractor", ImportedExtractor())
    orchestrator = Orchestrator.from_project_spec(
        _build_project_spec(Path(tempfile.mkdtemp())),
        registry=registry,
        dataset_loader=SingleItemDatasetLoader(),
    )

    model = ModelSpec(model_id="imported-model", provider="unregistered")
    task = TaskSpec(
        task_id="math",
        dataset=DatasetSpec(source=DatasetSource.MEMORY),
        output_transforms=[
            OutputTransformSpec(
                name="answer",
                extractor_chain=ExtractorChainSpec(
                    extractors=[ExtractorRefSpec(id="imported-extractor")]
                ),
            )
        ],
    )
    prompt = PromptTemplateSpec(id="baseline", messages=[])
    params = InferenceParamsSpec()
    item = DataItemContext(
        item_id="item1",
        payload={"item_id": "item1", "question": "6 * 7"},
    )
    trial_spec = TrialSpec(
        trial_id=_planned_trial_id(
            model=model,
            task=task,
            prompt=prompt,
            params=params,
            item=item,
        ),
        model=model,
        task=task,
        item_id=item.item_id,
        prompt=prompt,
        params=params,
        candidate_count=1,
    )
    import_record = TrialRecord(
        spec_hash=trial_spec.spec_hash,
        trial_spec=trial_spec,
        status=RecordStatus.OK,
        candidates=[
            CandidateRecord(
                spec_hash="cand_imported",
                candidate_id="cand_imported",
                sample_index=0,
                status=RecordStatus.OK,
                inference=InferenceRecord(spec_hash="inf_hash", raw_text="42"),
            )
        ],
    )
    orchestrator.import_candidates([import_record])

    experiment = ExperimentSpec(
        models=[model],
        tasks=[task],
        prompt_templates=[prompt],
        inference_grid=InferenceGridSpec(params=[params]),
        num_samples=1,
    )

    result = orchestrator.transform(experiment, runtime=RuntimeContext())
    trial = result.get_trial(trial_spec.spec_hash)

    assert trial is not None
    extraction = trial.candidates[0].best_extraction()
    assert extraction is not None
    assert extraction.parsed_answer == "42"


def test_evaluate_can_run_without_inference_engine_for_evaluation_only_task() -> None:
    registry = PluginRegistry()
    registry.register_metric("em", ImportedMetric())
    orchestrator = Orchestrator.from_project_spec(
        _build_project_spec(Path(tempfile.mkdtemp())),
        registry=registry,
        dataset_loader=SingleItemDatasetLoader(),
    )

    model = ModelSpec(model_id="imported-model", provider="unregistered")
    task = TaskSpec(
        task_id="math",
        dataset=DatasetSpec(source=DatasetSource.MEMORY),
        evaluations=[EvaluationSpec(name="score", metrics=["em"])],
    )
    prompt = PromptTemplateSpec(id="baseline", messages=[])
    params = InferenceParamsSpec()
    item = DataItemContext(
        item_id="item1",
        payload={"item_id": "item1", "question": "6 * 7"},
    )
    trial_spec = TrialSpec(
        trial_id=_planned_trial_id(
            model=model,
            task=task,
            prompt=prompt,
            params=params,
            item=item,
        ),
        model=model,
        task=task,
        item_id=item.item_id,
        prompt=prompt,
        params=params,
        candidate_count=1,
    )
    import_record = TrialRecord(
        spec_hash=trial_spec.spec_hash,
        trial_spec=trial_spec,
        status=RecordStatus.OK,
        candidates=[
            CandidateRecord(
                spec_hash="cand_imported",
                candidate_id="cand_imported",
                sample_index=0,
                status=RecordStatus.OK,
                inference=InferenceRecord(spec_hash="inf_hash", raw_text="42"),
            )
        ],
    )
    orchestrator.import_candidates([import_record])

    experiment = ExperimentSpec(
        models=[model],
        tasks=[task],
        prompt_templates=[prompt],
        inference_grid=InferenceGridSpec(params=[params]),
        num_samples=1,
    )

    result = orchestrator.evaluate(experiment, runtime=RuntimeContext())
    trial = result.get_trial(trial_spec.spec_hash)

    assert trial is not None
    assert trial.candidates[0].evaluation is not None
    assert trial.candidates[0].evaluation.aggregate_scores["em"] == 1.0


def test_generate_can_run_inside_running_event_loop() -> None:
    orchestrator, _engine = _build_orchestrator()

    async def run_in_loop() -> ExperimentResult:
        return orchestrator.generate(_build_experiment(), runtime=RuntimeContext())

    result = asyncio.run(run_in_loop())

    assert isinstance(result, ExperimentResult)


def test_orchestrator_plan_returns_manifest_with_stage_work_items(tmp_path) -> None:
    registry, _engine = _build_registry()
    project = _build_project_spec(tmp_path)
    orchestrator = Orchestrator.from_project_spec(
        project,
        registry=registry,
        dataset_loader=MockDatasetLoader(),
    )

    manifest = orchestrator.plan(_build_experiment())

    assert manifest.project_spec == project
    assert manifest.experiment_spec == _build_experiment()
    assert manifest.backend_kind == "local"
    assert len(manifest.trial_hashes) == 2
    assert len(manifest.work_items) == 12
    assert {item.stage for item in manifest.work_items} == {
        "generation",
        "transform",
        "evaluation",
    }
    assert {item.status for item in manifest.work_items} == {"pending"}
    assert orchestrator.plan(_build_experiment()).run_id == manifest.run_id


@pytest.mark.parametrize("failed_stage", ["transform", "evaluation"])
def test_orchestrator_plan_marks_failed_overlay_work_items_failed(
    tmp_path,
    failed_stage: str,
) -> None:
    registry, _engine = _build_registry()
    project = _build_project_spec(tmp_path)
    orchestrator = Orchestrator.from_project_spec(
        project,
        registry=registry,
        dataset_loader=SingleItemDatasetLoader(),
    )
    experiment = _build_experiment().model_copy(update={"num_samples": 1})
    bundle = orchestrator.export_generation_bundle(experiment)
    trial_spec = bundle.items[0].trial_spec
    resolved = resolve_task_stages(trial_spec.task)
    transform_hash = resolved.output_transforms[0].transform_hash
    evaluation_hash = resolved.evaluations[0].evaluation_hash
    event_repo = SqliteEventRepository(
        cast(StorageConnectionManager, orchestrator.db_manager)
    )
    projection_repo = SqliteProjectionRepository(
        cast(StorageConnectionManager, orchestrator.db_manager)
    )
    projection_handler = ProjectionHandler(
        cast(Any, event_repo), cast(Any, projection_repo)
    )
    candidate_id = bundle.items[0].candidate_id

    event_repo.save_spec(trial_spec)
    seed_events = [
        TrialEvent(
            trial_hash=trial_spec.spec_hash,
            event_seq=1,
            event_id="evt_1",
            event_type=TrialEventType.CANDIDATE_STARTED,
            candidate_id=candidate_id,
            payload={"sample_index": 0},
        ),
        TrialEvent(
            trial_hash=trial_spec.spec_hash,
            event_seq=2,
            event_id="evt_2",
            event_type=TrialEventType.CANDIDATE_COMPLETED,
            candidate_id=candidate_id,
            payload={"status": "ok"},
        ),
        TrialEvent(
            trial_hash=trial_spec.spec_hash,
            event_seq=3,
            event_id="evt_3",
            event_type=(
                TrialEventType.EXTRACTION_COMPLETED
                if failed_stage == "transform"
                else TrialEventType.EVALUATION_COMPLETED
            ),
            candidate_id=candidate_id,
            stage=TimelineStage.EXTRACTION
            if failed_stage == "transform"
            else TimelineStage.EVALUATION,
            status=RecordStatus.ERROR,
            metadata=cast(
                TrialEventMetadata,
                {
                    "transform_hash": transform_hash
                    if failed_stage == "transform"
                    else None,
                    "evaluation_hash": (
                        evaluation_hash
                        if failed_stage == TimelineStage.EVALUATION
                        else None
                    ),
                    "success": False if failed_stage == "transform" else None,
                },
            ),
            payload={
                "spec_hash": "extract_failed"
                if failed_stage == "transform"
                else "eval_failed",
                "extractor_id": "mock-extractor",
                "success": False,
                "failure_reason": "no match",
            }
            if failed_stage == "transform"
            else {
                "spec_hash": "eval_failed",
                "metric_scores": [{"metric_id": "em", "value": 0.0}],
            },
        ),
        TrialEvent(
            trial_hash=trial_spec.spec_hash,
            event_seq=4,
            event_id="evt_4",
            event_type=TrialEventType.CANDIDATE_FAILED,
            candidate_id=candidate_id,
            stage=TimelineStage.EXTRACTION
            if failed_stage == "transform"
            else TimelineStage.EVALUATION,
            status=RecordStatus.ERROR,
            metadata=cast(
                TrialEventMetadata,
                {
                    "transform_hash": transform_hash
                    if failed_stage == "transform"
                    else None,
                    "evaluation_hash": (
                        evaluation_hash
                        if failed_stage == TimelineStage.EVALUATION
                        else None
                    ),
                },
            ),
            error=ErrorRecord(
                where=ErrorWhere.EXTRACTOR
                if failed_stage == "transform"
                else ErrorWhere.METRIC,
                code=ErrorCode.PARSE_ERROR
                if failed_stage == "transform"
                else ErrorCode.METRIC_COMPUTATION,
                message="overlay failed",
                retryable=False,
                details={},
            ),
        ),
        TrialEvent(
            trial_hash=trial_spec.spec_hash,
            event_seq=5,
            event_id="evt_5",
            event_type=TrialEventType.TRIAL_COMPLETED,
            payload={"status": "ok"},
        ),
    ]
    for event in seed_events:
        event_repo.append_event(event)
    projection_handler.on_trial_completed(
        trial_spec.spec_hash,
        transform_hash=transform_hash if failed_stage == "transform" else None,
        evaluation_hash=evaluation_hash
        if failed_stage == TimelineStage.EVALUATION
        else None,
    )

    manifest = orchestrator.plan(experiment)
    statuses: dict[str, str] = {
        item.stage: item.status
        for item in manifest.work_items
        if item.trial_hash == trial_spec.spec_hash and item.candidate_id == candidate_id
    }

    assert statuses["generation"] == "completed"
    assert statuses[failed_stage] == "failed"


def test_orchestrator_plan_marks_failed_generation_work_items_failed(tmp_path) -> None:
    registry, _engine = _build_registry()
    project = _build_project_spec(tmp_path)
    orchestrator = Orchestrator.from_project_spec(
        project,
        registry=registry,
        dataset_loader=SingleItemDatasetLoader(),
    )
    experiment = _build_experiment().model_copy(update={"num_samples": 1})
    bundle = orchestrator.export_generation_bundle(experiment)
    trial_spec = bundle.items[0].trial_spec
    candidate_id = bundle.items[0].candidate_id
    event_repo = SqliteEventRepository(
        cast(StorageConnectionManager, orchestrator.db_manager)
    )

    event_repo.save_spec(trial_spec)
    for event in [
        TrialEvent(
            trial_hash=trial_spec.spec_hash,
            event_seq=1,
            event_id="evt_1",
            event_type=TrialEventType.CANDIDATE_STARTED,
            candidate_id=candidate_id,
            payload={"sample_index": 0},
        ),
        TrialEvent(
            trial_hash=trial_spec.spec_hash,
            event_seq=2,
            event_id="evt_2",
            event_type=TrialEventType.CANDIDATE_FAILED,
            candidate_id=candidate_id,
            stage=TimelineStage.INFERENCE,
            status=RecordStatus.ERROR,
            error=ErrorRecord(
                where=ErrorWhere.INFERENCE,
                code=ErrorCode.PROVIDER_TIMEOUT,
                message="provider timeout",
                retryable=True,
                details={},
            ),
        ),
        TrialEvent(
            trial_hash=trial_spec.spec_hash,
            event_seq=3,
            event_id="evt_3",
            event_type=TrialEventType.TRIAL_COMPLETED,
            payload={"status": "error"},
        ),
    ]:
        event_repo.append_event(event)

    manifest = orchestrator.plan(experiment)
    statuses: dict[str, str] = {
        item.stage: item.status
        for item in manifest.work_items
        if item.trial_hash == trial_spec.spec_hash and item.candidate_id == candidate_id
    }

    assert statuses["generation"] == "failed"


def test_orchestrator_diff_specs_reports_changed_trials_and_stage_hashes(
    tmp_path,
) -> None:
    registry, _engine = _build_registry()
    project = _build_project_spec(tmp_path)
    orchestrator = Orchestrator.from_project_spec(
        project,
        registry=registry,
        dataset_loader=MockDatasetLoader(),
    )
    baseline = _build_experiment()
    expanded = baseline.model_copy(
        update={
            "models": [
                *baseline.models,
                ModelSpec(model_id="mock-model-2", provider="mock"),
            ]
        }
    )

    diff = orchestrator.diff_specs(baseline, expanded)

    assert "models" in diff.changed_experiment_fields
    assert diff.project_hash_before == project.spec_hash
    assert diff.project_hash_after == project.spec_hash
    assert diff.added_trial_hashes
    assert diff.removed_trial_hashes == []
    assert diff.added_evaluation_hashes == []


def test_orchestrator_can_export_generation_bundle_and_import_results(tmp_path) -> None:
    registry, _engine = _build_registry()
    project = _build_project_spec(tmp_path)
    orchestrator = Orchestrator.from_project_spec(
        project,
        registry=registry,
        dataset_loader=MockDatasetLoader(),
    )
    experiment = _build_experiment()

    bundle = orchestrator.export_generation_bundle(experiment)

    assert bundle.stage == "generation"
    assert len(bundle.items) == 4

    trial_specs = {item.trial_hash: item.trial_spec for item in bundle.items}
    records: list[TrialRecord] = []
    for trial_hash, trial_spec in trial_specs.items():
        assert trial_spec is not None
        candidate_items = [
            item for item in bundle.items if item.trial_hash == trial_hash
        ]
        records.append(
            TrialRecord(
                spec_hash=trial_hash,
                trial_spec=trial_spec,
                status=RecordStatus.OK,
                candidates=[
                    CandidateRecord(
                        spec_hash=item.candidate_id,
                        candidate_id=item.candidate_id,
                        sample_index=item.candidate_index,
                        status=RecordStatus.OK,
                        inference=InferenceRecord(
                            spec_hash=f"inference_{item.candidate_id}",
                            raw_text="42",
                        ),
                    )
                    for item in candidate_items
                ],
            )
        )

    result = orchestrator.import_generation_results(bundle, records)

    assert isinstance(result, ExperimentResult)
    assert all(result.get_trial(trial_hash) is not None for trial_hash in trial_specs)
    assert orchestrator.export_generation_bundle(experiment).items == []


def test_orchestrator_can_export_evaluation_bundle_and_import_results(tmp_path) -> None:
    registry, _engine = _build_registry()
    project = _build_project_spec(tmp_path)
    orchestrator = Orchestrator.from_project_spec(
        project,
        registry=registry,
        dataset_loader=MockDatasetLoader(),
    )
    experiment = _build_experiment()
    generation_bundle = orchestrator.export_generation_bundle(experiment)
    generation_trial_specs = {
        item.trial_hash: item.trial_spec for item in generation_bundle.items
    }
    generation_records: list[TrialRecord] = []
    for trial_hash, trial_spec in generation_trial_specs.items():
        assert trial_spec is not None
        candidate_items = [
            item for item in generation_bundle.items if item.trial_hash == trial_hash
        ]
        generation_records.append(
            TrialRecord(
                spec_hash=trial_hash,
                trial_spec=trial_spec,
                status=RecordStatus.OK,
                candidates=[
                    CandidateRecord(
                        spec_hash=item.candidate_id,
                        candidate_id=item.candidate_id,
                        sample_index=item.candidate_index,
                        status=RecordStatus.OK,
                        inference=InferenceRecord(
                            spec_hash=f"inference_{item.candidate_id}",
                            raw_text="42",
                        ),
                    )
                    for item in candidate_items
                ],
            )
        )
    orchestrator.import_generation_results(generation_bundle, generation_records)

    evaluation_bundle = orchestrator.export_evaluation_bundle(experiment)

    assert evaluation_bundle.stage == TimelineStage.EVALUATION
    assert len(evaluation_bundle.items) == 4
    evaluation_hashes = {item.evaluation_hash for item in evaluation_bundle.items}
    assert len(evaluation_hashes) == 1

    evaluation_trial_specs = {
        item.trial_hash: item.trial_spec for item in evaluation_bundle.items
    }
    evaluation_records: list[TrialRecord] = []
    for trial_hash, trial_spec in evaluation_trial_specs.items():
        assert trial_spec is not None
        eval_candidate_items = [
            item for item in evaluation_bundle.items if item.trial_hash == trial_hash
        ]
        evaluation_records.append(
            TrialRecord(
                spec_hash=trial_hash,
                trial_spec=trial_spec,
                status=RecordStatus.OK,
                candidates=[
                    CandidateRecord(
                        spec_hash=item.candidate_id,
                        candidate_id=item.candidate_id,
                        sample_index=item.candidate_index,
                        status=RecordStatus.OK,
                        evaluation=EvaluationRecord(
                            spec_hash=f"evaluation_{item.candidate_id}",
                            metric_scores=[
                                MetricScore(metric_id="em", value=1.0),
                            ],
                        ),
                    )
                    for item in eval_candidate_items
                ],
            )
        )

    result = orchestrator.import_evaluation_results(
        evaluation_bundle,
        evaluation_records,
    )

    evaluation_hash = next(iter(evaluation_hashes))
    trial = result.for_evaluation(evaluation_hash).get_trial(
        evaluation_bundle.items[0].trial_hash
    )
    assert trial is not None
    assert trial.candidates[0].evaluation is not None
    assert trial.candidates[0].evaluation.aggregate_scores["em"] == 1.0
    assert orchestrator.export_evaluation_bundle(experiment).items == []


def test_orchestrator_submit_executes_local_backend_and_resume_returns_result(
    tmp_path,
) -> None:
    registry, _engine = _build_registry()
    project = _build_project_spec(tmp_path)
    orchestrator = Orchestrator.from_project_spec(
        project,
        registry=registry,
        dataset_loader=MockDatasetLoader(),
    )

    handle = orchestrator.submit(_build_experiment(), runtime=RuntimeContext())

    assert isinstance(handle, RunHandle)
    assert handle.backend_kind == "local"
    assert handle.status == "completed"
    assert handle.pending_work_items == 0
    assert handle.completed_work_items == handle.total_work_items

    resumed = orchestrator.resume(handle.run_id)

    assert isinstance(resumed, ExperimentResult)
    assert resumed.trial_hashes == handle.trial_hashes


def test_orchestrator_submit_preserves_pending_handles_for_batch_backend(
    tmp_path,
) -> None:
    registry, _engine = _build_registry()
    project = _build_batch_project_spec(tmp_path)
    orchestrator = Orchestrator.from_project_spec(
        project,
        registry=registry,
        dataset_loader=MockDatasetLoader(),
    )

    handle = orchestrator.submit(_build_experiment(), runtime=RuntimeContext())

    assert isinstance(handle, RunHandle)
    assert handle.backend_kind == "batch"
    assert handle.status == "pending"
    assert handle.pending_work_items == handle.total_work_items
    assert handle.completed_work_items == 0

    resumed = orchestrator.resume(handle.run_id)

    assert isinstance(resumed, RunHandle)
    assert resumed.run_id == handle.run_id
    assert resumed.status == "pending"


def test_orchestrator_get_run_progress_reports_pending_batch_work(tmp_path) -> None:
    registry, _engine = _build_registry()
    project = _build_batch_project_spec(tmp_path)
    orchestrator = Orchestrator.from_project_spec(
        project,
        registry=registry,
        dataset_loader=MockDatasetLoader(),
    )

    handle = orchestrator.submit(_build_experiment(), runtime=RuntimeContext())
    snapshot = orchestrator.get_run_progress(handle.run_id)

    assert snapshot is not None
    assert snapshot.run_id == handle.run_id
    assert snapshot.active_stage == RunStage.GENERATION
    assert snapshot.processed_items == 0
    assert snapshot.remaining_items == handle.total_work_items
    assert snapshot.stage_counts[RunStage.GENERATION].pending_items == 4
    assert snapshot.stage_counts[RunStage.TRANSFORM].pending_items == 4
    assert snapshot.stage_counts[RunStage.EVALUATION].pending_items == 4


def test_orchestrator_run_emits_progress_snapshots_to_callback(tmp_path) -> None:
    registry, _engine = _build_registry()
    project = _build_project_spec(tmp_path)
    orchestrator = Orchestrator.from_project_spec(
        project,
        registry=registry,
        dataset_loader=MockDatasetLoader(),
    )
    snapshots = []

    result = orchestrator.run(
        _build_experiment(),
        runtime=RuntimeContext(),
        progress=ProgressConfig(
            renderer=ProgressRendererType.LOG,
            callback=snapshots.append,
        ),
    )

    assert isinstance(result, ExperimentResult)
    assert snapshots
    assert snapshots[0].remaining_items == 12
    assert snapshots[-1].remaining_items == 0
    assert snapshots[-1].processed_items == 12

    persisted = orchestrator.get_run_progress(snapshots[-1].run_id)
    assert persisted is not None
    assert persisted.remaining_items == 0


def test_orchestrator_submit_preserves_terminal_progress_snapshot(tmp_path) -> None:
    registry, _engine = _build_registry()
    project = _build_project_spec(tmp_path)
    orchestrator = Orchestrator.from_project_spec(
        project,
        registry=registry,
        dataset_loader=MockDatasetLoader(),
    )
    snapshots = []

    handle = orchestrator.submit(
        _build_experiment(),
        runtime=RuntimeContext(),
        progress=ProgressConfig(callback=snapshots.append),
    )

    persisted = orchestrator.get_run_progress(handle.run_id)

    assert persisted is not None
    assert persisted.remaining_items == 0
    assert persisted.processed_items == snapshots[-1].processed_items
    assert persisted.ended_at == snapshots[-1].ended_at


def test_orchestrator_submit_preserves_failed_progress_snapshot(tmp_path) -> None:
    registry = PluginRegistry()
    registry.register_inference_engine("mock", RunnableEngine())
    registry.register_extractor("mock-extractor", FailingExtractor())
    registry.register_metric("em", RunnableMetric())
    project = _build_project_spec(tmp_path)
    orchestrator = Orchestrator.from_project_spec(
        project,
        registry=registry,
        dataset_loader=SingleItemDatasetLoader(),
    )
    snapshots = []

    handle = orchestrator.submit(
        _build_experiment().model_copy(update={"num_samples": 1}),
        runtime=RuntimeContext(),
        progress=ProgressConfig(callback=snapshots.append),
    )

    persisted = orchestrator.get_run_progress(handle.run_id)

    assert persisted is not None
    assert persisted.stage_counts[RunStage.TRANSFORM].failed_items == 1
    assert persisted.ended_at == snapshots[-1].ended_at


@pytest.mark.parametrize("entrypoint_name", ["generate", "transform", "evaluate"])
def test_stage_entrypoints_preserve_full_run_progress_snapshot(
    tmp_path,
    entrypoint_name: str,
) -> None:
    registry, _engine = _build_registry()
    project = _build_project_spec(tmp_path)
    orchestrator = Orchestrator.from_project_spec(
        project,
        registry=registry,
        dataset_loader=MockDatasetLoader(),
    )
    experiment = _build_experiment()
    orchestrator.run(experiment, runtime=RuntimeContext())
    snapshots = []

    getattr(orchestrator, entrypoint_name)(
        experiment,
        runtime=RuntimeContext(),
        progress=ProgressConfig(callback=snapshots.append),
    )

    persisted = orchestrator.get_run_progress(snapshots[-1].run_id)

    assert persisted is not None
    assert set(persisted.stage_counts) == {
        RunStage.GENERATION,
        RunStage.TRANSFORM,
        RunStage.EVALUATION,
    }


def test_orchestrator_estimate_reports_best_effort_work_item_and_token_counts(
    tmp_path,
) -> None:
    registry, _engine = _build_registry()
    project = _build_project_spec(tmp_path).model_copy(
        update={
            "execution_backend": WorkerPoolExecutionBackendSpec(
                lease_ttl_seconds=180,
                poll_interval_seconds=5,
                worker_tags=["gpu:a100"],
            )
        }
    )
    orchestrator = Orchestrator.from_project_spec(
        project,
        registry=registry,
        dataset_loader=MockDatasetLoader(),
    )

    estimate = orchestrator.estimate(_build_experiment())

    assert isinstance(estimate, CostEstimate)
    assert estimate.backend_kind == "worker_pool"
    assert estimate.total_work_items == 12
    assert estimate.work_items_by_stage == {
        "generation": 4,
        "transform": 4,
        "evaluation": 4,
    }
    assert estimate.estimated_prompt_tokens > 0
    assert estimate.estimated_completion_tokens == 4096
    assert estimate.estimated_total_tokens == (
        estimate.estimated_prompt_tokens + estimate.estimated_completion_tokens
    )
    assert estimate.estimated_total_cost is None
    assert estimate.notes
