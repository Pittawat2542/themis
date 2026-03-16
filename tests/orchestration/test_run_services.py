from __future__ import annotations

from themis.orchestration.run_manifest import RunManifest, StageWorkItem, WorkItemStatus
from themis.orchestration.run_services import (
    RunPlanningService,
    _evaluation_status,
    _generation_status,
    _transform_status,
)
from themis.specs.experiment import (
    ExperimentSpec,
    ExecutionPolicySpec,
    InferenceGridSpec,
    InferenceParamsSpec,
    ProjectSpec,
    PromptTemplateSpec,
    RuntimeContext,
    SqliteBlobStorageSpec,
)
from themis.specs.foundational import DatasetSpec, GenerationSpec, ModelSpec, TaskSpec
from themis.types.enums import DatasetSource, RecordStatus, RunStage
from themis.types.events import (
    EvaluationCompletedEventMetadata,
    ExtractionCompletedEventMetadata,
    TrialEvent,
    TrialEventType,
)


def _experiment() -> ExperimentSpec:
    return ExperimentSpec(
        models=[ModelSpec(model_id="mock-model", provider="mock")],
        tasks=[
            TaskSpec(
                task_id="task",
                dataset=DatasetSpec(source=DatasetSource.MEMORY),
                generation=GenerationSpec(),
            )
        ],
        prompt_templates=[PromptTemplateSpec(id="baseline", messages=[])],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec()]),
    )


def _event(
    *,
    event_seq: int,
    event_type: TrialEventType,
    status: RecordStatus | None = None,
    metadata=None,
) -> TrialEvent:
    return TrialEvent(
        trial_hash="trial-1",
        event_seq=event_seq,
        event_id=f"evt-{event_seq}",
        event_type=event_type,
        candidate_id="candidate-1",
        status=status,
        metadata=metadata,
    )


def test_generation_status_uses_latest_terminal_event() -> None:
    events = [
        _event(
            event_seq=1,
            event_type=TrialEventType.CANDIDATE_COMPLETED,
            status=RecordStatus.OK,
        ),
        _event(
            event_seq=2,
            event_type=TrialEventType.CANDIDATE_FAILED,
            status=RecordStatus.ERROR,
        ),
    ]

    assert _generation_status(events) == WorkItemStatus.FAILED


def test_transform_status_uses_latest_matching_overlay_event() -> None:
    transform_hash = "transform-1"
    events = [
        _event(
            event_seq=1,
            event_type=TrialEventType.EXTRACTION_COMPLETED,
            status=RecordStatus.OK,
            metadata=ExtractionCompletedEventMetadata(
                transform_hash=transform_hash,
                success=True,
            ),
        ),
        _event(
            event_seq=2,
            event_type=TrialEventType.CANDIDATE_FAILED,
            status=RecordStatus.ERROR,
            metadata={"transform_hash": transform_hash},
        ),
    ]

    assert (
        _transform_status(events, transform_hash=transform_hash)
        == WorkItemStatus.FAILED
    )


def test_evaluation_status_requires_explicit_success() -> None:
    evaluation_hash = "evaluation-1"
    events = [
        _event(
            event_seq=1,
            event_type=TrialEventType.EVALUATION_COMPLETED,
            status=RecordStatus.OK,
            metadata=EvaluationCompletedEventMetadata(
                evaluation_hash=evaluation_hash,
            ),
        ),
        _event(
            event_seq=2,
            event_type=TrialEventType.EVALUATION_COMPLETED,
            status=RecordStatus.SKIPPED,
            metadata=EvaluationCompletedEventMetadata(
                evaluation_hash=evaluation_hash,
            ),
        ),
    ]

    assert (
        _evaluation_status(events, evaluation_hash=evaluation_hash)
        != WorkItemStatus.COMPLETED
    )


def test_resume_reexecutes_local_runs_with_running_items() -> None:
    experiment = _experiment()
    stored_manifest = RunManifest(
        run_id="run-1",
        backend_kind="local",
        experiment_spec=experiment,
        work_items=[
            StageWorkItem(
                work_item_id="work-1",
                stage=RunStage.GENERATION,
                status=WorkItemStatus.RUNNING,
                trial_hash="trial-1",
                candidate_index=0,
                candidate_id="candidate-1",
            )
        ],
    )
    planned_manifest = stored_manifest.model_copy()
    execute_calls: list[tuple[ExperimentSpec, RuntimeContext | None]] = []

    class FakePlanner:
        def plan_experiment(self, experiment_spec: ExperimentSpec):
            del experiment_spec
            return []

    class FakeManifestRepo:
        def __init__(self) -> None:
            self._after_execute = False

        def get_manifest(self, run_id: str) -> RunManifest | None:
            assert run_id == stored_manifest.run_id
            if self._after_execute:
                return stored_manifest.model_copy(
                    update={
                        "work_items": [
                            stored_manifest.work_items[0].model_copy(
                                update={"status": WorkItemStatus.COMPLETED}
                            )
                        ]
                    }
                )
            return stored_manifest

        def reconcile_manifest(self, manifest: RunManifest) -> RunManifest:
            return planned_manifest

        def save_manifest(self, manifest: RunManifest) -> None:
            del manifest

    manifest_repo = FakeManifestRepo()
    service = RunPlanningService(
        planner=FakePlanner(),
        event_repo=object(),
        projection_repo=object(),
        projection_handler=object(),
        manifest_repo=manifest_repo,
        project_spec=ProjectSpec(
            project_name="project",
            researcher_id="researcher",
            global_seed=7,
            storage=SqliteBlobStorageSpec(root_dir=".cache/themis"),
            execution_policy=ExecutionPolicySpec(),
        ),
    )

    def execute_run(experiment_spec: ExperimentSpec, runtime: RuntimeContext | None):
        execute_calls.append((experiment_spec, runtime))
        manifest_repo._after_execute = True

        class Result:
            trial_hashes: list[str] = []

        return Result()

    resumed = service.resume(
        stored_manifest.run_id,
        runtime=RuntimeContext(),
        execute_run=execute_run,
    )

    assert execute_calls == [(experiment, RuntimeContext())]
    assert resumed is not None
