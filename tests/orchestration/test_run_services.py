from __future__ import annotations

from collections.abc import Iterator, Sequence

from themis.orchestration.run_manifest import RunManifest, StageWorkItem, WorkItemStatus
from themis.benchmark.specs import BenchmarkSpec, PromptVariantSpec, SliceSpec
from themis.orchestration.run_services import (
    RunPlanningService,
    _evaluation_status,
    _generation_status,
    _transform_status,
)
from themis.orchestration.trial_planner import PlannedTrial, TrialPlanner
from themis.records.conversation import Conversation
from themis.records.timeline import RecordTimeline
from themis.records.trial import TrialRecord
from themis.runtime.experiment_result import ExperimentResult
from themis.runtime import RecordTimelineView
from themis.specs.experiment import (
    ExperimentSpec,
    ExecutionPolicySpec,
    InferenceGridSpec,
    InferenceParamsSpec,
    PromptMessage,
    ProjectSpec,
    PromptTemplateSpec,
    RuntimeContext,
    SqliteBlobStorageSpec,
)
from themis.specs.base import SpecBase
from themis.specs.foundational import DatasetSpec, GenerationSpec, ModelSpec, TaskSpec
from themis.types.enums import DatasetSource, PromptRole, RecordStatus, RunStage
from themis.types.events import (
    EvaluationCompletedEventMetadata,
    ExtractionCompletedEventMetadata,
    ScoreRow,
    TrialEvent,
    TrialEventType,
    TrialSummaryRow,
)
from themis.storage.run_manifest_repo import RunManifestRepository


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


def _benchmark() -> BenchmarkSpec:
    return BenchmarkSpec(
        benchmark_id="benchmark",
        models=[ModelSpec(model_id="mock-model", provider="mock")],
        slices=[
            SliceSpec(
                slice_id="task",
                dataset=DatasetSpec(source=DatasetSource.MEMORY),
                generation=GenerationSpec(),
                prompt_variant_ids=["baseline"],
            )
        ],
        prompt_variants=[
            PromptVariantSpec(
                id="baseline",
                messages=[
                    PromptMessage(role=PromptRole.USER, content="Solve the problem.")
                ],
            )
        ],
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


def test_estimate_notes_report_empty_plans_without_heuristic_claims() -> None:
    class FakePlanner(TrialPlanner):
        def plan_experiment(
            self,
            experiment_spec: ExperimentSpec,
            *,
            required_stages=None,
        ) -> list[PlannedTrial]:
            del experiment_spec, required_stages
            return []

    class FakeEventRepo:
        def save_spec(self, spec: SpecBase) -> None:
            del spec

        def append_event(self, event: TrialEvent) -> None:
            del event

        def last_event_index(
            self, trial_hash: str, candidate_id: str | None = None
        ) -> int | None:
            del trial_hash, candidate_id
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

    class FakeProjectionRepo:
        def get_trial_record(self, *args, **kwargs) -> TrialRecord | None:
            del args, kwargs
            return None

        def get_conversation(self, *args, **kwargs) -> Conversation | None:
            del args, kwargs
            return None

        def get_record_timeline(self, *args, **kwargs) -> RecordTimeline | None:
            del args, kwargs
            return None

        def get_timeline_view(self, *args, **kwargs) -> RecordTimelineView | None:
            del args, kwargs
            return None

        def materialize_trial_record(
            self,
            trial_hash: str,
            *,
            transform_hash: str | None = None,
            evaluation_hash: str | None = None,
            extra_events: list[TrialEvent] | None = None,
        ) -> TrialRecord:
            del trial_hash, transform_hash, evaluation_hash, extra_events
            return TrialRecord(spec_hash="trial", candidates=[])

        def iter_candidate_scores(self, *args, **kwargs) -> Iterator[ScoreRow]:
            del args, kwargs
            return iter(())

        def iter_trial_summaries(self, *args, **kwargs) -> Iterator[TrialSummaryRow]:
            del args, kwargs
            return iter(())

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

    class FakeProjectionHandler:
        def on_trial_completed(self, *args, **kwargs) -> TrialRecord | None:
            del args, kwargs
            return None

    class FakeManifestRepo:
        def reconcile_manifest(
            self,
            manifest: RunManifest,
            *,
            stored_manifest: RunManifest | None = None,
        ) -> RunManifest:
            del stored_manifest
            return manifest

        def save_manifest(self, manifest: RunManifest) -> None:
            del manifest

    service = RunPlanningService(
        planner=FakePlanner(),
        event_repo=FakeEventRepo(),
        projection_repo=FakeProjectionRepo(),
        projection_handler=FakeProjectionHandler(),
        manifest_repo=FakeManifestRepo(),  # type: ignore[arg-type]
    )

    estimate = service.estimate(_experiment())

    assert estimate.estimated_prompt_tokens == 0
    assert estimate.total_work_items == 0
    assert "No planned trials: no prompt tokens to estimate." in estimate.notes
    assert not any(
        "Best-effort heuristic only" in note or "heuristic fallback" in note
        for note in estimate.notes
    )


def test_generation_status_reads_legacy_payload_status_when_event_status_missing() -> (
    None
):
    events = [
        TrialEvent(
            trial_hash="trial-1",
            event_seq=1,
            event_id="evt-1",
            event_type=TrialEventType.CANDIDATE_COMPLETED,
            candidate_id="candidate-1",
            payload={"status": "ok"},
        )
    ]

    assert _generation_status(events) == WorkItemStatus.COMPLETED


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
    execute_calls: list[
        tuple[ExperimentSpec | BenchmarkSpec, RuntimeContext | None]
    ] = []

    class FakePlanner(TrialPlanner):
        def plan_experiment(
            self,
            experiment_spec: ExperimentSpec,
            *,
            required_stages=None,
        ) -> list[PlannedTrial]:
            del experiment_spec, required_stages
            return []

    class FakeEventRepo:
        def save_spec(self, spec: SpecBase) -> None:
            del spec

        def append_event(self, event: TrialEvent) -> None:
            del event

        def last_event_index(
            self, trial_hash: str, candidate_id: str | None = None
        ) -> int | None:
            del trial_hash, candidate_id
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

    class FakeProjectionRepo:
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
            del trial_hash, candidate_id
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
        ) -> TrialRecord:
            del trial_hash, transform_hash, evaluation_hash, extra_events
            raise NotImplementedError

        def iter_candidate_scores(
            self,
            *,
            trial_hashes: Sequence[str] | None = None,
            metric_id: str | None = None,
            evaluation_hash: str | None = None,
        ) -> Iterator[ScoreRow]:
            del trial_hashes, metric_id, evaluation_hash
            return iter(())

        def iter_trial_summaries(
            self,
            *,
            trial_hashes: Sequence[str] | None = None,
            transform_hash: str | None = None,
            evaluation_hash: str | None = None,
        ) -> Iterator[TrialSummaryRow]:
            del trial_hashes, transform_hash, evaluation_hash
            return iter(())

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

    class FakeProjectionHandler:
        def on_trial_completed(
            self,
            trial_hash: str,
            *,
            transform_hash: str | None = None,
            evaluation_hash: str | None = None,
        ) -> TrialRecord | None:
            del trial_hash, transform_hash, evaluation_hash
            return None

    class FakeManifestRepo(RunManifestRepository):
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

        def reconcile_manifest(
            self,
            manifest: RunManifest,
            *,
            stored_manifest: RunManifest | None = None,
        ) -> RunManifest:
            del manifest, stored_manifest
            return planned_manifest

        def save_manifest(self, manifest: RunManifest) -> None:
            del manifest

    manifest_repo = FakeManifestRepo()
    service = RunPlanningService(
        planner=FakePlanner(),
        event_repo=FakeEventRepo(),
        projection_repo=FakeProjectionRepo(),
        projection_handler=FakeProjectionHandler(),
        manifest_repo=manifest_repo,
        project_spec=ProjectSpec(
            project_name="project",
            researcher_id="researcher",
            global_seed=7,
            storage=SqliteBlobStorageSpec(root_dir=".cache/themis"),
            execution_policy=ExecutionPolicySpec(),
        ),
    )

    def execute_run(
        experiment_spec: ExperimentSpec | BenchmarkSpec,
        runtime: RuntimeContext | None,
    ) -> ExperimentResult:
        execute_calls.append((experiment_spec, runtime))
        manifest_repo._after_execute = True
        return ExperimentResult(
            projection_repo=service.projection_repo, trial_hashes=[]
        )

    resumed = service.resume(
        stored_manifest.run_id,
        runtime=RuntimeContext(),
        execute_run=execute_run,
    )

    assert execute_calls == [(experiment, RuntimeContext())]
    assert resumed is not None


def test_submit_replans_local_run_after_execution_without_progress_tracker() -> None:
    experiment = _experiment()
    initial_manifest = RunManifest(
        run_id="run-1",
        backend_kind="local",
        experiment_spec=experiment,
        work_items=[
            StageWorkItem(
                work_item_id="work-1",
                stage=RunStage.GENERATION,
                status=WorkItemStatus.PENDING,
                trial_hash="trial-1",
                candidate_index=0,
                candidate_id="candidate-1",
            )
        ],
    )
    completed_manifest = initial_manifest.model_copy(
        update={
            "work_items": [
                initial_manifest.work_items[0].model_copy(
                    update={"status": WorkItemStatus.COMPLETED}
                )
            ]
        }
    )
    execute_calls: list[
        tuple[ExperimentSpec | BenchmarkSpec, RuntimeContext | None]
    ] = []

    class FakePlanner(TrialPlanner):
        def plan_experiment(
            self,
            experiment_spec: ExperimentSpec,
            *,
            required_stages=None,
        ) -> list[PlannedTrial]:
            del experiment_spec, required_stages
            return []

    class FakeEventRepo:
        def save_spec(self, spec: SpecBase) -> None:
            del spec

        def append_event(self, event: TrialEvent) -> None:
            del event

        def last_event_index(
            self, trial_hash: str, candidate_id: str | None = None
        ) -> int | None:
            del trial_hash, candidate_id
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

    class FakeProjectionRepo:
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
            del trial_hash, candidate_id
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
        ) -> TrialRecord:
            del trial_hash, transform_hash, evaluation_hash, extra_events
            raise NotImplementedError

        def iter_candidate_scores(
            self,
            *,
            trial_hashes: Sequence[str] | None = None,
            metric_id: str | None = None,
            evaluation_hash: str | None = None,
        ) -> Iterator[ScoreRow]:
            del trial_hashes, metric_id, evaluation_hash
            return iter(())

        def iter_trial_summaries(
            self,
            *,
            trial_hashes: Sequence[str] | None = None,
            transform_hash: str | None = None,
            evaluation_hash: str | None = None,
        ) -> Iterator[TrialSummaryRow]:
            del trial_hashes, transform_hash, evaluation_hash
            return iter(())

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

    class FakeProjectionHandler:
        def on_trial_completed(
            self,
            trial_hash: str,
            *,
            transform_hash: str | None = None,
            evaluation_hash: str | None = None,
        ) -> TrialRecord | None:
            del trial_hash, transform_hash, evaluation_hash
            return None

    class FakeManifestRepo(RunManifestRepository):
        def __init__(self) -> None:
            self._after_execute = False

        def get_manifest(self, run_id: str) -> RunManifest | None:
            assert run_id == initial_manifest.run_id
            return initial_manifest

        def reconcile_manifest(
            self,
            manifest: RunManifest,
            *,
            stored_manifest: RunManifest | None = None,
        ) -> RunManifest:
            del manifest, stored_manifest
            if self._after_execute:
                return completed_manifest
            return initial_manifest

        def save_manifest(self, manifest: RunManifest) -> None:
            del manifest

    manifest_repo = FakeManifestRepo()
    service = RunPlanningService(
        planner=FakePlanner(),
        event_repo=FakeEventRepo(),
        projection_repo=FakeProjectionRepo(),
        projection_handler=FakeProjectionHandler(),
        manifest_repo=manifest_repo,
        project_spec=ProjectSpec(
            project_name="project",
            researcher_id="researcher",
            global_seed=7,
            storage=SqliteBlobStorageSpec(root_dir=".cache/themis"),
            execution_policy=ExecutionPolicySpec(),
        ),
    )

    def execute_run(
        experiment_spec: ExperimentSpec | BenchmarkSpec,
        runtime: RuntimeContext | None,
    ) -> ExperimentResult:
        execute_calls.append((experiment_spec, runtime))
        manifest_repo._after_execute = True
        return ExperimentResult(
            projection_repo=service.projection_repo, trial_hashes=[]
        )

    handle = service.submit(
        experiment,
        runtime=RuntimeContext(),
        execute_run=execute_run,
    )

    assert execute_calls == [(experiment, RuntimeContext())]
    assert handle.status == "completed"


def test_submit_executes_local_benchmark_source_without_downgrading_to_experiment() -> (
    None
):
    experiment = _experiment()
    benchmark = _benchmark()
    initial_manifest = RunManifest(
        run_id="run-benchmark",
        backend_kind="local",
        experiment_spec=experiment,
        benchmark_spec=benchmark,
        work_items=[
            StageWorkItem(
                work_item_id="work-1",
                stage=RunStage.GENERATION,
                status=WorkItemStatus.PENDING,
                trial_hash="trial-1",
                candidate_index=0,
                candidate_id="candidate-1",
            )
        ],
    )
    completed_manifest = initial_manifest.model_copy(
        update={
            "work_items": [
                initial_manifest.work_items[0].model_copy(
                    update={"status": WorkItemStatus.COMPLETED}
                )
            ]
        }
    )
    execute_calls: list[
        tuple[ExperimentSpec | BenchmarkSpec, RuntimeContext | None]
    ] = []

    class FakePlanner(TrialPlanner):
        def plan_experiment(
            self,
            experiment_spec: ExperimentSpec,
            *,
            required_stages=None,
        ) -> list[PlannedTrial]:
            del experiment_spec, required_stages
            return []

    class FakeEventRepo:
        def save_spec(self, spec: SpecBase) -> None:
            del spec

        def append_event(self, event: TrialEvent) -> None:
            del event

        def last_event_index(
            self, trial_hash: str, candidate_id: str | None = None
        ) -> int | None:
            del trial_hash, candidate_id
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

    class FakeProjectionRepo:
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
            del trial_hash, candidate_id
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
        ) -> TrialRecord:
            del trial_hash, transform_hash, evaluation_hash, extra_events
            raise NotImplementedError

        def iter_candidate_scores(
            self,
            *,
            trial_hashes: Sequence[str] | None = None,
            metric_id: str | None = None,
            evaluation_hash: str | None = None,
        ) -> Iterator[ScoreRow]:
            del trial_hashes, metric_id, evaluation_hash
            return iter(())

        def iter_trial_summaries(
            self,
            *,
            trial_hashes: Sequence[str] | None = None,
            transform_hash: str | None = None,
            evaluation_hash: str | None = None,
        ) -> Iterator[TrialSummaryRow]:
            del trial_hashes, transform_hash, evaluation_hash
            return iter(())

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

    class FakeProjectionHandler:
        def on_trial_completed(
            self,
            trial_hash: str,
            *,
            transform_hash: str | None = None,
            evaluation_hash: str | None = None,
        ) -> TrialRecord | None:
            del trial_hash, transform_hash, evaluation_hash
            return None

    class FakeManifestRepo(RunManifestRepository):
        def __init__(self) -> None:
            self._after_execute = False

        def get_manifest(self, run_id: str) -> RunManifest | None:
            assert run_id == initial_manifest.run_id
            return initial_manifest

        def reconcile_manifest(
            self,
            manifest: RunManifest,
            *,
            stored_manifest: RunManifest | None = None,
        ) -> RunManifest:
            del manifest, stored_manifest
            if self._after_execute:
                return completed_manifest
            return initial_manifest

        def save_manifest(self, manifest: RunManifest) -> None:
            del manifest

    manifest_repo = FakeManifestRepo()
    service = RunPlanningService(
        planner=FakePlanner(),
        event_repo=FakeEventRepo(),
        projection_repo=FakeProjectionRepo(),
        projection_handler=FakeProjectionHandler(),
        manifest_repo=manifest_repo,
        project_spec=ProjectSpec(
            project_name="project",
            researcher_id="researcher",
            global_seed=7,
            storage=SqliteBlobStorageSpec(root_dir=".cache/themis"),
            execution_policy=ExecutionPolicySpec(),
        ),
    )

    def execute_run(
        experiment_spec: ExperimentSpec | BenchmarkSpec,
        runtime: RuntimeContext | None,
    ) -> ExperimentResult:
        execute_calls.append((experiment_spec, runtime))
        manifest_repo._after_execute = True
        return ExperimentResult(
            projection_repo=service.projection_repo, trial_hashes=[]
        )

    handle = service.submit(
        experiment,
        benchmark_spec=benchmark,
        runtime=RuntimeContext(),
        execute_run=execute_run,
    )

    assert execute_calls == [(benchmark, RuntimeContext())]
    assert handle.status == "completed"
