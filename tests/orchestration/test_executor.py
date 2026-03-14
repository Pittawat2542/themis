import asyncio
import threading
import time
from dataclasses import dataclass
from typing import cast

import pytest

from themis.orchestration._executor_support import ExecutionSupport
from themis.orchestration.executor import TrialExecutor, _ExecutionRunner
from themis.orchestration.projection_handler import ProjectionHandler
from themis.orchestration.runner_state import TrialExecutionSession
from themis.orchestration.task_resolution import resolve_task_stages
from themis.orchestration.trial_planner import PlannedTrial
from themis.records.provenance import ProvenanceRecord
from themis.orchestration.work_scheduler import WorkSchedulerStats
from themis.records.candidate import CandidateRecord
from themis.records.evaluation import EvaluationRecord, MetricScore
from themis.records.error import ErrorRecord
from themis.records.extraction import ExtractionRecord
from themis.records.trial import TrialRecord
from themis.specs.experiment import (
    DataItemContext,
    ExecutionPolicySpec,
    InferenceParamsSpec,
    PromptTemplateSpec,
    RuntimeContext,
    TrialSpec,
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
from themis.storage.event_repo import SqliteEventRepository
from themis.storage.projection_repo import SqliteProjectionRepository
from themis.storage.sqlite_schema import DatabaseManager
from themis.types.enums import RecordStatus
from themis.types.enums import ErrorCode, ErrorWhere
from themis.types.events import ArtifactRef, TrialEvent, TrialEventType


def _prepared_session(
    trial: TrialSpec,
    dataset_context: DataItemContext,
    runtime_context: RuntimeContext | None,
) -> TrialExecutionSession:
    return TrialExecutionSession(
        trial=trial,
        dataset_context=dataset_context,
        base_runtime=runtime_context or RuntimeContext(),
        provenance=ProvenanceRecord(
            themis_version="test",
            git_commit=None,
            python_version="3.12",
            platform="test",
            library_versions={},
            model_endpoint_meta={},
        ),
        resolved_stages=resolve_task_stages(trial.task),
        prompt_payload={"messages": []},
        prompt_artifact=(
            ArtifactRef(
                artifact_hash="sha256:test-prompt",
                media_type="application/json",
                label="rendered_prompt",
            ),
            "sha256:test-prompt",
        ),
        item_payload=dict(dataset_context.payload),
        dataset_metadata={},
        event_seq=0,
    )


class RecordingRunner:
    def __init__(self, sleep_s: float = 0.02) -> None:
        self.sleep_s = sleep_s
        self.max_seen_in_flight = 0
        self.interleaved_trial_ids: set[str] = set()
        self._current_in_flight = 0
        self._active_trials: set[str] = set()
        self._lock = threading.Lock()

    def prepare_trial_session(
        self,
        trial: TrialSpec,
        dataset_context: DataItemContext,
        runtime_context: RuntimeContext | None,
        *,
        required_stages=None,
    ) -> TrialExecutionSession:
        del required_stages
        return _prepared_session(trial, dataset_context, runtime_context)

    def run_generation_candidate(
        self,
        prepared_trial: TrialExecutionSession,
        candidate_index: int,
    ) -> CandidateRecord:
        with self._lock:
            self._current_in_flight += 1
            self.max_seen_in_flight = max(
                self.max_seen_in_flight,
                self._current_in_flight,
            )
            self._active_trials.add(prepared_trial.trial.trial_id)
            if len(self._active_trials) > 1:
                self.interleaved_trial_ids.update(self._active_trials)
        time.sleep(self.sleep_s)
        with self._lock:
            self._current_in_flight -= 1
            self._active_trials.discard(prepared_trial.trial.trial_id)
        candidate_id = f"{prepared_trial.trial.spec_hash}:{candidate_index}"
        return CandidateRecord(
            spec_hash=candidate_id,
            candidate_id=candidate_id,
            sample_index=candidate_index,
            status=RecordStatus.OK,
        )

    def finalize_generation_trial(
        self,
        prepared_trial: TrialExecutionSession,
        candidates: list[CandidateRecord],
    ) -> TrialRecord:
        ordered_candidates = sorted(
            candidates, key=lambda candidate: candidate.sample_index
        )
        return TrialRecord(
            spec_hash=prepared_trial.trial.spec_hash,
            trial_spec=prepared_trial.trial,
            status=RecordStatus.OK,
            candidates=ordered_candidates,
        )

    def run_output_transform(self, session, candidate, transform) -> CandidateRecord:
        del session, transform
        return candidate

    def run_evaluation_candidate(
        self, session, candidate, evaluation
    ) -> CandidateRecord:
        del session, evaluation
        return candidate


class StubProjectionRepository:
    def get_trial_record(self, trial_hash: str, *_, **__) -> TrialRecord | None:
        del trial_hash
        return None

    def has_trial(self, trial_hash: str, *_, **__) -> bool:
        del trial_hash
        return False

    def materialize_trial_record(self, trial_hash: str, *_, **__) -> TrialRecord:
        raise AssertionError(f"unexpected materialization for {trial_hash}")


class StubProjectionHandler:
    def __init__(self) -> None:
        self.completed_trial_hashes: list[str] = []
        self.completed_overlays: list[tuple[str, str | None, str | None]] = []

    def on_trial_completed(
        self,
        trial_hash: str,
        *,
        transform_hash: str | None = None,
        evaluation_hash: str | None = None,
    ) -> None:
        self.completed_trial_hashes.append(trial_hash)
        self.completed_overlays.append((trial_hash, transform_hash, evaluation_hash))
        return None


def _trial_spec(trial_id: str, *, candidate_count: int) -> TrialSpec:
    return TrialSpec(
        trial_id=trial_id,
        model=ModelSpec(model_id="mock-model", provider="mock-provider"),
        task=TaskSpec(
            task_id=f"task-{trial_id}",
            dataset=DatasetSpec(source="memory"),
            generation=GenerationSpec(),
        ),
        item_id=f"item-{trial_id}",
        prompt=PromptTemplateSpec(id="baseline", messages=[]),
        params=InferenceParamsSpec(),
        candidate_count=candidate_count,
    )


def _planned_trial(trial_id: str, *, candidate_count: int) -> PlannedTrial:
    trial = _trial_spec(trial_id, candidate_count=candidate_count)
    return PlannedTrial(
        trial_spec=trial,
        dataset_context=DataItemContext(
            item_id=trial.item_id, payload={"value": trial_id}
        ),
    )


def _overlay_trial_spec(trial_id: str) -> TrialSpec:
    return TrialSpec(
        trial_id=trial_id,
        model=ModelSpec(model_id="mock-model", provider="mock-provider"),
        task=TaskSpec(
            task_id=f"task-{trial_id}",
            dataset=DatasetSpec(source="memory"),
            generation=GenerationSpec(),
            output_transforms=[
                OutputTransformSpec(
                    name="parsed",
                    extractor_chain=ExtractorChainSpec(
                        extractors=[ExtractorRefSpec(id="mock-extractor")]
                    ),
                )
            ],
            evaluations=[
                EvaluationSpec(
                    name="score",
                    transform="parsed",
                    metrics=["exact_match"],
                )
            ],
        ),
        item_id=f"item-{trial_id}",
        prompt=PromptTemplateSpec(id="baseline", messages=[]),
        params=InferenceParamsSpec(),
        candidate_count=1,
    )


class OverlayRecordingRunner:
    def __init__(self) -> None:
        self.transform_calls: list[tuple[str, str, str]] = []
        self.evaluation_calls: list[tuple[str, str, str]] = []

    def prepare_trial_session(
        self,
        trial: TrialSpec,
        dataset_context: DataItemContext,
        runtime_context: RuntimeContext | None,
        *,
        required_stages=None,
    ) -> TrialExecutionSession:
        del required_stages
        return _prepared_session(trial, dataset_context, runtime_context)

    def run_generation_candidate(
        self,
        session: TrialExecutionSession,
        cand_index: int,
    ) -> CandidateRecord:
        del session, cand_index
        raise AssertionError(
            "generation should not run in overlay characterization test"
        )

    def finalize_generation_trial(
        self,
        session: TrialExecutionSession,
        candidates: list[CandidateRecord],
    ) -> TrialRecord:
        del session, candidates
        raise AssertionError(
            "generation should not run in overlay characterization test"
        )

    def run_output_transform(
        self,
        session: TrialExecutionSession,
        candidate: CandidateRecord,
        transform,
    ) -> CandidateRecord:
        self.transform_calls.append(
            (session.trial.spec_hash, candidate.spec_hash, transform.transform_hash)
        )
        return candidate.model_copy(
            update={
                "extractions": [
                    ExtractionRecord(
                        spec_hash=f"{candidate.spec_hash}:extract",
                        extractor_id="mock-extractor",
                        success=True,
                        parsed_answer="42",
                    )
                ]
            }
        )

    def run_evaluation_candidate(
        self,
        session: TrialExecutionSession,
        candidate: CandidateRecord,
        evaluation,
    ) -> CandidateRecord:
        self.evaluation_calls.append(
            (session.trial.spec_hash, candidate.spec_hash, evaluation.evaluation_hash)
        )
        return candidate.model_copy(
            update={
                "evaluation": EvaluationRecord(
                    spec_hash=evaluation.evaluation_hash,
                    metric_scores=[MetricScore(metric_id="exact_match", value=1.0)],
                )
            }
        )


class OverlayProjectionRepository:
    def __init__(
        self,
        *,
        generation_record: TrialRecord,
        transform_record: TrialRecord,
        transform_hash: str,
    ) -> None:
        self.generation_record = generation_record
        self.transform_record = transform_record
        self.transform_hash = transform_hash
        self.get_trial_calls: list[tuple[str, str | None, str | None]] = []

    def get_trial_record(
        self,
        trial_hash: str,
        *,
        transform_hash: str | None = None,
        evaluation_hash: str | None = None,
    ) -> TrialRecord | None:
        self.get_trial_calls.append((trial_hash, transform_hash, evaluation_hash))
        if (
            trial_hash != self.generation_record.spec_hash
            or evaluation_hash is not None
        ):
            return None
        if transform_hash is None:
            return self.generation_record
        if transform_hash == self.transform_hash:
            return self.transform_record
        return None

    def has_trial(
        self,
        trial_hash: str,
        *,
        transform_hash: str | None = None,
        evaluation_hash: str | None = None,
    ) -> bool:
        del trial_hash, transform_hash, evaluation_hash
        return False

    def materialize_trial_record(
        self,
        trial_hash: str,
        *,
        transform_hash: str | None = None,
        evaluation_hash: str | None = None,
    ) -> TrialRecord:
        raise AssertionError(
            "materialize_trial_record should not be called directly when a projection handler exists"
        )


@dataclass(slots=True)
class _FakePreparedSession:
    trial: TrialSpec
    dataset_context: DataItemContext
    runtime_context: RuntimeContext

    @property
    def trial_hash(self) -> str:
        return self.trial.spec_hash


def test_scheduler_uses_bounded_global_work_queue() -> None:
    runner = RecordingRunner()
    projection_handler = StubProjectionHandler()
    executor = TrialExecutor(
        runner=cast(_ExecutionRunner, runner),
        projection_repo=StubProjectionRepository(),
        projection_handler=projection_handler,
        execution_policy=ExecutionPolicySpec(max_in_flight_work_items=8),
    )

    trials = [
        _planned_trial("trial-a", candidate_count=8),
        _planned_trial("trial-b", candidate_count=8),
        _planned_trial("trial-c", candidate_count=8),
    ]

    executor.execute_generation_trials(trials, RuntimeContext(), resume=False)

    assert runner.max_seen_in_flight == 8
    assert executor.last_scheduler_stats is not None
    assert executor.last_scheduler_stats.max_buffered_work_items <= 16
    assert sorted(projection_handler.completed_trial_hashes) == sorted(
        planned_trial.trial_spec.spec_hash for planned_trial in trials
    )


def test_generation_work_items_can_interleave_across_trials() -> None:
    runner = RecordingRunner()
    executor = TrialExecutor(
        runner=cast(_ExecutionRunner, runner),
        projection_repo=StubProjectionRepository(),
        projection_handler=StubProjectionHandler(),
        execution_policy=ExecutionPolicySpec(max_in_flight_work_items=2),
    )
    trials = [
        _planned_trial("trial-a", candidate_count=4),
        _planned_trial("trial-b", candidate_count=4),
    ]

    executor.execute_generation_trials(trials, RuntimeContext(), resume=False)

    assert runner.interleaved_trial_ids == {"trial-a", "trial-b"}


def test_executor_requires_concrete_trial_execution_sessions() -> None:
    class InvalidSessionRunner(RecordingRunner):
        def prepare_trial_session(
            self,
            trial: TrialSpec,
            dataset_context: DataItemContext,
            runtime_context: RuntimeContext | None,
            *,
            required_stages=None,
        ) -> _FakePreparedSession:
            del required_stages
            return _FakePreparedSession(
                trial=trial,
                dataset_context=dataset_context,
                runtime_context=runtime_context or RuntimeContext(),
            )

    executor = TrialExecutor(
        runner=cast(_ExecutionRunner, InvalidSessionRunner()),
        projection_repo=StubProjectionRepository(),
        projection_handler=StubProjectionHandler(),
    )

    with pytest.raises(TypeError, match="TrialExecutionSession"):
        executor.execute_generation_trials(
            [_planned_trial("trial-a", candidate_count=1)],
            RuntimeContext(),
            resume=False,
        )


def test_executor_stores_internal_runtime_collaborators_privately() -> None:
    runner = RecordingRunner()
    event_repo = object()
    projection_handler = StubProjectionHandler()
    telemetry_bus = object()
    executor = TrialExecutor(
        runner=cast(_ExecutionRunner, runner),
        projection_repo=StubProjectionRepository(),
        event_repo=cast(object, event_repo),
        projection_handler=projection_handler,
        telemetry_bus=cast(object, telemetry_bus),
    )

    assert "_support" in executor.__dict__
    assert "_generation_execution" in executor.__dict__
    assert "_overlay_execution" in executor.__dict__
    for attr_name in (
        "runner",
        "projection_repo",
        "event_repo",
        "projection_handler",
        "telemetry_bus",
        "generation_execution",
        "overlay_execution",
    ):
        assert attr_name not in executor.__dict__


@pytest.mark.parametrize(
    "attr_name",
    [
        "runner",
        "projection_repo",
        "event_repo",
        "projection_handler",
        "telemetry_bus",
    ],
)
def test_executor_internal_attrs_are_not_public_api(attr_name: str) -> None:
    runner = RecordingRunner()
    projection_repo = StubProjectionRepository()
    event_repo = object()
    projection_handler = StubProjectionHandler()
    telemetry_bus = object()
    executor = TrialExecutor(
        runner=cast(_ExecutionRunner, runner),
        projection_repo=projection_repo,
        event_repo=cast(object, event_repo),
        projection_handler=projection_handler,
        telemetry_bus=cast(object, telemetry_bus),
    )

    with pytest.raises(AttributeError, match=attr_name):
        getattr(executor, attr_name)


def test_executor_does_not_expose_legacy_execute_trials_alias() -> None:
    runner = RecordingRunner()
    executor = TrialExecutor(
        runner=cast(_ExecutionRunner, runner),
        projection_repo=StubProjectionRepository(),
        projection_handler=StubProjectionHandler(),
    )

    with pytest.raises(AttributeError, match="execute_trials"):
        getattr(executor, "execute_trials")


def test_executor_generation_can_run_inside_running_event_loop() -> None:
    runner = RecordingRunner()
    executor = TrialExecutor(
        runner=cast(_ExecutionRunner, runner),
        projection_repo=StubProjectionRepository(),
        projection_handler=StubProjectionHandler(),
        execution_policy=ExecutionPolicySpec(max_in_flight_work_items=2),
    )
    trials = [
        _planned_trial("trial-a", candidate_count=2),
        _planned_trial("trial-b", candidate_count=2),
    ]

    async def run_in_loop() -> None:
        executor.execute_generation_trials(trials, RuntimeContext(), resume=False)

    asyncio.run(run_in_loop())

    assert runner.max_seen_in_flight == 2


def test_executor_delegates_generation_execution_to_generation_coordinator() -> None:
    runner = RecordingRunner()
    executor = TrialExecutor(
        runner=cast(_ExecutionRunner, runner),
        projection_repo=StubProjectionRepository(),
        projection_handler=StubProjectionHandler(),
    )
    planned_trial = _planned_trial("trial-a", candidate_count=2)
    runtime = RuntimeContext()
    expected_stats = WorkSchedulerStats(
        max_seen_in_flight=5,
        max_buffered_work_items=10,
    )

    class StubGenerationExecution:
        def __init__(self) -> None:
            self.calls: list[tuple[object, object, object, bool]] = []

        def execute_generation_trials(
            self,
            trials,
            runtime_context,
            *,
            dataset_context=None,
            resume: bool = True,
        ) -> WorkSchedulerStats:
            self.calls.append((list(trials), runtime_context, dataset_context, resume))
            return expected_stats

    stub = StubGenerationExecution()
    executor._generation_execution = stub

    executor.execute_generation_trials([planned_trial], runtime, resume=False)

    assert stub.calls == [([planned_trial], runtime, None, False)]
    assert executor.last_scheduler_stats == expected_stats


def test_executor_delegates_transform_execution_to_overlay_coordinator() -> None:
    runner = RecordingRunner()
    executor = TrialExecutor(
        runner=cast(_ExecutionRunner, runner),
        projection_repo=StubProjectionRepository(),
        projection_handler=StubProjectionHandler(),
    )
    planned_trial = PlannedTrial(
        trial_spec=_overlay_trial_spec("trial-overlay"),
        dataset_context=DataItemContext(item_id="item-trial-overlay", payload={}),
    )
    runtime = RuntimeContext()
    expected_stats = WorkSchedulerStats(
        max_seen_in_flight=3,
        max_buffered_work_items=6,
    )

    class StubOverlayExecution:
        def __init__(self) -> None:
            self.transform_calls: list[tuple[object, object, object, bool]] = []

        def execute_transforms(
            self,
            trials,
            runtime_context,
            *,
            dataset_context=None,
            resume: bool = True,
        ) -> WorkSchedulerStats:
            self.transform_calls.append(
                (list(trials), runtime_context, dataset_context, resume)
            )
            return expected_stats

    stub = StubOverlayExecution()
    executor._overlay_execution = stub

    executor.execute_transforms([planned_trial], runtime, resume=False)

    assert stub.transform_calls == [([planned_trial], runtime, None, False)]
    assert executor.last_scheduler_stats == expected_stats


def test_executor_delegates_evaluation_execution_to_overlay_coordinator() -> None:
    runner = RecordingRunner()
    executor = TrialExecutor(
        runner=cast(_ExecutionRunner, runner),
        projection_repo=StubProjectionRepository(),
        projection_handler=StubProjectionHandler(),
    )
    planned_trial = PlannedTrial(
        trial_spec=_overlay_trial_spec("trial-overlay"),
        dataset_context=DataItemContext(item_id="item-trial-overlay", payload={}),
    )
    runtime = RuntimeContext()
    expected_stats = WorkSchedulerStats(
        max_seen_in_flight=2,
        max_buffered_work_items=4,
    )

    class StubOverlayExecution:
        def __init__(self) -> None:
            self.evaluation_calls: list[tuple[object, object, object, bool]] = []

        def execute_evaluations(
            self,
            trials,
            runtime_context,
            *,
            dataset_context=None,
            resume: bool = True,
        ) -> WorkSchedulerStats:
            self.evaluation_calls.append(
                (list(trials), runtime_context, dataset_context, resume)
            )
            return expected_stats

    stub = StubOverlayExecution()
    executor._overlay_execution = stub

    executor.execute_evaluations([planned_trial], runtime, resume=False)

    assert stub.evaluation_calls == [([planned_trial], runtime, None, False)]
    assert executor.last_scheduler_stats == expected_stats


def test_executor_uses_overlay_specific_inputs_and_materializes_overlay_projections() -> (
    None
):
    trial = _overlay_trial_spec("trial-overlay")
    planned_trial = PlannedTrial(
        trial_spec=trial,
        dataset_context=DataItemContext(item_id=trial.item_id, payload={"value": "x"}),
    )
    resolved = resolve_task_stages(trial.task)
    transform_hash = resolved.output_transforms[0].transform_hash
    evaluation_hash = resolved.evaluations[0].evaluation_hash

    base_candidate = CandidateRecord(
        spec_hash="candidate_1",
        candidate_id="candidate_1",
        sample_index=0,
        status=RecordStatus.OK,
    )
    transformed_candidate = base_candidate.model_copy(
        update={
            "extractions": [
                ExtractionRecord(
                    spec_hash="extract_1",
                    extractor_id="mock-extractor",
                    success=True,
                    parsed_answer="42",
                )
            ]
        }
    )
    generation_record = TrialRecord(
        spec_hash=trial.spec_hash,
        trial_spec=trial,
        status=RecordStatus.OK,
        candidates=[base_candidate],
    )
    transform_record = TrialRecord(
        spec_hash=trial.spec_hash,
        trial_spec=trial,
        status=RecordStatus.OK,
        candidates=[transformed_candidate],
    )

    runner = OverlayRecordingRunner()
    projection_handler = StubProjectionHandler()
    projection_repo = OverlayProjectionRepository(
        generation_record=generation_record,
        transform_record=transform_record,
        transform_hash=transform_hash,
    )
    executor = TrialExecutor(
        runner=cast(_ExecutionRunner, runner),
        projection_repo=projection_repo,
        projection_handler=projection_handler,
        execution_policy=ExecutionPolicySpec(max_in_flight_work_items=1),
    )

    executor.execute_transforms([planned_trial], RuntimeContext(), resume=False)
    executor.execute_evaluations([planned_trial], RuntimeContext(), resume=False)

    assert runner.transform_calls == [
        (trial.spec_hash, base_candidate.spec_hash, transform_hash)
    ]
    assert runner.evaluation_calls == [
        (trial.spec_hash, transformed_candidate.spec_hash, evaluation_hash)
    ]
    assert projection_repo.get_trial_calls == [
        (trial.spec_hash, None, None),
        (trial.spec_hash, transform_hash, None),
    ]
    assert projection_handler.completed_overlays == [
        (trial.spec_hash, transform_hash, None),
        (trial.spec_hash, None, evaluation_hash),
    ]


@pytest.mark.parametrize("failed_stage", ["transform", "evaluation"])
def test_executor_resume_does_not_skip_failed_overlay_projection(
    tmp_path,
    failed_stage: str,
) -> None:
    manager = DatabaseManager(f"sqlite:///{tmp_path}/executor_resume.db")
    manager.initialize()
    event_repo = SqliteEventRepository(manager)
    projection_repo = SqliteProjectionRepository(manager)
    projection_handler = ProjectionHandler(event_repo, projection_repo)
    support = ExecutionSupport(
        projection_repo=projection_repo,
        event_repo=event_repo,
        projection_handler=projection_handler,
        execution_policy=ExecutionPolicySpec(),
        telemetry_bus=None,
    )
    trial = _overlay_trial_spec("trial-resume")
    resolved = resolve_task_stages(trial.task)
    transform_hash = resolved.output_transforms[0].transform_hash
    evaluation_hash = resolved.evaluations[0].evaluation_hash

    event_repo.save_spec(trial)
    seed_events = [
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=1,
            event_id="evt_1",
            event_type=TrialEventType.CANDIDATE_STARTED,
            candidate_id="candidate_1",
            payload={"sample_index": 0},
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=2,
            event_id="evt_2",
            event_type=TrialEventType.CANDIDATE_FAILED,
            candidate_id="candidate_1",
            stage="extraction" if failed_stage == "transform" else "evaluation",
            status=RecordStatus.ERROR,
            metadata={
                "transform_hash": transform_hash
                if failed_stage == "transform"
                else None,
                "evaluation_hash": (
                    evaluation_hash if failed_stage == "evaluation" else None
                ),
            },
            error=ErrorRecord(
                code=ErrorCode.PARSE_ERROR
                if failed_stage == "transform"
                else ErrorCode.METRIC_COMPUTATION,
                message="overlay failed",
                retryable=False,
                where=ErrorWhere.EXTRACTOR
                if failed_stage == "transform"
                else ErrorWhere.METRIC,
            ),
        ),
        TrialEvent(
            trial_hash=trial.spec_hash,
            event_seq=3,
            event_id="evt_3",
            event_type=TrialEventType.TRIAL_COMPLETED,
            payload={"status": "ok"},
        ),
    ]
    for event in seed_events:
        event_repo.append_event(event)

    overlay_kwargs = (
        {"transform_hash": transform_hash}
        if failed_stage == "transform"
        else {"evaluation_hash": evaluation_hash}
    )
    record = projection_handler.on_trial_completed(trial.spec_hash, **overlay_kwargs)

    assert record.status == RecordStatus.ERROR
    assert support.should_skip_trial(trial.spec_hash, **overlay_kwargs) is False
