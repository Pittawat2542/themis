from __future__ import annotations

import pytest

from themis.orchestration.generation_execution import GenerationExecutionCoordinator
from themis.orchestration.overlay_execution import OverlayExecutionCoordinator
from themis.orchestration.run_manifest import WorkItemStatus
from themis.orchestration.runner_state import TrialExecutionSession
from themis.orchestration.task_resolution import resolve_task_stages
from themis.orchestration.trial_planner import PlannedTrial
from themis.progress.tracker import RunProgressTracker
from themis.records.candidate import CandidateRecord
from themis.records.provenance import ProvenanceRecord
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
from themis.types.enums import DatasetSource, RecordStatus
from themis.types.events import ArtifactRef


def _trial() -> TrialSpec:
    return TrialSpec(
        trial_id="trial-1",
        model=ModelSpec(model_id="mock-model", provider="mock"),
        task=TaskSpec(
            task_id="task-1",
            dataset=DatasetSpec(source=DatasetSource.MEMORY),
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
        item_id="item-1",
        prompt=PromptTemplateSpec(id="baseline", messages=[]),
        params=InferenceParamsSpec(),
        candidate_count=1,
    )


def _provenance() -> ProvenanceRecord:
    return ProvenanceRecord(
        themis_version="test",
        git_commit=None,
        python_version="3.12",
        platform="test",
        library_versions={},
        model_endpoint_meta={},
    )


def _session(trial: TrialSpec) -> TrialExecutionSession:
    return TrialExecutionSession(
        trial=trial,
        dataset_context=DataItemContext(item_id=trial.item_id, payload={}),
        base_runtime=RuntimeContext(),
        provenance=_provenance(),
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
        item_payload={},
        dataset_metadata={},
        event_seq=0,
    )


class _Tracker(RunProgressTracker):
    finished_statuses: list[WorkItemStatus]

    def __init__(self, *, finished_statuses: list[WorkItemStatus]) -> None:
        self.finished_statuses = finished_statuses

    def stage_started(self) -> None:
        return None

    def generation_work_item_id(self, trial_hash: str, candidate_index: int) -> str:
        return f"{trial_hash}:{candidate_index}"

    def transform_work_item_id(
        self,
        trial_hash: str,
        candidate_index: int,
        transform_hash: str,
    ) -> str:
        return f"{trial_hash}:{candidate_index}:{transform_hash}"

    def evaluation_work_item_id(
        self,
        trial_hash: str,
        candidate_index: int,
        evaluation_hash: str,
    ) -> str:
        return f"{trial_hash}:{candidate_index}:{evaluation_hash}"

    def mark_running(self, work_item_id: str) -> None:
        del work_item_id

    def mark_finished(
        self, work_item_id: str, *, status: WorkItemStatus, **kwargs
    ) -> None:
        del work_item_id, kwargs
        self.finished_statuses.append(status)


def test_generation_progress_marks_skipped_candidates_as_skipped() -> None:
    trial = _trial()
    planned = PlannedTrial(
        trial_spec=trial,
        dataset_context=DataItemContext(item_id=trial.item_id, payload={}),
    )
    tracker = _Tracker(finished_statuses=[])

    class Runner:
        def prepare_trial_session(self, *args, **kwargs):
            del args, kwargs
            return _session(trial)

        def run_generation_candidate(self, session, cand_index):
            del session, cand_index
            return CandidateRecord(
                spec_hash="candidate-1",
                candidate_id="candidate-1",
                sample_index=0,
                status=RecordStatus.SKIPPED,
            )

        def finalize_generation_trial(self, session, candidates):
            del session
            return TrialRecord(
                spec_hash=trial.spec_hash,
                trial_spec=trial,
                status=RecordStatus.OK,
                candidates=candidates,
            )

    coordinator = GenerationExecutionCoordinator(
        runner=Runner(),
        execution_policy=ExecutionPolicySpec(max_in_flight_work_items=1),
        should_skip_trial=lambda *args, **kwargs: False,
        iter_trials=lambda trials, dataset_context: [
            (planned.trial_spec, planned.dataset_context)
        ],
        materialize_projection=lambda *args, **kwargs: None,
        update_circuit_breaker=lambda trial_record: None,
        record_terminal_failure=lambda trial_spec, exc: None,
    )

    coordinator.execute_generation_trials(
        [planned],
        RuntimeContext(),
        resume=False,
        progress_tracker=tracker,
    )

    assert tracker.finished_statuses == [WorkItemStatus.SKIPPED]


@pytest.mark.parametrize(
    ("status", "mode"),
    [
        (RecordStatus.SKIPPED, "transform"),
        (RecordStatus.PARTIAL, "transform"),
        (RecordStatus.SKIPPED, "evaluation"),
    ],
)
def test_overlay_progress_marks_non_success_results_as_failed(
    status: RecordStatus,
    mode: str,
) -> None:
    trial = _trial()
    planned = PlannedTrial(
        trial_spec=trial,
        dataset_context=DataItemContext(item_id=trial.item_id, payload={}),
    )
    tracker = _Tracker(finished_statuses=[])
    candidate = CandidateRecord(
        spec_hash="candidate-1",
        candidate_id="candidate-1",
        sample_index=0,
        status=RecordStatus.OK,
    )
    record = TrialRecord(
        spec_hash=trial.spec_hash,
        trial_spec=trial,
        status=RecordStatus.OK,
        candidates=[candidate],
    )

    class Runner:
        def prepare_trial_session(self, *args, **kwargs):
            del args, kwargs
            return _session(trial)

        def run_output_transform(self, session, base_candidate, resolved_transform):
            del session, base_candidate, resolved_transform
            return candidate.model_copy(update={"status": status})

        def run_evaluation_candidate(self, session, base_candidate, evaluation):
            del session, base_candidate, evaluation
            return candidate.model_copy(update={"status": status})

    coordinator = OverlayExecutionCoordinator(
        runner=Runner(),
        projection_repo=type(
            "ProjectionRepo",
            (),
            {
                "get_trial_record": lambda self, trial_hash, **kwargs: record,
            },
        )(),
        execution_policy=ExecutionPolicySpec(max_in_flight_work_items=1),
        should_skip_trial=lambda *args, **kwargs: False,
        materialize_projection=lambda *args, **kwargs: record,
        iter_trials=lambda trials, dataset_context: [
            (planned.trial_spec, planned.dataset_context)
        ],
    )

    if mode == "transform":
        coordinator.execute_transforms(
            [planned],
            RuntimeContext(),
            resume=False,
            progress_tracker=tracker,
        )
    else:
        coordinator.execute_evaluations(
            [planned],
            RuntimeContext(),
            resume=False,
            progress_tracker=tracker,
        )

    assert tracker.finished_statuses == [WorkItemStatus.FAILED]
