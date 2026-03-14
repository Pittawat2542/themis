"""Internal services for run planning, manifest state, and stage handoffs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import hashlib
import json

from themis.contracts.protocols import (
    ProjectionHandler as ProjectionHandlerProtocol,
    ProjectionRepository,
    TrialEventRepository,
)
from themis.errors import SpecValidationError
from themis.orchestration.candidate_pipeline import candidate_hash_for_index
from themis.orchestration.importer import CandidateImporter, StageResultImporter
from themis.orchestration.run_manifest import (
    CostEstimate,
    EvaluationBundleItem,
    EvaluationWorkBundle,
    GenerationBundleItem,
    GenerationWorkBundle,
    RunDiff,
    RunHandle,
    RunManifest,
    RunStatus,
    StageWorkItem,
    WorkItemStatus,
)
from themis.orchestration.task_resolution import resolve_task_stages
from themis.orchestration.trial_planner import PlannedTrial, TrialPlanner
from themis.records.trial import TrialRecord
from themis.runtime import ExperimentResult
from themis.specs.experiment import ExperimentSpec, ProjectSpec, RuntimeContext
from themis.storage.run_manifest_repo import RunManifestRepository
from themis.types.enums import ErrorCode, RecordStatus
from themis.types.events import (
    EvaluationCompletedEventMetadata,
    ExtractionCompletedEventMetadata,
    TrialEvent,
    TrialEventType,
)

RunExecutor = Callable[[ExperimentSpec, RuntimeContext | None], ExperimentResult]


def generation_trials(planned_trials: Sequence[PlannedTrial]) -> list[PlannedTrial]:
    return [
        planned_trial
        for planned_trial in planned_trials
        if planned_trial.trial_spec.task.generation is not None
    ]


def transform_trials(planned_trials: Sequence[PlannedTrial]) -> list[PlannedTrial]:
    return [
        planned_trial
        for planned_trial in planned_trials
        if planned_trial.trial_spec.task.output_transforms
    ]


def evaluation_trials(planned_trials: Sequence[PlannedTrial]) -> list[PlannedTrial]:
    return [
        planned_trial
        for planned_trial in planned_trials
        if planned_trial.trial_spec.task.evaluations
    ]


def collect_transform_hashes(planned_trials: Sequence[PlannedTrial]) -> list[str]:
    hashes: list[str] = []
    for planned_trial in planned_trials:
        resolved = resolve_task_stages(planned_trial.trial_spec.task)
        for transform in resolved.output_transforms:
            if transform.transform_hash not in hashes:
                hashes.append(transform.transform_hash)
    return hashes


def collect_evaluation_hashes(planned_trials: Sequence[PlannedTrial]) -> list[str]:
    hashes: list[str] = []
    for planned_trial in planned_trials:
        resolved = resolve_task_stages(planned_trial.trial_spec.task)
        for evaluation in resolved.evaluations:
            if evaluation.evaluation_hash not in hashes:
                hashes.append(evaluation.evaluation_hash)
    return hashes


@dataclass(frozen=True, slots=True)
class RunPlanningService:
    """Owns manifest planning, run-state derivation, and handoff bundle assembly."""

    planner: TrialPlanner
    event_repo: TrialEventRepository
    projection_repo: ProjectionRepository
    projection_handler: ProjectionHandlerProtocol
    manifest_repo: RunManifestRepository
    project_spec: ProjectSpec | None = None

    @property
    def backend_kind(self) -> str:
        if self.project_spec is None:
            return "local"
        return self.project_spec.execution_backend.kind

    def plan(self, experiment: ExperimentSpec) -> RunManifest:
        planned_trials = self.planner.plan_experiment(experiment)
        manifest = self.build_manifest(experiment, planned_trials)
        self.manifest_repo.save_manifest(manifest)
        return manifest

    def diff_specs(
        self,
        baseline: ExperimentSpec,
        treatment: ExperimentSpec,
    ) -> RunDiff:
        baseline_manifest = self.build_manifest(
            baseline,
            self.planner.plan_experiment(baseline),
        )
        treatment_manifest = self.build_manifest(
            treatment,
            self.planner.plan_experiment(treatment),
        )
        baseline_payload = baseline.model_dump(mode="json")
        treatment_payload = treatment.model_dump(mode="json")
        changed_experiment_fields = sorted(
            key
            for key in set(baseline_payload) | set(treatment_payload)
            if baseline_payload.get(key) != treatment_payload.get(key)
        )
        project_hash = (
            self.project_spec.spec_hash if self.project_spec is not None else None
        )
        return RunDiff(
            project_hash_before=project_hash,
            project_hash_after=project_hash,
            experiment_hash_before=baseline.spec_hash,
            experiment_hash_after=treatment.spec_hash,
            changed_project_fields=[],
            changed_experiment_fields=changed_experiment_fields,
            added_trial_hashes=sorted(
                set(treatment_manifest.trial_hashes)
                - set(baseline_manifest.trial_hashes)
            ),
            removed_trial_hashes=sorted(
                set(baseline_manifest.trial_hashes)
                - set(treatment_manifest.trial_hashes)
            ),
            added_transform_hashes=sorted(
                set(treatment_manifest.transform_hashes)
                - set(baseline_manifest.transform_hashes)
            ),
            removed_transform_hashes=sorted(
                set(baseline_manifest.transform_hashes)
                - set(treatment_manifest.transform_hashes)
            ),
            added_evaluation_hashes=sorted(
                set(treatment_manifest.evaluation_hashes)
                - set(baseline_manifest.evaluation_hashes)
            ),
            removed_evaluation_hashes=sorted(
                set(baseline_manifest.evaluation_hashes)
                - set(treatment_manifest.evaluation_hashes)
            ),
        )

    def submit(
        self,
        experiment: ExperimentSpec,
        *,
        runtime: RuntimeContext | None,
        execute_run: RunExecutor,
    ) -> RunHandle:
        manifest = self.plan(experiment)
        if self.backend_kind == "local":
            execute_run(experiment, runtime)
            manifest = self.plan(experiment)
        return self.run_handle_from_manifest(manifest)

    def resume(
        self,
        run_id: str,
        *,
        runtime: RuntimeContext | None,
        execute_run: RunExecutor,
    ) -> RunHandle | ExperimentResult:
        stored_manifest = self.manifest_repo.get_manifest(run_id)
        if stored_manifest is None:
            raise SpecValidationError(
                code=ErrorCode.SCHEMA_MISMATCH,
                message=f"Unknown run_id '{run_id}'.",
            )
        if (
            self.project_spec is not None
            and stored_manifest.project_spec is not None
            and self.project_spec.spec_hash != stored_manifest.project_spec.spec_hash
        ):
            raise SpecValidationError(
                code=ErrorCode.SCHEMA_MISMATCH,
                message=(
                    "Cannot resume a run planned under a different ProjectSpec. "
                    f"Expected '{stored_manifest.project_spec.spec_hash}', got "
                    f"'{self.project_spec.spec_hash}'."
                ),
            )

        manifest = self.plan(stored_manifest.experiment_spec)
        if self.backend_kind == "local" and any(
            item.status == WorkItemStatus.PENDING for item in manifest.work_items
        ):
            execute_run(stored_manifest.experiment_spec, runtime)
            manifest = self.plan(stored_manifest.experiment_spec)

        handle = self.run_handle_from_manifest(manifest)
        if handle.pending_work_items == 0:
            return self.result_from_manifest(manifest)
        return handle

    def estimate(self, experiment: ExperimentSpec) -> CostEstimate:
        planned_trials = self.planner.plan_experiment(experiment)
        manifest = self.build_manifest(experiment, planned_trials)
        prompt_char_budget = 0
        completion_token_budget = 0
        for planned_trial in planned_trials:
            prompt_chars = sum(
                len(message.content)
                for message in planned_trial.trial_spec.prompt.messages
            )
            dataset_chars = len(
                json.dumps(
                    planned_trial.dataset_context.payload,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            candidate_count = planned_trial.trial_spec.candidate_count
            prompt_char_budget += (prompt_chars + dataset_chars) * candidate_count
            if planned_trial.trial_spec.task.generation is not None:
                completion_token_budget += (
                    planned_trial.trial_spec.params.max_tokens * candidate_count
                )
        estimated_prompt_tokens = max(1, (prompt_char_budget + 3) // 4)
        work_items_by_stage = {
            stage: sum(1 for item in manifest.work_items if item.stage == stage)
            for stage in ("generation", "transform", "evaluation")
        }
        return CostEstimate(
            run_id=manifest.run_id,
            backend_kind=manifest.backend_kind,
            total_work_items=len(manifest.work_items),
            work_items_by_stage=work_items_by_stage,
            estimated_prompt_tokens=estimated_prompt_tokens,
            estimated_completion_tokens=completion_token_budget,
            estimated_total_tokens=estimated_prompt_tokens + completion_token_budget,
            estimated_total_cost=None,
            notes=[
                "Best-effort heuristic only: prompt tokens are estimated from prompt templates and dataset payload size.",
                "Provider pricing is not configured, so estimated_total_cost is unavailable.",
            ],
        )

    def export_generation_bundle(
        self,
        experiment: ExperimentSpec,
    ) -> GenerationWorkBundle:
        planned_trials = self.planner.plan_experiment(experiment)
        manifest = self.build_manifest(experiment, planned_trials)
        self.manifest_repo.save_manifest(manifest)
        planned_by_hash = {
            planned_trial.trial_spec.spec_hash: planned_trial
            for planned_trial in planned_trials
        }
        items = [
            GenerationBundleItem(
                work_item_id=item.work_item_id,
                trial_hash=item.trial_hash,
                candidate_index=item.candidate_index,
                candidate_id=item.candidate_id,
                trial_spec=planned_by_hash[item.trial_hash].trial_spec,
                dataset_context=planned_by_hash[item.trial_hash].dataset_context,
            )
            for item in manifest.work_items
            if item.stage == "generation" and item.status == WorkItemStatus.PENDING
        ]
        return GenerationWorkBundle(manifest=manifest, items=items)

    def export_evaluation_bundle(
        self,
        experiment: ExperimentSpec,
    ) -> EvaluationWorkBundle:
        planned_trials = self.planner.plan_experiment(experiment)
        manifest = self.build_manifest(experiment, planned_trials)
        self.manifest_repo.save_manifest(manifest)
        planned_by_hash = {
            planned_trial.trial_spec.spec_hash: planned_trial
            for planned_trial in planned_trials
        }
        items: list[EvaluationBundleItem] = []
        for item in manifest.work_items:
            if item.stage != "evaluation" or item.status != WorkItemStatus.PENDING:
                continue
            planned_trial = planned_by_hash[item.trial_hash]
            trial_record = self._evaluation_source_trial(
                planned_trial,
                transform_hash=item.transform_hash,
            )
            if trial_record is None:
                raise SpecValidationError(
                    code=ErrorCode.SCHEMA_MISMATCH,
                    message=(
                        "Cannot export evaluation bundle before generation results "
                        f"exist for trial '{item.trial_hash}'."
                    ),
                )
            candidate = next(
                candidate
                for candidate in trial_record.candidates
                if (candidate.candidate_id or candidate.spec_hash) == item.candidate_id
            )
            items.append(
                EvaluationBundleItem(
                    work_item_id=item.work_item_id,
                    trial_hash=item.trial_hash,
                    candidate_index=item.candidate_index,
                    candidate_id=item.candidate_id,
                    transform_hash=item.transform_hash,
                    evaluation_hash=item.evaluation_hash or "",
                    trial_spec=planned_trial.trial_spec,
                    dataset_context=planned_trial.dataset_context,
                    candidate=candidate,
                )
            )
        return EvaluationWorkBundle(manifest=manifest, items=items)

    def build_result(
        self,
        planned_trials: Sequence[PlannedTrial],
        *,
        transform_hashes: Sequence[str] | None = None,
        evaluation_hashes: Sequence[str] | None = None,
    ) -> ExperimentResult:
        resolved_transform_hashes = list(transform_hashes or [])
        resolved_evaluation_hashes = list(evaluation_hashes or [])
        active_transform_hash = (
            resolved_transform_hashes[0]
            if len(resolved_transform_hashes) == 1 and not resolved_evaluation_hashes
            else None
        )
        active_evaluation_hash = (
            resolved_evaluation_hashes[0]
            if len(resolved_evaluation_hashes) == 1
            else None
        )
        return ExperimentResult(
            projection_repo=self.projection_repo,
            trial_hashes=[
                planned_trial.trial_spec.spec_hash for planned_trial in planned_trials
            ],
            transform_hashes=resolved_transform_hashes,
            evaluation_hashes=resolved_evaluation_hashes,
            active_transform_hash=active_transform_hash,
            active_evaluation_hash=active_evaluation_hash,
        )

    def build_manifest(
        self,
        experiment: ExperimentSpec,
        planned_trials: Sequence[PlannedTrial],
    ) -> RunManifest:
        event_cache: dict[tuple[str, str], list[TrialEvent]] = {}
        work_items: list[StageWorkItem] = []
        transform_hashes: list[str] = []
        evaluation_hashes: list[str] = []

        for planned_trial in planned_trials:
            trial = planned_trial.trial_spec
            resolved = resolve_task_stages(trial.task)
            for transform in resolved.output_transforms:
                if transform.transform_hash not in transform_hashes:
                    transform_hashes.append(transform.transform_hash)
            for evaluation in resolved.evaluations:
                if evaluation.evaluation_hash not in evaluation_hashes:
                    evaluation_hashes.append(evaluation.evaluation_hash)
            for candidate_index in range(trial.candidate_count):
                candidate_id = candidate_hash_for_index(trial, candidate_index)
                candidate_events = event_cache.setdefault(
                    (trial.spec_hash, candidate_id),
                    self.event_repo.get_events(
                        trial.spec_hash,
                        candidate_id=candidate_id,
                    ),
                )
                work_items.append(
                    StageWorkItem(
                        work_item_id=_work_item_id(
                            "generation",
                            trial_hash=trial.spec_hash,
                            candidate_id=candidate_id,
                        ),
                        stage="generation",
                        status=_generation_status(candidate_events),
                        trial_hash=trial.spec_hash,
                        candidate_index=candidate_index,
                        candidate_id=candidate_id,
                    )
                )
                for transform in resolved.output_transforms:
                    work_items.append(
                        StageWorkItem(
                            work_item_id=_work_item_id(
                                "transform",
                                trial_hash=trial.spec_hash,
                                candidate_id=candidate_id,
                                transform_hash=transform.transform_hash,
                            ),
                            stage="transform",
                            status=_transform_status(
                                candidate_events,
                                transform_hash=transform.transform_hash,
                            ),
                            trial_hash=trial.spec_hash,
                            candidate_index=candidate_index,
                            candidate_id=candidate_id,
                            transform_hash=transform.transform_hash,
                        )
                    )
                for evaluation in resolved.evaluations:
                    work_items.append(
                        StageWorkItem(
                            work_item_id=_work_item_id(
                                "evaluation",
                                trial_hash=trial.spec_hash,
                                candidate_id=candidate_id,
                                transform_hash=(
                                    evaluation.transform.transform_hash
                                    if evaluation.transform is not None
                                    else None
                                ),
                                evaluation_hash=evaluation.evaluation_hash,
                            ),
                            stage="evaluation",
                            status=_evaluation_status(
                                candidate_events,
                                evaluation_hash=evaluation.evaluation_hash,
                            ),
                            trial_hash=trial.spec_hash,
                            candidate_index=candidate_index,
                            candidate_id=candidate_id,
                            transform_hash=(
                                evaluation.transform.transform_hash
                                if evaluation.transform is not None
                                else None
                            ),
                            evaluation_hash=evaluation.evaluation_hash,
                        )
                    )

        trial_hashes = [
            planned_trial.trial_spec.spec_hash for planned_trial in planned_trials
        ]
        return RunManifest(
            run_id=_run_id(
                experiment=experiment,
                trial_hashes=trial_hashes,
                work_items=work_items,
                backend_kind=self.backend_kind,
                project_spec=self.project_spec,
            ),
            backend_kind=self.backend_kind,
            project_spec=self.project_spec,
            experiment_spec=experiment,
            trial_hashes=trial_hashes,
            transform_hashes=transform_hashes,
            evaluation_hashes=evaluation_hashes,
            work_items=work_items,
        )

    def result_from_manifest(self, manifest: RunManifest) -> ExperimentResult:
        completed_transform_hashes = sorted(
            {
                item.transform_hash
                for item in manifest.work_items
                if item.stage == "transform"
                and item.status == WorkItemStatus.COMPLETED
                and item.transform_hash is not None
            }
        )
        completed_evaluation_hashes = sorted(
            {
                item.evaluation_hash
                for item in manifest.work_items
                if item.stage == "evaluation"
                and item.status == WorkItemStatus.COMPLETED
                and item.evaluation_hash is not None
            }
        )
        return ExperimentResult(
            projection_repo=self.projection_repo,
            trial_hashes=manifest.trial_hashes,
            transform_hashes=completed_transform_hashes,
            evaluation_hashes=completed_evaluation_hashes,
            active_transform_hash=(
                completed_transform_hashes[0]
                if len(completed_transform_hashes) == 1
                and not completed_evaluation_hashes
                else None
            ),
            active_evaluation_hash=(
                completed_evaluation_hashes[0]
                if len(completed_evaluation_hashes) == 1
                else None
            ),
        )

    def run_handle_from_manifest(self, manifest: RunManifest) -> RunHandle:
        pending_work_items = sum(
            1 for item in manifest.work_items if item.status == WorkItemStatus.PENDING
        )
        completed_work_items = len(manifest.work_items) - pending_work_items
        if pending_work_items == 0:
            status = RunStatus.COMPLETED
        elif completed_work_items > 0:
            status = RunStatus.RUNNING
        else:
            status = RunStatus.PENDING
        warnings: list[str] = []
        if self.backend_kind != "local" and pending_work_items > 0:
            warnings.append(
                "This backend currently persists the run handle but relies on external workers or imports to complete pending work items."
            )
        return RunHandle(
            run_id=manifest.run_id,
            backend_kind=manifest.backend_kind,
            status=status,
            total_work_items=len(manifest.work_items),
            pending_work_items=pending_work_items,
            completed_work_items=completed_work_items,
            trial_hashes=manifest.trial_hashes,
            transform_hashes=manifest.transform_hashes,
            evaluation_hashes=manifest.evaluation_hashes,
            warnings=warnings,
        )

    def _evaluation_source_trial(
        self,
        planned_trial: PlannedTrial,
        *,
        transform_hash: str | None,
    ) -> TrialRecord | None:
        trial_hash = planned_trial.trial_spec.spec_hash
        if transform_hash is None:
            return self.projection_repo.get_trial_record(trial_hash)
        record = self.projection_repo.get_trial_record(
            trial_hash,
            transform_hash=transform_hash,
        )
        if record is not None:
            return record
        return self.projection_handler.on_trial_completed(
            trial_hash,
            transform_hash=transform_hash,
        )


@dataclass(frozen=True, slots=True)
class RunImportService:
    """Owns validation and import of externally produced stage results."""

    event_repo: TrialEventRepository
    projection_repo: ProjectionRepository
    projection_handler: ProjectionHandlerProtocol

    def import_candidates(self, trial_records: Sequence[TrialRecord]) -> list[str]:
        importer = CandidateImporter(
            event_repo=self.event_repo,
            projection_repo=self.projection_repo,
            projection_handler=self.projection_handler,
        )
        return importer.import_candidates(trial_records)

    def import_generation_results(
        self,
        bundle: GenerationWorkBundle,
        trial_records: Sequence[TrialRecord],
    ) -> None:
        self._validate_generation_import(bundle, trial_records)
        importer = CandidateImporter(
            event_repo=self.event_repo,
            projection_repo=self.projection_repo,
            projection_handler=self.projection_handler,
        )
        importer.import_candidates(trial_records)

    def import_evaluation_results(
        self,
        bundle: EvaluationWorkBundle,
        trial_records: Sequence[TrialRecord],
    ) -> None:
        self._validate_evaluation_import(bundle, trial_records)
        importer = StageResultImporter(
            event_repo=self.event_repo,
            projection_handler=self.projection_handler,
        )
        importer.import_evaluation_results(bundle.manifest, trial_records)

    def _validate_generation_import(
        self,
        bundle: GenerationWorkBundle,
        trial_records: Sequence[TrialRecord],
    ) -> None:
        expected_candidate_ids = {item.candidate_id for item in bundle.items}
        imported_candidate_ids = {
            candidate.candidate_id or candidate.spec_hash
            for trial_record in trial_records
            for candidate in trial_record.candidates
        }
        unexpected = imported_candidate_ids - expected_candidate_ids
        if unexpected:
            raise SpecValidationError(
                code=ErrorCode.SCHEMA_MISMATCH,
                message=f"Generation import contains unexpected candidates: {sorted(unexpected)}",
            )

    def _validate_evaluation_import(
        self,
        bundle: EvaluationWorkBundle,
        trial_records: Sequence[TrialRecord],
    ) -> None:
        expected_pairs = {
            (item.candidate_id, item.evaluation_hash) for item in bundle.items
        }
        imported_pairs: set[tuple[str, str]] = set()
        for trial_record in trial_records:
            for candidate in trial_record.candidates:
                candidate_id = candidate.candidate_id or candidate.spec_hash
                matching_items = [
                    item for item in bundle.items if item.candidate_id == candidate_id
                ]
                if candidate.evaluation is None:
                    raise SpecValidationError(
                        code=ErrorCode.SCHEMA_MISMATCH,
                        message=f"Evaluation import is missing evaluation payload for '{candidate_id}'.",
                    )
                if len(matching_items) != 1:
                    raise SpecValidationError(
                        code=ErrorCode.SCHEMA_MISMATCH,
                        message=(
                            "Evaluation import requires exactly one exported evaluation "
                            f"work item per candidate for '{candidate_id}'."
                        ),
                    )
                imported_pairs.add((candidate_id, matching_items[0].evaluation_hash))
        unexpected = imported_pairs - expected_pairs
        if unexpected:
            raise SpecValidationError(
                code=ErrorCode.SCHEMA_MISMATCH,
                message=f"Evaluation import contains unexpected work items: {sorted(unexpected)}",
            )


def _run_id(
    *,
    experiment: ExperimentSpec,
    trial_hashes: Sequence[str],
    work_items: Sequence[StageWorkItem],
    backend_kind: str,
    project_spec: ProjectSpec | None,
) -> str:
    payload = {
        "project_hash": project_spec.spec_hash if project_spec else None,
        "experiment_hash": experiment.spec_hash,
        "backend_kind": backend_kind,
        "trial_hashes": list(trial_hashes),
        "work_item_ids": [item.work_item_id for item in work_items],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"run_{digest[:12]}"


def _work_item_id(
    stage: str,
    *,
    trial_hash: str,
    candidate_id: str,
    transform_hash: str | None = None,
    evaluation_hash: str | None = None,
) -> str:
    payload = {
        "stage": stage,
        "trial_hash": trial_hash,
        "candidate_id": candidate_id,
        "transform_hash": transform_hash,
        "evaluation_hash": evaluation_hash,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]


def _generation_status(candidate_events: Sequence[TrialEvent]) -> WorkItemStatus:
    if any(
        event.event_type
        in {TrialEventType.CANDIDATE_COMPLETED, TrialEventType.CANDIDATE_FAILED}
        for event in candidate_events
    ):
        return WorkItemStatus.COMPLETED
    return WorkItemStatus.PENDING


def _transform_status(
    candidate_events: Sequence[TrialEvent],
    *,
    transform_hash: str,
) -> WorkItemStatus:
    for event in candidate_events:
        if event.event_type != TrialEventType.EXTRACTION_COMPLETED:
            continue
        metadata = event.metadata
        if (
            isinstance(metadata, ExtractionCompletedEventMetadata)
            and metadata.transform_hash == transform_hash
            and metadata.success is True
            and event.status != RecordStatus.ERROR
        ):
            return WorkItemStatus.COMPLETED
    return WorkItemStatus.PENDING


def _evaluation_status(
    candidate_events: Sequence[TrialEvent],
    *,
    evaluation_hash: str,
) -> WorkItemStatus:
    for event in candidate_events:
        if event.event_type != TrialEventType.EVALUATION_COMPLETED:
            continue
        metadata = event.metadata
        if (
            isinstance(metadata, EvaluationCompletedEventMetadata)
            and metadata.evaluation_hash == evaluation_hash
            and event.status != RecordStatus.ERROR
        ):
            return WorkItemStatus.COMPLETED
    return WorkItemStatus.PENDING
