"""Top-level orchestration facade for planning, execution, and projections."""

from __future__ import annotations

from pathlib import Path
import tomllib

from pydantic import ValidationError

from themis.contracts.protocols import DatasetLoader
from themis.errors import SpecValidationError
from themis.orchestration._orchestrator_services import (
    OrchestratorServices,
    build_orchestrator_services,
)
from themis.progress import ProgressConfig, RunProgressTracker
from themis.progress.models import RunProgressSnapshot
from themis.orchestration.run_manifest import (
    CostEstimate,
    EvaluationWorkBundle,
    GenerationWorkBundle,
    RunDiff,
    RunHandle,
    RunManifest,
)
from themis.orchestration.run_services import (
    RunImportService,
    RunPlanningService,
    collect_evaluation_hashes,
    collect_transform_hashes,
    evaluation_trials,
    generation_trials,
    transform_trials,
)
from themis.orchestration.trial_planner import PlannedTrial
from themis.records.trial import TrialRecord
from themis.registry.plugin_registry import PluginRegistry
from themis.runtime import ExperimentResult
from themis.specs.experiment import (
    ExecutionPolicySpec,
    ExperimentSpec,
    ProjectSpec,
    RuntimeContext,
)
from themis.storage.factory import StorageBundle, build_storage_bundle
from themis.storage._protocols import StorageConnectionManager
from themis.storage.run_manifest_repo import RunManifestRepository
from themis.telemetry.bus import TelemetryBus
from themis.types.enums import ErrorCode, RunStage
from themis.types.json_validation import format_validation_error


class Orchestrator:
    """Main facade that plans, executes, projects, and returns experiment results."""

    @classmethod
    def from_project_spec(
        cls,
        project: ProjectSpec,
        *,
        registry: PluginRegistry | None = None,
        dataset_loader: DatasetLoader | None = None,
        parallel_candidates: int = 5,
        telemetry_bus: TelemetryBus | None = None,
        storage_bundle: StorageBundle | None = None,
    ) -> Orchestrator:
        """Construct an orchestrator from a validated project specification."""
        return cls(
            registry or PluginRegistry(),
            storage_bundle or build_storage_bundle(project.storage),
            dataset_loader=dataset_loader,
            execution_policy=project.execution_policy,
            parallel_candidates=parallel_candidates,
            project_seed=project.global_seed,
            store_item_payloads=project.storage.store_item_payloads,
            telemetry_bus=telemetry_bus,
            project_spec=project,
            _allow_runtime_construction=True,
        )

    @classmethod
    def from_project_file(
        cls,
        path: str,
        *,
        registry: PluginRegistry | None = None,
        dataset_loader: DatasetLoader | None = None,
        parallel_candidates: int = 5,
        telemetry_bus: TelemetryBus | None = None,
        storage_bundle: StorageBundle | None = None,
    ) -> Orchestrator:
        """Load a project file, validate it, and build an orchestrator."""
        project_path = Path(path)
        try:
            if project_path.suffix == ".toml":
                with project_path.open("rb") as fh:
                    project_data = tomllib.load(fh)
                project = ProjectSpec.model_validate(project_data)
            elif project_path.suffix == ".json":
                project = ProjectSpec.model_validate_json(project_path.read_text())
            else:
                raise ValueError("Project files must use .toml or .json.")
        except tomllib.TOMLDecodeError as exc:
            raise SpecValidationError(
                code=ErrorCode.SCHEMA_MISMATCH,
                message=f"Failed to parse project config {project_path.name}: {exc}",
            ) from exc
        except ValidationError as exc:
            raise SpecValidationError(
                code=ErrorCode.SCHEMA_MISMATCH,
                message=f"Failed to parse project config {project_path.name}: {format_validation_error(exc)}",
            ) from exc
        return cls.from_project_spec(
            project,
            registry=registry,
            dataset_loader=dataset_loader,
            parallel_candidates=parallel_candidates,
            telemetry_bus=telemetry_bus,
            storage_bundle=storage_bundle,
        )

    def __init__(
        self,
        registry: PluginRegistry | None = None,
        storage_bundle: StorageBundle | None = None,
        *,
        dataset_loader: DatasetLoader | None = None,
        execution_policy: ExecutionPolicySpec | None = None,
        parallel_candidates: int = 5,
        project_seed: int | None = None,
        store_item_payloads: bool = True,
        telemetry_bus: TelemetryBus | None = None,
        project_spec: ProjectSpec | None = None,
        _allow_runtime_construction: bool = False,
    ) -> None:
        if not _allow_runtime_construction:
            raise TypeError(
                "Use Orchestrator.from_project_spec(...) or "
                "Orchestrator.from_project_file(...)."
            )
        if registry is None or storage_bundle is None or execution_policy is None:
            raise TypeError(
                "Internal orchestrator construction requires registry, "
                "storage_bundle, and execution_policy."
            )

        self.registry = registry
        self.dataset_loader = dataset_loader
        self.project_spec = project_spec
        self.execution_policy = execution_policy
        self.telemetry_bus = telemetry_bus
        self._services: OrchestratorServices = build_orchestrator_services(
            registry=self.registry,
            storage_bundle=storage_bundle,
            dataset_loader=self.dataset_loader,
            execution_policy=self.execution_policy,
            parallel_candidates=parallel_candidates,
            project_seed=project_seed,
            store_item_payloads=store_item_payloads,
            telemetry_bus=self.telemetry_bus,
        )
        manifest_repo = RunManifestRepository(storage_bundle.manager)
        self._run_planning = RunPlanningService(
            planner=self._services.planner,
            event_repo=self._services.event_repo,
            projection_repo=self._services.projection_repo,
            projection_handler=self._services.projection_handler,
            manifest_repo=manifest_repo,
            project_spec=self.project_spec,
        )
        self._run_imports = RunImportService(
            event_repo=self._services.event_repo,
            projection_repo=self._services.projection_repo,
            projection_handler=self._services.projection_handler,
        )

    def run(
        self,
        experiment: ExperimentSpec,
        *,
        runtime: RuntimeContext | None = None,
        progress: ProgressConfig | None = None,
    ) -> ExperimentResult:
        """Execute generation, transforms, and evaluations for one experiment."""
        planned_trials = self._services.planner.plan_experiment(experiment)
        progress_tracker = self._build_progress_tracker(
            experiment,
            planned_trials,
            progress=progress,
            allowed_stages={
                RunStage.GENERATION,
                RunStage.TRANSFORM,
                RunStage.EVALUATION,
            },
        )
        pending_generation_trials = generation_trials(planned_trials)
        pending_transform_trials = transform_trials(planned_trials)
        pending_evaluation_trials = evaluation_trials(planned_trials)

        if progress_tracker is not None:
            progress_tracker.start_run()
        try:
            if pending_generation_trials:
                self._services.executor.execute_generation_trials(
                    pending_generation_trials,
                    runtime,
                    progress_tracker=progress_tracker,
                )
            if pending_transform_trials:
                self._services.executor.execute_transforms(
                    pending_transform_trials,
                    runtime,
                    progress_tracker=progress_tracker,
                )
            if pending_evaluation_trials:
                self._services.executor.execute_evaluations(
                    pending_evaluation_trials,
                    runtime,
                    progress_tracker=progress_tracker,
                )
        finally:
            if progress_tracker is not None:
                progress_tracker.finish_run()

        return self._run_planning.build_result(
            planned_trials,
            transform_hashes=collect_transform_hashes(pending_transform_trials),
            evaluation_hashes=collect_evaluation_hashes(pending_evaluation_trials),
        )

    def generate(
        self,
        experiment: ExperimentSpec,
        *,
        runtime: RuntimeContext | None = None,
        progress: ProgressConfig | None = None,
    ) -> ExperimentResult:
        """Execute only generation-stage work for one experiment."""
        planned_trials = generation_trials(
            self._services.planner.plan_experiment(
                experiment,
                required_stages={RunStage.GENERATION},
            )
        )
        progress_tracker = self._build_progress_tracker(
            experiment,
            planned_trials,
            progress=progress,
            allowed_stages={RunStage.GENERATION},
        )
        if planned_trials:
            if progress_tracker is not None:
                progress_tracker.start_run()
            try:
                self._services.executor.execute_generation_trials(
                    planned_trials,
                    runtime,
                    progress_tracker=progress_tracker,
                )
            finally:
                if progress_tracker is not None:
                    progress_tracker.finish_run()
        return self._run_planning.build_result(planned_trials)

    def transform(
        self,
        experiment: ExperimentSpec,
        *,
        runtime: RuntimeContext | None = None,
        progress: ProgressConfig | None = None,
    ) -> ExperimentResult:
        """Execute output transforms against existing generation candidates."""
        planned_trials = transform_trials(
            self._services.planner.plan_experiment(
                experiment,
                required_stages={RunStage.TRANSFORM},
            )
        )
        transform_hashes = collect_transform_hashes(planned_trials)
        progress_tracker = self._build_progress_tracker(
            experiment,
            planned_trials,
            progress=progress,
            allowed_stages={RunStage.TRANSFORM},
        )
        if planned_trials:
            if progress_tracker is not None:
                progress_tracker.start_run()
            try:
                self._services.executor.execute_transforms(
                    planned_trials,
                    runtime,
                    progress_tracker=progress_tracker,
                )
            finally:
                if progress_tracker is not None:
                    progress_tracker.finish_run()
        return self._run_planning.build_result(
            planned_trials,
            transform_hashes=transform_hashes,
        )

    def evaluate(
        self,
        experiment: ExperimentSpec,
        *,
        runtime: RuntimeContext | None = None,
        progress: ProgressConfig | None = None,
    ) -> ExperimentResult:
        """Execute evaluation-stage work, reusing generation when possible."""
        planned_trials = evaluation_trials(
            self._services.planner.plan_experiment(
                experiment,
                required_stages={RunStage.TRANSFORM, RunStage.EVALUATION},
            )
        )
        progress_tracker = self._build_progress_tracker(
            experiment,
            planned_trials,
            progress=progress,
            allowed_stages={RunStage.TRANSFORM, RunStage.EVALUATION},
        )
        if planned_trials:
            if progress_tracker is not None:
                progress_tracker.start_run()
            try:
                self._services.executor.execute_transforms(
                    planned_trials,
                    runtime,
                    resume=True,
                    progress_tracker=progress_tracker,
                )
                self._services.executor.execute_evaluations(
                    planned_trials,
                    runtime,
                    resume=True,
                    progress_tracker=progress_tracker,
                )
            finally:
                if progress_tracker is not None:
                    progress_tracker.finish_run()
        return self._run_planning.build_result(
            planned_trials,
            transform_hashes=collect_transform_hashes(planned_trials),
            evaluation_hashes=collect_evaluation_hashes(planned_trials),
        )

    def import_candidates(
        self,
        trial_records: list[TrialRecord],
    ) -> ExperimentResult:
        """Import prebuilt generation artifacts into the current store."""
        trial_hashes = self._run_imports.import_candidates(trial_records)
        return ExperimentResult(
            projection_repo=self._services.projection_repo,
            trial_hashes=list(trial_hashes),
        )

    def plan(self, experiment: ExperimentSpec) -> RunManifest:
        """Build and persist a deterministic run manifest for one experiment."""
        return self._run_planning.plan(experiment)

    def diff_specs(
        self,
        baseline: ExperimentSpec,
        treatment: ExperimentSpec,
    ) -> RunDiff:
        """Return a high-level diff between two experiment specifications."""
        return self._run_planning.diff_specs(baseline, treatment)

    def submit(
        self,
        experiment: ExperimentSpec,
        *,
        runtime: RuntimeContext | None = None,
        progress: ProgressConfig | None = None,
    ) -> RunHandle:
        """Persist one run manifest and start execution if the backend is local."""
        return self._run_planning.submit(
            experiment,
            runtime=runtime,
            execute_run=lambda spec, runtime_context: self.run(
                spec,
                runtime=runtime_context,
                progress=progress,
            ),
        )

    def resume(
        self,
        run_id: str,
        *,
        runtime: RuntimeContext | None = None,
        progress: ProgressConfig | None = None,
    ) -> RunHandle | ExperimentResult:
        """Refresh a persisted run and continue it when possible."""
        return self._run_planning.resume(
            run_id,
            runtime=runtime,
            execute_run=lambda spec, runtime_context: self.run(
                spec,
                runtime=runtime_context,
                progress=progress,
            ),
        )

    def get_run_progress(self, run_id: str) -> RunProgressSnapshot | None:
        """Return the persisted progress snapshot for one known run."""
        return self._run_planning.get_progress_snapshot(run_id)

    def estimate(self, experiment: ExperimentSpec) -> CostEstimate:
        """Return a best-effort work-item and token estimate for an experiment."""
        return self._run_planning.estimate(experiment)

    def _build_progress_tracker(
        self,
        experiment: ExperimentSpec,
        planned_trials: list[PlannedTrial],
        *,
        progress: ProgressConfig | None,
        allowed_stages: set[RunStage],
    ) -> RunProgressTracker | None:
        manifest = self._run_planning.plan_from_trials(experiment, planned_trials)
        return RunProgressTracker(
            manifest,
            self._run_planning.manifest_repo,
            progress,
            frozenset(allowed_stages),
        )

    def export_generation_bundle(
        self,
        experiment: ExperimentSpec,
    ) -> GenerationWorkBundle:
        """Export only pending generation items for an experiment."""
        return self._run_planning.export_generation_bundle(experiment)

    def import_generation_results(
        self,
        bundle: GenerationWorkBundle,
        trial_records: list[TrialRecord],
    ) -> ExperimentResult:
        """Import externally generated results for a previously exported bundle."""
        self._run_imports.import_generation_results(bundle, trial_records)
        manifest = self.plan(bundle.manifest.experiment_spec)
        return self._run_planning.result_from_manifest(manifest)

    def export_evaluation_bundle(
        self,
        experiment: ExperimentSpec,
    ) -> EvaluationWorkBundle:
        """Export only pending evaluation items for an experiment."""
        return self._run_planning.export_evaluation_bundle(experiment)

    def import_evaluation_results(
        self,
        bundle: EvaluationWorkBundle,
        trial_records: list[TrialRecord],
    ) -> ExperimentResult:
        """Import externally evaluated results for a previously exported bundle."""
        self._run_imports.import_evaluation_results(bundle, trial_records)
        manifest = self.plan(bundle.manifest.experiment_spec)
        return self._run_planning.result_from_manifest(manifest)

    @property
    def db_manager(self) -> StorageConnectionManager:
        """Return the active backend-specific database manager for this orchestrator."""
        return self._services.storage_bundle.manager
