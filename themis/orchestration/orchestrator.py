"""Top-level orchestration facade for planning, execution, and projections."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Literal

from pydantic import ValidationError

from themis.benchmark.compiler import compile_benchmark
from themis.benchmark.specs import BenchmarkSpec
from themis.contracts.protocols import DatasetLoader
from themis.contracts.protocols import DatasetProvider
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
from themis.runtime import BenchmarkResult
from themis.runtime.experiment_result import ExperimentResult
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


@dataclass(frozen=True, slots=True)
class _NormalizedSourceSpec:
    source_kind: Literal["benchmark", "experiment"]
    source_spec: ExperimentSpec | BenchmarkSpec
    experiment_spec: ExperimentSpec
    benchmark_spec: BenchmarkSpec | None


class Orchestrator:
    """Main facade that plans, executes, projects, and returns experiment results."""

    @classmethod
    def from_project_spec(
        cls,
        project: ProjectSpec,
        *,
        registry: PluginRegistry | None = None,
        dataset_provider: DatasetProvider | None = None,
        dataset_loader: DatasetLoader | None = None,
        parallel_candidates: int = 5,
        telemetry_bus: TelemetryBus | None = None,
        storage_bundle: StorageBundle | None = None,
    ) -> Orchestrator:
        """Construct an orchestrator from a validated project specification."""
        return cls(
            registry or PluginRegistry(),
            storage_bundle or build_storage_bundle(project.storage),
            dataset_provider=dataset_provider,
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
        dataset_provider: DatasetProvider | None = None,
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
            dataset_provider=dataset_provider,
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
        dataset_provider: DatasetProvider | None = None,
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
        self.dataset_provider = dataset_provider
        self.dataset_loader = dataset_loader
        self.project_spec = project_spec
        self.execution_policy = execution_policy
        self.telemetry_bus = telemetry_bus
        self._services: OrchestratorServices = build_orchestrator_services(
            registry=self.registry,
            storage_bundle=storage_bundle,
            dataset_provider=self.dataset_provider,
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
            registry=self.registry,
        )
        self._run_imports = RunImportService(
            event_repo=self._services.event_repo,
            projection_repo=self._services.projection_repo,
            projection_handler=self._services.projection_handler,
        )

    def run(
        self,
        experiment: ExperimentSpec | BenchmarkSpec,
        *,
        runtime: RuntimeContext | None = None,
        progress: ProgressConfig | None = None,
    ) -> ExperimentResult | BenchmarkResult:
        """Execute generation, transforms, and evaluations for one experiment."""
        normalized = self._normalize_source_spec(experiment)
        planned_trials = self._services.planner.plan_experiment(
            normalized.experiment_spec
        )
        progress_tracker = self._build_progress_tracker(
            normalized.experiment_spec,
            planned_trials,
            progress=progress,
            allowed_stages={
                RunStage.GENERATION,
                RunStage.TRANSFORM,
                RunStage.EVALUATION,
            },
            benchmark_spec=normalized.benchmark_spec,
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

        result = self._run_planning.build_result(
            planned_trials,
            transform_hashes=collect_transform_hashes(pending_transform_trials),
            evaluation_hashes=collect_evaluation_hashes(pending_evaluation_trials),
        )
        return self._wrap_source_result(normalized, result)

    def run_benchmark(
        self,
        benchmark: BenchmarkSpec,
        *,
        runtime: RuntimeContext | None = None,
        progress: ProgressConfig | None = None,
    ) -> BenchmarkResult:
        """Compile and execute one benchmark specification."""
        result = self.run(benchmark, runtime=runtime, progress=progress)
        if not isinstance(result, BenchmarkResult):
            raise TypeError(
                "Orchestrator.run_benchmark expected run() to return a "
                f"BenchmarkResult, got {type(result).__name__}."
            )
        return result

    def generate(
        self,
        experiment: ExperimentSpec | BenchmarkSpec,
        *,
        runtime: RuntimeContext | None = None,
        progress: ProgressConfig | None = None,
    ) -> ExperimentResult | BenchmarkResult:
        """Execute only generation-stage work for one experiment."""
        normalized = self._normalize_source_spec(experiment)
        planned_trials = generation_trials(
            self._services.planner.plan_experiment(
                normalized.experiment_spec,
                required_stages={RunStage.GENERATION},
            )
        )
        progress_tracker = self._build_progress_tracker(
            normalized.experiment_spec,
            planned_trials,
            progress=progress,
            allowed_stages={RunStage.GENERATION},
            benchmark_spec=normalized.benchmark_spec,
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
        result = self._run_planning.build_result(planned_trials)
        return self._wrap_source_result(normalized, result)

    def transform(
        self,
        experiment: ExperimentSpec | BenchmarkSpec,
        *,
        runtime: RuntimeContext | None = None,
        progress: ProgressConfig | None = None,
    ) -> ExperimentResult | BenchmarkResult:
        """Execute output transforms against existing generation candidates."""
        normalized = self._normalize_source_spec(experiment)
        planned_trials = transform_trials(
            self._services.planner.plan_experiment(
                normalized.experiment_spec,
                required_stages={RunStage.TRANSFORM},
            )
        )
        transform_hashes = collect_transform_hashes(planned_trials)
        progress_tracker = self._build_progress_tracker(
            normalized.experiment_spec,
            planned_trials,
            progress=progress,
            allowed_stages={RunStage.TRANSFORM},
            benchmark_spec=normalized.benchmark_spec,
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
        result = self._run_planning.build_result(
            planned_trials,
            transform_hashes=transform_hashes,
        )
        return self._wrap_source_result(normalized, result)

    def evaluate(
        self,
        experiment: ExperimentSpec | BenchmarkSpec,
        *,
        runtime: RuntimeContext | None = None,
        progress: ProgressConfig | None = None,
    ) -> ExperimentResult | BenchmarkResult:
        """Execute evaluation-stage work, reusing generation when possible."""
        normalized = self._normalize_source_spec(experiment)
        planned_trials = evaluation_trials(
            self._services.planner.plan_experiment(
                normalized.experiment_spec,
                required_stages={RunStage.TRANSFORM, RunStage.EVALUATION},
            )
        )
        progress_tracker = self._build_progress_tracker(
            normalized.experiment_spec,
            planned_trials,
            progress=progress,
            allowed_stages={RunStage.TRANSFORM, RunStage.EVALUATION},
            benchmark_spec=normalized.benchmark_spec,
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
        result = self._run_planning.build_result(
            planned_trials,
            transform_hashes=collect_transform_hashes(planned_trials),
            evaluation_hashes=collect_evaluation_hashes(planned_trials),
        )
        return self._wrap_source_result(normalized, result)

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

    def plan(self, experiment: ExperimentSpec | BenchmarkSpec) -> RunManifest:
        """Build and persist a deterministic run manifest for one experiment."""
        normalized = self._normalize_source_spec(experiment)
        return self._run_planning.plan(
            normalized.experiment_spec,
            benchmark_spec=normalized.benchmark_spec,
        )

    def diff_specs(
        self,
        baseline: ExperimentSpec | BenchmarkSpec,
        treatment: ExperimentSpec | BenchmarkSpec,
    ) -> RunDiff:
        """Return a high-level diff between two experiment specifications."""
        normalized_baseline = self._normalize_source_spec(baseline)
        normalized_treatment = self._normalize_source_spec(treatment)
        return self._run_planning.diff_specs(
            normalized_baseline.experiment_spec,
            normalized_treatment.experiment_spec,
            baseline_source_spec=normalized_baseline.source_spec,
            treatment_source_spec=normalized_treatment.source_spec,
        )

    def submit(
        self,
        experiment: ExperimentSpec | BenchmarkSpec,
        *,
        runtime: RuntimeContext | None = None,
        progress: ProgressConfig | None = None,
    ) -> RunHandle:
        """Persist one run manifest and start execution if the backend is local."""
        normalized = self._normalize_source_spec(experiment)
        return self._run_planning.submit(
            normalized.experiment_spec,
            benchmark_spec=normalized.benchmark_spec,
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
    ) -> RunHandle | ExperimentResult | BenchmarkResult:
        """Refresh a persisted run and continue it when possible."""
        resumed = self._run_planning.resume(
            run_id,
            runtime=runtime,
            execute_run=lambda spec, runtime_context: self.run(
                spec,
                runtime=runtime_context,
                progress=progress,
            ),
        )
        if isinstance(resumed, ExperimentResult):
            manifest = self._run_planning.manifest_repo.get_manifest(run_id)
            if manifest is not None:
                return self._wrap_manifest_result(manifest, resumed)
        return resumed

    def get_run_progress(self, run_id: str) -> RunProgressSnapshot | None:
        """Return the persisted progress snapshot for one known run."""
        return self._run_planning.get_progress_snapshot(run_id)

    def estimate(self, experiment: ExperimentSpec | BenchmarkSpec) -> CostEstimate:
        """Return a best-effort work-item and token estimate for an experiment."""
        normalized = self._normalize_source_spec(experiment)
        return self._run_planning.estimate(normalized.experiment_spec)

    def _build_progress_tracker(
        self,
        experiment: ExperimentSpec,
        planned_trials: list[PlannedTrial],
        *,
        progress: ProgressConfig | None,
        allowed_stages: set[RunStage],
        benchmark_spec: BenchmarkSpec | None = None,
    ) -> RunProgressTracker | None:
        if progress is None:
            return None
        manifest = self._run_planning.plan_from_trials(
            experiment,
            planned_trials,
            benchmark_spec=benchmark_spec,
        )
        return RunProgressTracker(
            manifest,
            self._run_planning.manifest_repo,
            progress,
            frozenset(allowed_stages),
        )

    def export_generation_bundle(
        self,
        experiment: ExperimentSpec | BenchmarkSpec,
    ) -> GenerationWorkBundle:
        """Export only pending generation items for an experiment."""
        normalized = self._normalize_source_spec(experiment)
        return self._run_planning.export_generation_bundle(
            normalized.experiment_spec,
            benchmark_spec=normalized.benchmark_spec,
        )

    def import_generation_results(
        self,
        bundle: GenerationWorkBundle,
        trial_records: list[TrialRecord],
    ) -> ExperimentResult | BenchmarkResult:
        """Import externally generated results for a previously exported bundle."""
        self._run_imports.import_generation_results(bundle, trial_records)
        manifest = self.plan(self._manifest_source_spec(bundle.manifest))
        result = self._run_planning.result_from_manifest(manifest)
        return self._wrap_manifest_result(manifest, result)

    def export_evaluation_bundle(
        self,
        experiment: ExperimentSpec | BenchmarkSpec,
    ) -> EvaluationWorkBundle:
        """Export only pending evaluation items for an experiment."""
        normalized = self._normalize_source_spec(experiment)
        return self._run_planning.export_evaluation_bundle(
            normalized.experiment_spec,
            benchmark_spec=normalized.benchmark_spec,
        )

    def import_evaluation_results(
        self,
        bundle: EvaluationWorkBundle,
        trial_records: list[TrialRecord],
    ) -> ExperimentResult | BenchmarkResult:
        """Import externally evaluated results for a previously exported bundle."""
        self._run_imports.import_evaluation_results(bundle, trial_records)
        manifest = self.plan(self._manifest_source_spec(bundle.manifest))
        result = self._run_planning.result_from_manifest(manifest)
        return self._wrap_manifest_result(manifest, result)

    @property
    def db_manager(self) -> StorageConnectionManager:
        """Return the active backend-specific database manager for this orchestrator."""
        return self._services.storage_bundle.manager

    def _benchmark_result_from_public_spec(
        self,
        benchmark: BenchmarkSpec,
        result: ExperimentResult,
    ) -> BenchmarkResult:
        prompt_variant_ids = [
            prompt.id for prompt in benchmark.prompt_variants if prompt.id is not None
        ]
        return BenchmarkResult(
            projection_repo=result.projection_repo,
            trial_hashes=result.trial_hashes,
            transform_hashes=result.transform_hashes,
            evaluation_hashes=result.evaluation_hashes,
            active_transform_hash=result.active_transform_hash,
            active_evaluation_hash=result.active_evaluation_hash,
            benchmark_id=benchmark.benchmark_id,
            slice_ids=[slice_spec.slice_id for slice_spec in benchmark.slices],
            prompt_variant_ids=prompt_variant_ids,
        )

    def _wrap_source_result(
        self,
        normalized: _NormalizedSourceSpec,
        result: ExperimentResult,
    ) -> ExperimentResult | BenchmarkResult:
        if normalized.benchmark_spec is not None:
            return self._benchmark_result_from_public_spec(
                normalized.benchmark_spec,
                result,
            )
        return result

    def _wrap_manifest_result(
        self,
        manifest: RunManifest,
        result: ExperimentResult,
    ) -> ExperimentResult | BenchmarkResult:
        if manifest.benchmark_spec is not None:
            return self._benchmark_result_from_public_spec(
                manifest.benchmark_spec,
                result,
            )
        experiment = manifest.experiment_spec
        benchmark_ids = {
            task.benchmark_id
            for task in experiment.tasks
            if task.benchmark_id is not None
        }
        if not benchmark_ids:
            return result
        slice_ids = [
            task.slice_id or task.task_id
            for task in experiment.tasks
            if (task.slice_id or task.task_id) is not None
        ]
        prompt_variant_ids = [
            prompt.id for prompt in experiment.prompt_templates if prompt.id is not None
        ]
        return BenchmarkResult(
            projection_repo=result.projection_repo,
            trial_hashes=result.trial_hashes,
            transform_hashes=result.transform_hashes,
            evaluation_hashes=result.evaluation_hashes,
            active_transform_hash=result.active_transform_hash,
            active_evaluation_hash=result.active_evaluation_hash,
            benchmark_id=sorted(benchmark_ids)[0],
            slice_ids=slice_ids,
            prompt_variant_ids=prompt_variant_ids,
        )

    def _manifest_source_spec(
        self, manifest: RunManifest
    ) -> ExperimentSpec | BenchmarkSpec:
        if manifest.source_kind == "benchmark" and manifest.benchmark_spec is not None:
            return manifest.benchmark_spec
        return manifest.experiment_spec

    def _normalize_source_spec(
        self,
        source_spec: ExperimentSpec | BenchmarkSpec,
    ) -> _NormalizedSourceSpec:
        if isinstance(source_spec, BenchmarkSpec):
            return _NormalizedSourceSpec(
                source_kind="benchmark",
                source_spec=source_spec,
                experiment_spec=compile_benchmark(source_spec),
                benchmark_spec=source_spec,
            )
        return _NormalizedSourceSpec(
            source_kind="experiment",
            source_spec=source_spec,
            experiment_spec=source_spec,
            benchmark_spec=None,
        )
