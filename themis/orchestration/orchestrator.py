"""Top-level orchestration facade for planning, execution, and projections."""

from __future__ import annotations

from pathlib import Path
import tomllib
from typing import Any

from pydantic import ValidationError

from themis.errors.exceptions import SpecValidationError
from themis.orchestration.executor import TrialExecutor
from themis.orchestration.projection_handler import ProjectionHandler
from themis.orchestration.trial_planner import TrialPlanner
from themis.orchestration.trial_runner import TrialRunner
from themis.registry.plugin_registry import PluginRegistry
from themis.runtime import ExperimentResult
from themis.specs.experiment import (
    ExecutionPolicySpec,
    ExperimentSpec,
    ProjectSpec,
    RuntimeContext,
)
from themis.storage.artifact_store import ArtifactStore
from themis.storage.event_repo import SqliteEventRepository
from themis.storage.observability import SqliteObservabilityStore
from themis.storage.projection_repo import SqliteProjectionRepository
from themis.storage.sqlite_schema import DatabaseManager
from themis.telemetry.bus import TelemetryBus
from themis.types.enums import ErrorCode
from themis.types.json_validation import format_validation_error


def _validate_parallel_candidates(parallel_candidates: int) -> None:
    if parallel_candidates < 1:
        raise SpecValidationError(
            code=ErrorCode.SCHEMA_MISMATCH,
            message="parallel_candidates must be >= 1.",
        )


class Orchestrator:
    """Main v2 facade that plans, executes, projects, and returns experiment results."""

    @classmethod
    def from_project_spec(
        cls,
        project: ProjectSpec,
        *,
        registry: PluginRegistry | None = None,
        dataset_loader: Any = None,
        parallel_candidates: int = 5,
        telemetry_bus: TelemetryBus | None = None,
    ) -> Orchestrator:
        """Construct an orchestrator from a validated project specification."""
        _validate_parallel_candidates(parallel_candidates)
        root_dir = Path(project.storage.root_dir)
        root_dir.mkdir(parents=True, exist_ok=True)

        db_manager = DatabaseManager(f"sqlite:///{root_dir / 'themis.sqlite3'}")
        db_manager.initialize()
        observability_store = SqliteObservabilityStore(db_manager)
        artifact_store = (
            ArtifactStore(root_dir / "artifacts", manager=db_manager)
            if project.storage.compression == "zstd"
            else None
        )
        return cls(
            registry or PluginRegistry(),
            db_manager,
            dataset_loader=dataset_loader,
            artifact_store=artifact_store,
            execution_policy=project.execution_policy,
            parallel_candidates=parallel_candidates,
            project_seed=project.global_seed,
            store_item_payloads=project.storage.store_item_payloads,
            observability_store=observability_store,
            telemetry_bus=telemetry_bus,
        )

    @classmethod
    def from_project_file(
        cls,
        path: str,
        *,
        registry: PluginRegistry | None = None,
        dataset_loader: Any = None,
        parallel_candidates: int = 5,
        telemetry_bus: TelemetryBus | None = None,
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
        )

    def __init__(
        self,
        registry: PluginRegistry,
        db_manager: DatabaseManager,
        dataset_loader: Any = None,
        artifact_store: ArtifactStore | None = None,
        max_retries: int = 3,
        execution_policy: ExecutionPolicySpec | None = None,
        parallel_candidates: int = 5,
        project_seed: int | None = None,
        store_item_payloads: bool = True,
        observability_store: SqliteObservabilityStore | None = None,
        telemetry_bus: TelemetryBus | None = None,
    ):
        _validate_parallel_candidates(parallel_candidates)
        self.registry = registry
        self.db_manager = db_manager
        self.dataset_loader = dataset_loader

        self.event_repo = SqliteEventRepository(self.db_manager)
        self.observability_store = observability_store or SqliteObservabilityStore(
            self.db_manager
        )
        self.projection_repo = SqliteProjectionRepository(
            self.db_manager,
            artifact_store=artifact_store,
            observability_store=self.observability_store,
        )
        self.execution_policy = execution_policy or ExecutionPolicySpec(
            max_retries=max_retries
        )
        self.telemetry_bus = telemetry_bus

        self.planner = TrialPlanner(
            dataset_loader=self.dataset_loader, registry=self.registry
        )
        self.runner = TrialRunner(
            registry=self.registry,
            event_repo=self.event_repo,
            artifact_store=artifact_store,
            max_retries=self.execution_policy.max_retries,
            retry_backoff_factor=self.execution_policy.retry_backoff_factor,
            parallel_candidates=parallel_candidates,
            project_seed=project_seed,
            store_item_payloads=store_item_payloads,
            telemetry_bus=self.telemetry_bus,
        )
        self.projection_handler = ProjectionHandler(
            event_repo=self.event_repo,
            projection_repo=self.projection_repo,
        )
        self.executor = TrialExecutor(
            runner=self.runner,
            projection_repo=self.projection_repo,
            event_repo=self.event_repo,
            projection_handler=self.projection_handler,
            execution_policy=self.execution_policy,
            telemetry_bus=self.telemetry_bus,
        )

    def run(
        self,
        experiment: ExperimentSpec,
        *,
        runtime: RuntimeContext | None = None,
        eval_revision: str = "latest",
    ) -> ExperimentResult:
        """Plan an experiment, execute all trials, and return the result facade."""
        trials = self.planner.plan_experiment(experiment)
        self.executor.execute_trials(
            trials,
            runtime,
            eval_revision=eval_revision,
        )
        return ExperimentResult(
            projection_repo=self.projection_repo,
            trial_hashes=[
                planned_trial.trial_spec.spec_hash for planned_trial in trials
            ],
            eval_revision=eval_revision,
        )
