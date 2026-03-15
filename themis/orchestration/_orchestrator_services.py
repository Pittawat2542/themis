"""Internal collaborator bundle for the public orchestrator facade."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from themis.contracts.protocols import (
    DatasetLoader,
    ObservabilityStore,
    ProjectionRepository,
    TrialEventRepository,
)
from themis.orchestration.executor import TrialExecutor, _ExecutionRunner
from themis.orchestration.projection_handler import ProjectionHandler
from themis.orchestration.trial_planner import TrialPlanner
from themis.orchestration.trial_runner import TrialRunner
from themis.registry.plugin_registry import PluginRegistry
from themis.specs.experiment import ExecutionPolicySpec
from themis.storage.artifact_store import ArtifactStore
from themis.storage.factory import StorageBundle
from themis.telemetry.bus import TelemetryBus


@dataclass(frozen=True, slots=True)
class OrchestratorServices:
    """Concrete collaborators backing one `Orchestrator` instance."""

    storage_bundle: StorageBundle
    event_repo: TrialEventRepository
    observability_store: ObservabilityStore
    projection_repo: ProjectionRepository
    planner: TrialPlanner
    runner: TrialRunner
    projection_handler: ProjectionHandler
    executor: TrialExecutor


def build_orchestrator_services(
    *,
    registry: PluginRegistry,
    storage_bundle: StorageBundle,
    dataset_loader: DatasetLoader | None,
    execution_policy: ExecutionPolicySpec,
    parallel_candidates: int,
    project_seed: int | None,
    store_item_payloads: bool,
    telemetry_bus: TelemetryBus | None,
) -> OrchestratorServices:
    """Build the concrete orchestration collaborators behind the public facade."""
    event_repo = storage_bundle.event_repo
    active_observability_store = storage_bundle.observability_store
    projection_repo = storage_bundle.projection_repo
    planner = TrialPlanner(
        dataset_loader=dataset_loader,
        registry=registry,
    )
    runner = TrialRunner(
        registry=registry,
        event_repo=event_repo,
        artifact_store=cast(ArtifactStore | None, storage_bundle.blob_store),
        max_retries=execution_policy.max_retries,
        retry_backoff_factor=execution_policy.retry_backoff_factor,
        parallel_candidates=parallel_candidates,
        project_seed=project_seed,
        store_item_payloads=store_item_payloads,
        telemetry_bus=telemetry_bus,
    )
    projection_handler = ProjectionHandler(
        event_repo=event_repo,
        projection_repo=projection_repo,
    )
    executor = TrialExecutor(
        runner=cast(_ExecutionRunner, runner),
        projection_repo=projection_repo,
        event_repo=event_repo,
        projection_handler=projection_handler,
        execution_policy=execution_policy,
        telemetry_bus=telemetry_bus,
    )
    return OrchestratorServices(
        storage_bundle=storage_bundle,
        event_repo=event_repo,
        observability_store=active_observability_store,
        projection_repo=projection_repo,
        planner=planner,
        runner=runner,
        projection_handler=projection_handler,
        executor=executor,
    )
