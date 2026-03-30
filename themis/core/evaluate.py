"""Layer 1 convenience API for Themis experiments."""

from __future__ import annotations

from themis.core.config import EvaluationConfig, GenerationConfig, RuntimeConfig, StorageConfig
from themis.core.experiment import Experiment
from themis.core.models import Dataset
from themis.core.protocols import LifecycleSubscriber, TracingProvider
from themis.core.store import RunStore


def evaluate(
    *,
    generation: GenerationConfig,
    evaluation: EvaluationConfig,
    storage: StorageConfig,
    datasets: list[Dataset],
    runtime: RuntimeConfig | None = None,
    seeds: list[int] | None = None,
    environment_metadata: dict[str, str] | None = None,
    themis_version: str = "4.0.0rc1",
    python_version: str = "3.12",
    platform: str = "unknown",
    store: RunStore | None = None,
    subscribers: list[LifecycleSubscriber] | None = None,
    tracing_provider: TracingProvider | None = None,
):
    """Compile and run a Themis experiment through the Layer 1 API.

    Args:
        generation: Generation configuration for the run.
        evaluation: Evaluation configuration for the run.
        storage: Store configuration used to build a store when `store` is not passed.
        datasets: Datasets to evaluate.
        runtime: Optional runtime overrides for execution controls.
        seeds: Optional seed list captured in snapshot identity.
        environment_metadata: Optional non-secret metadata stored in provenance.
        themis_version: Version string persisted in provenance.
        python_version: Python version persisted in provenance.
        platform: Platform label persisted in provenance.
        store: Optional prebuilt store instance.
        subscribers: Optional lifecycle subscribers.
        tracing_provider: Optional tracing provider.

    Returns:
        The completed run result returned by `Experiment.run()`.
    """

    experiment = Experiment(
        generation=generation,
        evaluation=evaluation,
        storage=storage,
        runtime=runtime or RuntimeConfig(),
        datasets=datasets,
        seeds=list(seeds or []),
        environment_metadata=dict(environment_metadata or {}),
        themis_version=themis_version,
        python_version=python_version,
        platform=platform,
    )
    return experiment.run(
        store=store,
        subscribers=subscribers,
        tracing_provider=tracing_provider,
    )
