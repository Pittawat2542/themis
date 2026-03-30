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
    themis_version: str = "4.0.0a0",
    python_version: str = "3.12",
    platform: str = "unknown",
    store: RunStore | None = None,
    subscribers: list[LifecycleSubscriber] | None = None,
    tracing_provider: TracingProvider | None = None,
):
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
