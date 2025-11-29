"""Registry for experiment builders."""

from __future__ import annotations

from typing import Callable

from themis.config import schema
from themis.experiment import orchestrator

ExperimentBuilder = Callable[
    [schema.ExperimentConfig], orchestrator.ExperimentOrchestrator
]

_EXPERIMENT_BUILDERS: dict[str, ExperimentBuilder] = {}


def register_experiment_builder(task: str) -> Callable[[ExperimentBuilder], ExperimentBuilder]:
    """Decorator to register an experiment builder for a specific task."""

    def decorator(builder: ExperimentBuilder) -> ExperimentBuilder:
        _EXPERIMENT_BUILDERS[task] = builder
        return builder

    return decorator


def get_experiment_builder(task: str) -> ExperimentBuilder:
    """Get the experiment builder for a specific task."""
    if task not in _EXPERIMENT_BUILDERS:
        raise ValueError(
            f"No experiment builder registered for task '{task}'. "
            f"Available tasks: {', '.join(sorted(_EXPERIMENT_BUILDERS.keys()))}"
        )
    return _EXPERIMENT_BUILDERS[task]
