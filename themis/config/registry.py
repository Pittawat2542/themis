"""Registry for experiment builders."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from themis.config import schema
    from themis.experiment import orchestrator

    ExperimentBuilder = Callable[
        [schema.ExperimentConfig], orchestrator.ExperimentOrchestrator
    ]

_EXPERIMENT_BUILDERS: dict[str, Callable] = {}


def register_experiment_builder(
    task: str,
) -> Callable:
    """Decorator to register an experiment builder for a specific task."""

    def decorator(builder: Callable) -> Callable:
        _EXPERIMENT_BUILDERS[task] = builder
        return builder

    return decorator


def get_experiment_builder(task: str) -> Callable:
    """Get the experiment builder for a specific task."""
    if task not in _EXPERIMENT_BUILDERS:
        from themis.exceptions import ConfigurationError

        raise ConfigurationError(
            f"No experiment builder registered for task '{task}'. "
            f"Available tasks: {', '.join(sorted(_EXPERIMENT_BUILDERS.keys()))}"
        )
    return _EXPERIMENT_BUILDERS[task]
