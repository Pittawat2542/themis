"""Project-level definitions for grouping experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from themis.experiment.definitions import ExperimentDefinition


@dataclass(frozen=True)
class ProjectExperiment:
    """Metadata wrapper that pairs a name with an experiment definition."""

    name: str
    definition: ExperimentDefinition
    description: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class Project:
    """Container that organizes multiple experiments under a shared project."""

    project_id: str
    name: str
    description: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = field(default_factory=tuple)
    experiments: Sequence[ProjectExperiment] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self._experiment_index: dict[str, ProjectExperiment] = {}
        normalized: list[ProjectExperiment] = []
        for experiment in self.experiments:
            self._register_experiment(experiment)
            normalized.append(experiment)
        self.experiments = tuple(normalized)

    def add_experiment(self, experiment: ProjectExperiment) -> ProjectExperiment:
        """Attach an experiment to the project, enforcing unique names."""

        self._register_experiment(experiment)
        self.experiments = tuple(list(self.experiments) + [experiment])
        return experiment

    def create_experiment(
        self,
        *,
        name: str,
        definition: ExperimentDefinition,
        description: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        tags: Sequence[str] | None = None,
    ) -> ProjectExperiment:
        """Convenience helper to register an experiment from raw components."""

        experiment = ProjectExperiment(
            name=name,
            description=description,
            definition=definition,
            metadata=dict(metadata or {}),
            tags=tuple(tags or ()),
        )
        return self.add_experiment(experiment)

    def get_experiment(self, name: str) -> ProjectExperiment:
        try:
            return self._experiment_index[name]
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise KeyError(
                f"Experiment '{name}' not registered in project '{self.project_id}'"
            ) from exc

    def metadata_for_experiment(self, name: str) -> dict[str, Any]:
        """Merge project-level metadata with experiment-specific overrides."""

        combined = dict(self.metadata)
        combined.update(self.get_experiment(name).metadata)
        return combined

    def list_experiment_names(self) -> tuple[str, ...]:
        return tuple(self._experiment_index.keys())

    def _register_experiment(self, experiment: ProjectExperiment) -> None:
        if experiment.name in self._experiment_index:
            raise ValueError(
                f"Experiment '{experiment.name}' already registered "
                f"in project '{self.project_id}'"
            )
        self._experiment_index[experiment.name] = experiment


__all__ = [
    "Project",
    "ProjectExperiment",
]
