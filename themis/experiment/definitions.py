"""Shared experiment definitions used by the builder."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from themis.core import entities as core_entities


@dataclass
class ModelBinding:
    spec: core_entities.ModelSpec
    provider_name: str
    provider_options: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentDefinition:
    templates: Sequence
    sampling_parameters: Sequence[core_entities.SamplingConfig]
    model_bindings: Sequence[ModelBinding]
    dataset_id_field: str = "id"
    reference_field: str | None = "expected"
    metadata_fields: Sequence[str] = field(default_factory=tuple)
    context_builder: Callable[[dict[str, Any]], dict[str, Any]] | None = None


@dataclass
class BuiltExperiment:
    plan: Any
    runner: Any
    pipeline: Any
    storage: Any
    router: Any
    orchestrator: Any


__all__ = [
    "ModelBinding",
    "ExperimentDefinition",
    "BuiltExperiment",
]
