"""Shared experiment definitions used by the builder."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Sequence

from themis.core import entities as core_entities

if TYPE_CHECKING:
    from themis.evaluation.pipelines.standard_pipeline import EvaluationPipeline
    from themis.experiment.orchestrator import ExperimentOrchestrator
    from themis.experiment.storage import ExperimentStorage
    from themis.generation.plan import GenerationPlan
    from themis.generation.runner import GenerationRunner
    from themis.interfaces import ModelProvider


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
    """Built experiment with all components assembled.

    Attributes:
        plan: Generation plan for expanding tasks from dataset samples
        runner: Generation runner for executing tasks via providers
        pipeline: Evaluation pipeline for scoring outputs
        storage: Optional experiment storage for caching and resumability
        router: Provider router for dispatching to correct LLM provider
        orchestrator: Main orchestrator coordinating generation and evaluation
    """

    plan: "GenerationPlan"
    runner: "GenerationRunner"
    pipeline: "EvaluationPipeline"
    storage: "ExperimentStorage | None"
    router: "ModelProvider"
    orchestrator: "ExperimentOrchestrator"


__all__ = [
    "ModelBinding",
    "ExperimentDefinition",
    "BuiltExperiment",
]
