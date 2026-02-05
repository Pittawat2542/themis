"""Experiment specification for vNext workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ExperimentSpec:
    """Canonical experiment specification.

    This spec is the single source of truth for the experiment's
    dataset, prompt, model, sampling config, and evaluation pipeline.
    """

    dataset: object
    prompt: str
    model: str
    sampling: Mapping[str, Any] = field(default_factory=dict)
    provider_options: Mapping[str, Any] = field(default_factory=dict)
    pipeline: object | None = None
    run_id: str | None = None
    num_samples: int = 1
    max_records_in_memory: int | None = None
    dataset_id_field: str = "id"
    reference_field: str | None = "answer"
    metadata_fields: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.prompt:
            raise ValueError("ExperimentSpec.prompt must be a non-empty string.")
        if not self.model:
            raise ValueError("ExperimentSpec.model must be a non-empty string.")
        if self.pipeline is None:
            raise ValueError("ExperimentSpec.pipeline must be provided.")
        if self.num_samples < 1:
            raise ValueError("ExperimentSpec.num_samples must be >= 1.")
        if self.max_records_in_memory is not None and self.max_records_in_memory < 1:
            raise ValueError("ExperimentSpec.max_records_in_memory must be >= 1.")


__all__ = ["ExperimentSpec"]
