"""Experiment specification for vNext workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


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
    pipeline: object | None = None
    run_id: str | None = None

    def __post_init__(self) -> None:
        if not self.prompt:
            raise ValueError("ExperimentSpec.prompt must be a non-empty string.")
        if not self.model:
            raise ValueError("ExperimentSpec.model must be a non-empty string.")
        if self.pipeline is None:
            raise ValueError("ExperimentSpec.pipeline must be provided.")


__all__ = ["ExperimentSpec"]
