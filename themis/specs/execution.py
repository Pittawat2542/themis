"""Execution specification for vNext workflows."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionSpec:
    """Execution configuration for running experiments."""

    backend: object | None = None
    workers: int = 4
    max_retries: int = 3
    retry_initial_delay: float = 0.5
    retry_backoff_multiplier: float = 2.0
    retry_max_delay: float | None = 2.0

    def __post_init__(self) -> None:
        if self.workers < 1:
            raise ValueError("ExecutionSpec.workers must be >= 1.")
        if self.max_retries < 1:
            raise ValueError("ExecutionSpec.max_retries must be >= 1.")


__all__ = ["ExecutionSpec"]
