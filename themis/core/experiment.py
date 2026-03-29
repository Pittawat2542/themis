"""Minimal experiment placeholder for the first TDD slice."""

from pydantic import BaseModel, ConfigDict


class Experiment(BaseModel):
    """Placeholder experiment model until the full Phase 1 core is implemented."""

    model_config = ConfigDict(frozen=True, extra="forbid")
