from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ObservabilityRefs(BaseModel):
    """Projection-side observability links that are not hashed into domain records."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    langfuse_trace_id: str | None = None
    langfuse_url: str | None = None
    wandb_url: str | None = None
    extras: dict[str, str] = Field(default_factory=dict)
