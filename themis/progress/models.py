"""Public models for configuring and inspecting run progress reporting."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from themis.types.enums import RunStage


class ProgressRendererType(StrEnum):
    """Configured operator-facing progress renderer."""

    NONE = "none"
    RICH = "rich"
    LOG = "log"


class ProgressVerbosity(StrEnum):
    """Configured verbosity for progress output."""

    QUIET = "quiet"
    NORMAL = "normal"
    DEBUG = "debug"


class StageProgressSnapshot(BaseModel):
    """Aggregate progress counters for one orchestration stage."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    stage: RunStage
    total_items: int
    pending_items: int
    running_items: int
    completed_items: int
    failed_items: int
    skipped_items: int


class RunProgressSnapshot(BaseModel):
    """Operator-facing snapshot of run progress across all stages."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    backend_kind: str
    active_stage: RunStage | None = None
    processed_items: int
    remaining_items: int
    in_flight_items: int
    stage_counts: dict[RunStage, StageProgressSnapshot] = Field(default_factory=dict)
    started_at: datetime | None = None
    ended_at: datetime | None = None


class ProgressConfig(BaseModel):
    """Runtime-scoped progress configuration for orchestration entrypoints."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    enabled: bool = True
    renderer: ProgressRendererType | None = None
    verbosity: ProgressVerbosity = ProgressVerbosity.NORMAL
    callback: Callable[[RunProgressSnapshot], None] | None = Field(
        default=None,
        exclude=True,
        repr=False,
    )
