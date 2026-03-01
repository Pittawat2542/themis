"""Data schemas for the Themis server."""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class RunSummary(BaseModel):
    """Summary of an experiment run."""

    run_id: str
    experiment_id: str = "default"
    status: str
    num_samples: int = 0
    metrics: Dict[str, float] = Field(default_factory=dict)
    created_at: str | None = None


class RunDetail(BaseModel):
    """Detailed information about a run."""

    run_id: str
    experiment_id: str = "default"
    status: str
    num_samples: int
    metrics: Dict[str, float]
    samples: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ComparisonRequest(BaseModel):
    """Request to compare multiple runs."""

    run_ids: List[str]
    metrics: List[str] | None = None
    statistical_test: str = "bootstrap"
    alpha: float = 0.05


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str
    detail: str | None = None
