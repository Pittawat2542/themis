from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from themis.types.enums import IssueSeverity


class Issue(BaseModel):
    """Structured validation issue surfaced by compatibility checks."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str
    path: str
    message: str
    severity: IssueSeverity = IssueSeverity.ERROR
    suggestion: str | None = None
