"""Core types, enums, and JSON helpers shared across the runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from themis.types.enums import (
    ErrorCode,
    ErrorWhere,
    InferenceStatus,
    IssueSeverity,
    RecordStatus,
)
from themis.types.hashable import HashableMixin
from themis.types.issues import Issue
from themis.types.json_types import JSONScalar, JSONValueType, ParsedValue

if TYPE_CHECKING:
    from themis.types.events import ArtifactRole, TrialEventType

__all__ = [
    "ArtifactRole",
    "ErrorCode",
    "ErrorWhere",
    "HashableMixin",
    "InferenceStatus",
    "Issue",
    "IssueSeverity",
    "JSONScalar",
    "JSONValueType",
    "ParsedValue",
    "RecordStatus",
    "TrialEventType",
]


def __getattr__(name: str) -> Any:
    if name in {"ArtifactRole", "TrialEventType"}:
        from themis.types.events import ArtifactRole, TrialEventType

        return {"ArtifactRole": ArtifactRole, "TrialEventType": TrialEventType}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
