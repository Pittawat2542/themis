"""Core types, enums, and JSON helpers shared across the runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING

from themis.types.enums import (
    CompressionCodec,
    DatasetSource,
    ErrorCode,
    ErrorWhere,
    InferenceStatus,
    IssueSeverity,
    PValueCorrection,
    RecordStatus,
    RecordType,
    ResponseFormat,
    SamplingKind,
    StorageBackend,
)
from themis.types.hashable import HashableMixin
from themis.types.issues import Issue
from themis.types.json_types import JSONScalar, JSONValueType, ParsedValue

if TYPE_CHECKING:
    from themis.types.events import ArtifactRole, TrialEventType

__all__ = [
    "ArtifactRole",
    "CompressionCodec",
    "DatasetSource",
    "ErrorCode",
    "ErrorWhere",
    "HashableMixin",
    "InferenceStatus",
    "Issue",
    "IssueSeverity",
    "JSONScalar",
    "JSONValueType",
    "PValueCorrection",
    "ParsedValue",
    "RecordStatus",
    "RecordType",
    "ResponseFormat",
    "SamplingKind",
    "StorageBackend",
    "TrialEventType",
]


def __getattr__(name: str) -> object:
    if name in {"ArtifactRole", "TrialEventType"}:
        from themis.types.events import ArtifactRole, TrialEventType

        return {"ArtifactRole": ArtifactRole, "TrialEventType": TrialEventType}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
