from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from themis.types.enums import ErrorCode, ErrorWhere
from themis.types.hashable import HashableMixin
from themis.types.json_types import JSONDict


class ErrorRecord(HashableMixin, BaseModel):
    """
    Structured error payload used in events, projections, and reports.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: ErrorCode
    message: str
    retryable: bool = Field(json_schema_extra={"exclude_from_hash": True})
    where: ErrorWhere

    # Optional fields not included in the fingerprint
    type: str | None = Field(
        default=None, json_schema_extra={"exclude_from_hash": True}
    )
    details: JSONDict = Field(
        default_factory=dict, json_schema_extra={"exclude_from_hash": True}
    )
    cause_chain: list["ErrorRecord"] = Field(
        default_factory=list, json_schema_extra={"exclude_from_hash": True}
    )

    @property
    def fingerprint(self) -> str:
        """
        Deterministic 12-character hash based only on `(code, message, where)`.

        The fingerprint intentionally ignores retry metadata and details so
        repeated failures can be grouped for reporting and circuit breaking.
        """
        return self.compute_hash(short=True)


ErrorRecord.model_rebuild()
