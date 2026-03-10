from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field, model_validator

from themis.records.error import ErrorRecord
from themis.records.provenance import ProvenanceRecord
from themis.types.enums import RecordStatus


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


class RecordBase(BaseModel):
    """
    Shared immutable base for persisted output artifacts.

    Records always carry a stable `spec_hash`, terminal `status`, timestamps, and
    optional structured error/provenance payloads.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    spec_hash: str = Field(
        ...,
        description="The shortened canonical hash of the Spec that produced this Record.",
    )

    status: RecordStatus = Field(
        default=RecordStatus.OK,
        description="The terminal status of this record's execution.",
    )

    created_at: datetime = Field(
        default_factory=_now_utc,
        description="UTC timestamp of when this record was generated.",
    )

    error: ErrorRecord | None = Field(
        default=None, description="Structured error details if status is ERROR."
    )

    provenance: ProvenanceRecord | None = Field(
        default=None, description="Hydrated execution provenance for this record."
    )

    @model_validator(mode="after")
    def validate_error_status(self) -> "RecordBase":
        """Enforces that an error record must be attached if status is ERROR."""
        if self.status == RecordStatus.ERROR and self.error is None:
            raise ValueError(
                "Record status is ERROR, but no error details were provided."
            )
        if self.status != RecordStatus.ERROR and self.error is not None:
            raise ValueError(
                f"Record has an error attached, but status is {self.status.name}."
            )
        return self
