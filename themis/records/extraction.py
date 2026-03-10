from pydantic import Field

from themis.records.base import RecordBase
from themis.types.json_types import ParsedValue


class ExtractionRecord(RecordBase):
    """Result of one attempt to parse structured data from model output."""

    extractor_id: str = Field(
        ..., description="ID of the extractor that produced this record."
    )
    success: bool = Field(
        ..., description="True if extraction succeeded, False otherwise."
    )

    parsed_answer: ParsedValue | None = Field(
        default=None,
        description="The extracted content. Often a JSON-like dict, but could be a string.",
    )

    warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal warnings encountered during extraction.",
    )

    failure_reason: str | None = Field(
        default=None,
        description="Text description of why extraction failed, if success is False.",
    )
