from pydantic import Field

from themis._version import __version__
from themis.records.base import RecordBase
from themis.types.json_types import JSONDict


class ReportTable(RecordBase):
    """
    One materialized table in an evaluation report.
    """

    id: str = Field(
        ...,
        description="Unique identifier for the table (e.g. 'main_results', 'paired_comparisons').",
    )
    title: str = Field(..., description="Human-readable title.")
    description: str | None = Field(
        default=None, description="Optional description of the table's contents."
    )
    data: list[JSONDict] = Field(
        ..., description="The dataset rows as parsed dictionaries."
    )


class ReportMetadata(RecordBase):
    """Report-level metadata such as version info, spec hashes, and provenance."""

    spec_hashes: list[str] = Field(
        default_factory=list, description="Spec hashes included in the report."
    )
    themis_version: str = Field(
        default_factory=lambda: __version__,
        description="Themis evaluation framework version.",
    )
    extras: JSONDict = Field(default_factory=dict, description="Other custom metadata.")


class EvaluationReport(RecordBase):
    """Final assembled report containing multiple tables plus report metadata."""

    tables: list[ReportTable] = Field(default_factory=list)
    metadata: ReportMetadata = Field(..., description="Report metadata.")

    def get_table(self, table_id: str) -> ReportTable | None:
        for t in self.tables:
            if t.id == table_id:
                return t
        return None
