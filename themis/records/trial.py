from pydantic import Field

from themis.records.base import RecordBase
from themis.records.candidate import CandidateRecord
from themis.records.timeline import RecordTimeline
from themis.specs.experiment import TrialSpec


class TrialRecord(RecordBase):
    """
    Materialized projection for one trial and the candidates produced under it.
    """

    trial_spec: TrialSpec | None = Field(
        default=None,
        description="The concrete trial specification used to produce this record.",
    )

    timeline: RecordTimeline | None = Field(
        default=None,
        description="Optional trial-scoped timeline materialized from the event log.",
    )

    candidates: list[CandidateRecord] = Field(
        ..., description="List of all candidate samples generated for this trial."
    )
