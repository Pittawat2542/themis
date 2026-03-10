from pydantic import BaseModel, ConfigDict, Field, model_validator

from themis.records.base import RecordBase
from themis.types.json_types import JSONDict


class MetricScore(BaseModel):
    """One metric output with a scalar value plus structured details."""

    model_config = ConfigDict(frozen=True)

    metric_id: str
    value: float
    details: JSONDict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None


class EvaluationRecord(RecordBase):
    """Aggregates all metric outputs computed for a single candidate."""

    metric_scores: list[MetricScore] = Field(default_factory=list)

    # Flattened scores for easy querying
    aggregate_scores: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def compute_aggregate_scores(cls, data: object) -> object:
        """Recomputes aggregate_scores dynamically based on metric_scores."""
        if isinstance(data, dict):
            metric_scores = data.get("metric_scores", [])
            aggregate_scores: dict[str, float] = {}
            for score in metric_scores:
                if isinstance(score, dict):
                    metric_id = score.get("metric_id")
                    value = score.get("value")
                else:
                    metric_id = score.metric_id
                    value = score.value
                if metric_id is not None and value is not None:
                    aggregate_scores[metric_id] = value
            data["aggregate_scores"] = aggregate_scores
        return data
