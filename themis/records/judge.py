from pydantic import Field

from themis.records.base import RecordBase
from themis.records.inference import InferenceRecord
from themis.specs.experiment import PromptMessage
from themis.specs.foundational import JudgeInferenceSpec


class JudgeCallRecord(RecordBase):
    """
    Records one judge-model inference call issued by a metric.
    """

    metric_id: str = Field(
        ..., description="The ID of the metric that requested the judgement."
    )
    judge_spec: JudgeInferenceSpec = Field(
        ..., description="The requested configuration of the judge model."
    )
    rendered_prompt: list[PromptMessage] = Field(
        default_factory=list,
        description="Provider-neutral prompt payload used for the judge call.",
    )
    inference: InferenceRecord = Field(
        ..., description="The raw execution trace of the judge."
    )


class JudgeAuditTrail(RecordBase):
    """
    Audit trail for judge-dependent metrics.

    Trails are linked to candidate hashes and stored independently from the main
    `EvaluationRecord` so expensive judge traces stay inspectable without
    bloating scalar score rows.
    """

    candidate_hash: str = Field(
        ..., description="Hash of the CandidateRecord this trail belongs to."
    )
    judge_calls: list[JudgeCallRecord] = Field(
        default_factory=list, description="All judge calls made for this candidate."
    )
