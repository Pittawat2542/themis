from pydantic import BaseModel, ConfigDict

from themis.records.base import RecordBase
from themis.types.enums import InferenceStatus
from themis.records.conversation import Conversation
from themis.types.json_types import JSONDict


class TokenUsage(BaseModel):
    """Token usage tracking for InferenceRecord."""

    model_config = ConfigDict(frozen=True)

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int | None = None


class InferenceRecord(RecordBase):
    """Raw output emitted by an inference engine for one candidate attempt."""

    raw_text: str | None = None
    structured_output: JSONDict | None = None
    finish_reason: str | None = None

    # Latency tracking (in milliseconds)
    latency_ms: float | None = None
    time_to_first_token_ms: float | None = None
    provider_request_id: str | None = None

    token_usage: TokenUsage | None = None

    logprobs: list[JSONDict] | None = None
    reasoning_trace: str | None = None
    conversation: Conversation | None = None
    inference_status: InferenceStatus = InferenceStatus.OK
