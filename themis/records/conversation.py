from datetime import datetime, timezone
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from themis.types.enums import PromptRole
from themis.types.json_types import JSONDict, JSONValueType


class MessagePayload(BaseModel):
    model_config = ConfigDict(frozen=True)
    content: str


class ToolCallPayload(BaseModel):
    model_config = ConfigDict(frozen=True)
    tool_name: str
    tool_arguments: JSONDict
    call_id: str


class ToolResultPayload(BaseModel):
    model_config = ConfigDict(frozen=True)
    call_id: str
    result: JSONValueType
    is_error: bool = False


class NodeEnterPayload(BaseModel):
    model_config = ConfigDict(frozen=True)
    node_id: str


class NodeExitPayload(BaseModel):
    model_config = ConfigDict(frozen=True)
    node_id: str


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


class BaseConversationEvent(BaseModel):
    """Shared fields for discriminated conversation event variants."""

    model_config = ConfigDict(frozen=True)

    role: PromptRole
    timestamp: datetime = Field(default_factory=_now_utc)
    event_index: int


class MessageEvent(BaseConversationEvent):
    kind: Literal["message"] = "message"
    payload: MessagePayload


class ToolCallEvent(BaseConversationEvent):
    kind: Literal["tool_call"] = "tool_call"
    payload: ToolCallPayload


class ToolResultEvent(BaseConversationEvent):
    kind: Literal["tool_result"] = "tool_result"
    payload: ToolResultPayload


class NodeEnterEvent(BaseConversationEvent):
    kind: Literal["node_enter"] = "node_enter"
    payload: NodeEnterPayload


class NodeExitEvent(BaseConversationEvent):
    kind: Literal["node_exit"] = "node_exit"
    payload: NodeExitPayload


ConversationEvent: TypeAlias = Annotated[
    MessageEvent | ToolCallEvent | ToolResultEvent | NodeEnterEvent | NodeExitEvent,
    Field(discriminator="kind"),
]


class Conversation(BaseModel):
    """Ordered sequence of discriminated conversation events for one candidate."""

    model_config = ConfigDict(frozen=True)

    events: list[ConversationEvent] = Field(default_factory=list)
    root_trace_id: str | None = None
    meta: JSONDict = Field(default_factory=dict)

    def with_event(self, event: ConversationEvent) -> "Conversation":
        """Returns a new Conversation with the event appended."""
        new_events = list(self.events)
        new_events.append(event)
        return self.model_copy(update={"events": new_events})

    def slice(self, start: int, end: int | None = None) -> "Conversation":
        """Returns a new Conversation with a slice of the events."""
        return self.model_copy(update={"events": self.events[start:end]})
