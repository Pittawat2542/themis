from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from themis.records.conversation import (
    Conversation,
    ConversationEvent,
    MessageEvent,
    MessagePayload,
    ToolCallPayload,
)


def test_conversation_accepts_message_event_variant() -> None:
    event = MessageEvent(
        role="assistant",
        payload=MessagePayload(content="42"),
        event_index=0,
    )
    conversation = Conversation(events=[event])

    assert conversation.events[0].kind == "message"
    assert conversation.events[0].payload.content == "42"


def test_conversation_event_rejects_mismatched_payload_for_kind() -> None:
    adapter: TypeAdapter[ConversationEvent] = TypeAdapter(ConversationEvent)

    with pytest.raises(ValidationError):
        adapter.validate_python(
            {
                "role": "assistant",
                "kind": "message",
                "payload": {
                    "tool_name": "lookup",
                    "tool_arguments": {},
                    "call_id": "call-1",
                },
                "event_index": 0,
            }
        )


def test_conversation_event_parses_discriminated_union() -> None:
    adapter: TypeAdapter[ConversationEvent] = TypeAdapter(ConversationEvent)

    event = adapter.validate_python(
        {
            "role": "tool",
            "kind": "tool_call",
            "payload": ToolCallPayload(
                tool_name="lookup",
                tool_arguments={"x": 1},
                call_id="call-1",
            ).model_dump(mode="json"),
            "event_index": 2,
        }
    )

    assert event.kind == "tool_call"
    assert event.payload.call_id == "call-1"
