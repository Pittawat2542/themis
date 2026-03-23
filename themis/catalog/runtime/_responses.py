"""OpenAI Responses/MCP parsing helpers for catalog runtime."""

from __future__ import annotations

import json

from themis.errors import SpecValidationError
from themis.records import (
    MessageEvent,
    MessagePayload,
    TokenUsage,
    ToolCallEvent,
    ToolCallPayload,
    ToolResultEvent,
    ToolResultPayload,
)
from themis.records.conversation import Conversation, ConversationEvent
from themis.types.enums import ErrorCode, PromptRole
from themis.types.json_validation import validate_json_dict, validate_json_value

from ._coercion import _coerce_usage_int


def _runtime_secret(runtime, key: str) -> str | None:
    secrets = getattr(runtime, "secrets", {}) or {}
    value = secrets.get(key)
    if value is None:
        return None
    if hasattr(value, "get_secret_value"):
        return value.get_secret_value()
    return str(value)


def _openai_mcp_tool_payload(server, runtime) -> dict[str, object]:
    if server.require_approval == "always":
        raise SpecValidationError(
            code=ErrorCode.PLUGIN_INCOMPATIBLE,
            message=(
                "OpenAI MCP integration does not support approval-gated MCP "
                f"servers: {server.id}."
            ),
        )
    payload: dict[str, object] = {
        "type": "mcp",
        "server_label": server.server_label,
        "require_approval": server.require_approval,
    }
    if server.server_description is not None:
        payload["server_description"] = server.server_description
    if server.server_url is not None:
        payload["server_url"] = server.server_url
    if server.connector_id is not None:
        payload["connector_id"] = server.connector_id
    if server.allowed_tools:
        payload["allowed_tools"] = list(server.allowed_tools)
    if server.authorization_secret_name is not None:
        authorization = _runtime_secret(runtime, server.authorization_secret_name)
        if authorization is None:
            raise SpecValidationError(
                code=ErrorCode.MISSING_OPTIONAL_DEPENDENCY,
                message=(
                    "Missing runtime secret for MCP authorization: "
                    f"{server.authorization_secret_name}."
                ),
            )
        payload["authorization"] = authorization
    return payload


def _response_output_text(response: object) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str):
        return output_text
    return ""


def _response_output_items(response: object) -> list[object]:
    items = getattr(response, "output", None)
    if isinstance(items, list):
        return items
    return []


def _response_item_type(item: object) -> str | None:
    if isinstance(item, dict):
        item_type = item.get("type")
        return str(item_type) if item_type is not None else None
    item_type = getattr(item, "type", None)
    return str(item_type) if item_type is not None else None


def _response_item_attr(item: object, name: str) -> object:
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)


def _response_usage(response: object) -> object:
    if isinstance(response, dict):
        return response.get("usage")
    return getattr(response, "usage", None)


def _usage_attr(usage: object, name: str) -> object:
    if isinstance(usage, dict):
        return usage.get(name)
    return getattr(usage, name, None)


def _response_token_usage(response: object) -> TokenUsage | None:
    usage = _response_usage(response)
    if usage is None:
        return None
    prompt_tokens = _usage_attr(usage, "prompt_tokens")
    completion_tokens = _usage_attr(usage, "completion_tokens")
    if completion_tokens is None:
        completion_tokens = _usage_attr(usage, "output_tokens")
    total_tokens = _usage_attr(usage, "total_tokens")
    reasoning_tokens = _usage_attr(usage, "reasoning_tokens")
    if reasoning_tokens is None:
        output_details = _usage_attr(usage, "output_tokens_details")
        reasoning_tokens = _usage_attr(output_details, "reasoning_tokens")
    if (
        prompt_tokens is None
        and completion_tokens is None
        and total_tokens is None
        and reasoning_tokens is None
    ):
        return None
    prompt_value = _coerce_usage_int(prompt_tokens) or 0
    completion_value = _coerce_usage_int(completion_tokens) or 0
    total_value = _coerce_usage_int(total_tokens) or (prompt_value + completion_value)
    return TokenUsage(
        prompt_tokens=prompt_value,
        completion_tokens=completion_value,
        total_tokens=total_value,
        reasoning_tokens=_coerce_usage_int(reasoning_tokens),
    )


def _response_mcp_tool_name(item: object) -> str:
    server_label = str(_response_item_attr(item, "server_label") or "mcp")
    tool_name = str(
        _response_item_attr(item, "name")
        or _response_item_attr(item, "tool_name")
        or "tool"
    )
    return f"{server_label}:{tool_name}"


def _maybe_json_loads(value: object) -> object:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _response_conversation(response: object, *, final_text: str) -> Conversation | None:
    events: list[ConversationEvent] = []
    event_index = 0
    for item in _response_output_items(response):
        item_type = _response_item_type(item)
        if item_type == "mcp_call":
            call_id = str(
                _response_item_attr(item, "call_id")
                or _response_item_attr(item, "id")
                or f"mcp-call-{event_index}"
            )
            parsed_arguments = _maybe_json_loads(_response_item_attr(item, "arguments"))
            events.append(
                ToolCallEvent(
                    role=PromptRole.ASSISTANT,
                    payload=ToolCallPayload(
                        tool_name=_response_mcp_tool_name(item),
                        tool_arguments=validate_json_dict(
                            parsed_arguments,
                            label="MCP tool arguments",
                        ),
                        call_id=call_id,
                    ),
                    event_index=event_index,
                )
            )
            event_index += 1
            continue
        if item_type in {"mcp_call_output", "mcp_tool_result"}:
            call_id = str(
                _response_item_attr(item, "call_id")
                or _response_item_attr(item, "id")
                or f"mcp-call-{event_index}"
            )
            events.append(
                ToolResultEvent(
                    role=PromptRole.TOOL,
                    payload=ToolResultPayload(
                        call_id=call_id,
                        result=validate_json_value(
                            _maybe_json_loads(_response_item_attr(item, "output")),
                            label="MCP tool result",
                        ),
                        is_error=bool(_response_item_attr(item, "error")),
                    ),
                    event_index=event_index,
                )
            )
            event_index += 1
    if final_text:
        events.append(
            MessageEvent(
                role=PromptRole.ASSISTANT,
                payload=MessagePayload(content=final_text),
                event_index=event_index,
            )
        )
    if not events:
        return None
    return Conversation(events=events)
