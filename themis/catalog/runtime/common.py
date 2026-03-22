"""Shared catalog runtime helpers."""

from __future__ import annotations

import json
import os
import re
from time import perf_counter
from typing import Literal, cast

from themis import InferenceParamsSpec, ModelSpec, PromptMessage
from themis._optional import import_optional
from themis.contracts.protocols import InferenceResult
from themis.errors import InferenceError, RetryableProviderError, SpecValidationError
from themis.extractors.builtin import _normalize_text
from themis.records import (
    InferenceRecord,
    MessageEvent,
    MessagePayload,
    TokenUsage,
    ToolCallEvent,
    ToolCallPayload,
    ToolResultEvent,
    ToolResultPayload,
)
from themis.records.conversation import Conversation, ConversationEvent
from themis.specs.foundational import JudgeInferenceSpec
from themis.types.enums import ErrorCode, PromptRole, ResponseFormat
from themis.types.json_types import JSONDict
from themis.types.json_validation import validate_json_dict, validate_json_value

from ..datasets import _prompt_messages_from_context

_SIMPLEQA_GRADE_PATTERN = re.compile(
    r"\b(CORRECT|INCORRECT|NOT[_ ]ATTEMPTED|A|B|C)\b", flags=re.IGNORECASE
)
_LPFQA_REFERENCE_PATTERN = re.compile(
    r"<参考答案>[：:]\s*(?P<answer>.*?)\s*(?:<评估要点>|$)",
    flags=re.DOTALL,
)
_HLE_ANSWER_PATTERN = re.compile(r"(?im)^answer:\s*(?P<value>.+)$")
_HLE_CONFIDENCE_PATTERN = re.compile(r"(?im)^confidence:\s*(?P<value>\d+)")


def _run_openai_chat_inference(
    trial,
    context,
    runtime,
    *,
    base_url: str | None,
    provider_label: str,
    missing_extra: str,
) -> InferenceResult:
    if getattr(trial, "mcp_servers", ()):
        return _run_openai_responses_mcp_inference(
            trial,
            context,
            runtime,
            base_url=base_url,
            provider_label=provider_label,
            missing_extra=missing_extra,
        )
    if trial.params.response_format not in (None, ResponseFormat.TEXT):
        raise InferenceError(
            code=ErrorCode.PLUGIN_INCOMPATIBLE,
            message=f"{provider_label} catalog engine currently supports text responses only.",
            details={"response_format": str(trial.params.response_format)},
        )
    openai = import_optional("openai", extra=missing_extra)
    extras = dict(trial.model.extras)
    timeout_seconds = float(extras.get("timeout_seconds", 60.0))
    client_kwargs: dict[str, object] = {"timeout": timeout_seconds}
    if base_url is not None:
        client_kwargs["base_url"] = base_url
    api_key = (
        _runtime_secret(runtime, "OPENAI_API_KEY")
        or _runtime_secret(runtime, "OPENAI_COMPAT_API_KEY")
        or extras.get("api_key")
        or "dummy"
    )
    client_kwargs["api_key"] = str(api_key)
    client = openai.OpenAI(**client_kwargs)
    messages = _resolved_messages(trial, context)
    request_kwargs: dict[str, object] = {
        "model": trial.model.model_id,
        "messages": messages,
        "temperature": trial.params.temperature,
        "max_tokens": trial.params.max_tokens,
    }
    if trial.params.top_p is not None:
        request_kwargs["top_p"] = trial.params.top_p
    if trial.params.stop_sequences:
        request_kwargs["stop"] = trial.params.stop_sequences
    if trial.params.seed is not None:
        request_kwargs["seed"] = trial.params.seed & 0xFFFFFFFF
    if trial.params.logprobs is not None:
        request_kwargs["logprobs"] = True
        request_kwargs["top_logprobs"] = trial.params.logprobs
    if trial.params.top_k is not None:
        extra_body = cast(
            dict[str, object], request_kwargs.setdefault("extra_body", {})
        )
        extra_body["top_k"] = trial.params.top_k
    for key, value in trial.params.extras.items():
        extra_body = cast(
            dict[str, object], request_kwargs.setdefault("extra_body", {})
        )
        extra_body[key] = value

    start = perf_counter()
    try:
        response = client.chat.completions.create(**request_kwargs)
    except openai.AuthenticationError as exc:
        raise InferenceError(
            code=ErrorCode.PROVIDER_AUTH,
            message=f"{provider_label} rejected authentication.",
            details={"body": getattr(exc, "body", None)},
        ) from exc
    except openai.RateLimitError as exc:
        raise RetryableProviderError(
            code=ErrorCode.PROVIDER_RATE_LIMIT,
            message=f"{provider_label} rate limited the request.",
            details={"body": getattr(exc, "body", None)},
        ) from exc
    except openai.APIConnectionError as exc:
        raise RetryableProviderError(
            code=ErrorCode.PROVIDER_UNAVAILABLE,
            message=f"Could not reach {provider_label}: {exc}",
        ) from exc
    except openai.APIStatusError as exc:
        error_cls = RetryableProviderError if exc.status_code >= 500 else InferenceError
        raise error_cls(
            code=ErrorCode.PROVIDER_UNAVAILABLE,
            message=f"{provider_label} returned HTTP {exc.status_code}.",
            details={"body": getattr(exc, "body", None)},
        ) from exc

    choice = response.choices[0] if getattr(response, "choices", None) else None
    if choice is None or getattr(choice, "message", None) is None:
        raise InferenceError(
            code=ErrorCode.PARSE_ERROR,
            message=f"{provider_label} returned no message choices.",
            details={"provider_request_id": getattr(response, "id", None)},
        )
    latency_ms = (perf_counter() - start) * 1000
    usage = getattr(response, "usage", None)
    return InferenceResult(
        inference=InferenceRecord(
            spec_hash=f"inference_{trial.item_id}",
            raw_text=_coerce_message_text(choice.message.content),
            finish_reason=getattr(choice, "finish_reason", None),
            latency_ms=latency_ms,
            provider_request_id=getattr(response, "id", None),
            token_usage=TokenUsage(
                prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
            ),
        )
    )


def _resolved_messages(trial, context: object) -> list[dict[str, object]]:
    context_messages = _prompt_messages_from_context(context)
    if context_messages:
        return [_openai_message_payload(dict(message)) for message in context_messages]
    return [
        _openai_message_payload(message.model_dump(mode="json"))
        for message in trial.prompt.messages
    ]


def _run_openai_responses_mcp_inference(
    trial,
    context,
    runtime,
    *,
    base_url: str | None,
    provider_label: str,
    missing_extra: str,
) -> InferenceResult:
    if trial.prompt.follow_up_turns:
        raise InferenceError(
            code=ErrorCode.PLUGIN_INCOMPATIBLE,
            message=f"{provider_label} MCP runs do not support follow-up turns.",
        )
    if trial.params.response_format not in (None, ResponseFormat.TEXT):
        raise InferenceError(
            code=ErrorCode.PLUGIN_INCOMPATIBLE,
            message=f"{provider_label} MCP runs currently support text responses only.",
            details={"response_format": str(trial.params.response_format)},
        )
    unsupported_params: list[str] = []
    if trial.params.logprobs is not None:
        unsupported_params.append("logprobs")
    if trial.params.top_k is not None:
        unsupported_params.append("top_k")
    if trial.params.stop_sequences:
        unsupported_params.append("stop_sequences")
    if unsupported_params:
        raise InferenceError(
            code=ErrorCode.PLUGIN_INCOMPATIBLE,
            message=(
                f"{provider_label} MCP runs do not support parameter(s): "
                f"{', '.join(unsupported_params)}."
            ),
        )
    openai = import_optional("openai", extra=missing_extra)
    extras = dict(trial.model.extras)
    timeout_seconds = float(extras.get("timeout_seconds", 60.0))
    client_kwargs: dict[str, object] = {"timeout": timeout_seconds}
    if base_url is not None:
        client_kwargs["base_url"] = base_url
    api_key = (
        _runtime_secret(runtime, "OPENAI_API_KEY")
        or _runtime_secret(runtime, "OPENAI_COMPAT_API_KEY")
        or extras.get("api_key")
        or "dummy"
    )
    client_kwargs["api_key"] = str(api_key)
    client = openai.OpenAI(**client_kwargs)
    request_kwargs: dict[str, object] = {
        "model": trial.model.model_id,
        "input": _resolved_response_input(trial, context),
        "tools": [
            _openai_mcp_tool_payload(server, runtime) for server in trial.mcp_servers
        ],
        "max_output_tokens": trial.params.max_tokens,
    }
    if trial.params.temperature is not None:
        request_kwargs["temperature"] = trial.params.temperature
    if trial.params.top_p is not None:
        request_kwargs["top_p"] = trial.params.top_p
    if trial.params.seed is not None:
        request_kwargs["seed"] = trial.params.seed & 0xFFFFFFFF
    for key, value in trial.params.extras.items():
        extra_body = cast(
            dict[str, object], request_kwargs.setdefault("extra_body", {})
        )
        extra_body[key] = value

    start = perf_counter()
    try:
        response = client.responses.create(**request_kwargs)
    except openai.AuthenticationError as exc:
        raise InferenceError(
            code=ErrorCode.PROVIDER_AUTH,
            message=f"{provider_label} rejected authentication.",
            details={"body": getattr(exc, "body", None)},
        ) from exc
    except openai.RateLimitError as exc:
        raise RetryableProviderError(
            code=ErrorCode.PROVIDER_RATE_LIMIT,
            message=f"{provider_label} rate limited the request.",
            details={"body": getattr(exc, "body", None)},
        ) from exc
    except openai.APIConnectionError as exc:
        raise RetryableProviderError(
            code=ErrorCode.PROVIDER_UNAVAILABLE,
            message=f"Could not reach {provider_label}: {exc}",
        ) from exc
    except openai.APIStatusError as exc:
        error_cls = RetryableProviderError if exc.status_code >= 500 else InferenceError
        raise error_cls(
            code=ErrorCode.PROVIDER_UNAVAILABLE,
            message=f"{provider_label} returned HTTP {exc.status_code}.",
            details={"body": getattr(exc, "body", None)},
        ) from exc

    latency_ms = (perf_counter() - start) * 1000
    output_text = _response_output_text(response)
    conversation = _response_conversation(response, final_text=output_text)
    # Responses/MCP usage metadata is optional. Callers that need token
    # accounting may need a follow-up request to the provider's token endpoint.
    token_usage = _response_token_usage(response)
    return InferenceResult(
        inference=InferenceRecord(
            spec_hash=(
                "inference_"
                f"{getattr(trial, 'trial_id', getattr(trial, 'item_id', _context_item_id(context)))}"
            ),
            raw_text=output_text,
            latency_ms=latency_ms,
            provider_request_id=getattr(response, "id", None),
            token_usage=token_usage,
            conversation=conversation,
        ),
        conversation=conversation,
    )


def _resolved_response_input(trial, context: object) -> list[dict[str, object]]:
    return [
        _openai_response_input_message(message)
        for message in _resolved_messages(trial, context)
    ]


def _run_text_judge(
    *,
    judge_service,
    metric_id: str,
    trial,
    candidate,
    context,
    judge_model_id: str,
    judge_provider: str,
    messages: list[PromptMessage],
    demo_expected_response: str,
) -> str:
    runtime_context = context.get("runtime_context")
    dataset_context = dict(context)
    if judge_provider == "demo":
        dataset_context["judge_expected_response"] = demo_expected_response
    prompt = trial.prompt.model_copy(
        update={"messages": messages, "follow_up_turns": []}
    )
    record = judge_service.judge(
        metric_id,
        candidate,
        _build_judge_spec(model_id=judge_model_id, provider=judge_provider),
        prompt,
        {
            "runtime_context": runtime_context,
            "dataset_context": dataset_context,
            "task_spec": trial.task,
        },
    )
    return record.raw_text or ""


def _build_judge_spec(*, model_id: str, provider: str):
    return JudgeInferenceSpec(
        model=ModelSpec(
            model_id=model_id,
            provider=provider,
            extras=_provider_model_extras(provider),
        ),
        params=InferenceParamsSpec(max_tokens=8192, temperature=0.0),
    )


def _normalize_provider_name(provider: str) -> str:
    return provider.replace("-", "_")


def _provider_model_extras(provider: str) -> JSONDict:
    normalized = _normalize_provider_name(provider)
    if normalized == "openai_compatible":
        return {
            "base_url": os.getenv(
                "OPENAI_COMPAT_BASE_URL",
                "http://127.0.0.1:8000/v1",
            ),
            "timeout_seconds": 60.0,
        }
    return {}


def _context_item_id(context: object) -> str:
    item_id = getattr(context, "item_id", None)
    if item_id is not None:
        return str(item_id)
    if hasattr(context, "get"):
        resolved = context.get("item_id")  # type: ignore[attr-defined]
        if resolved is not None:
            return str(resolved)
    return "item"


def _expected_text(context: object) -> str:
    if hasattr(context, "get"):
        for key in ("expected", "answer", "answer_letter"):
            resolved = context.get(key)  # type: ignore[attr-defined]
            if resolved is not None:
                return _coerce_text(resolved)
    return ""


def _expected_demo_response(context: object) -> str:
    if hasattr(context, "get"):
        for key in ("judge_expected_response", "expected_response"):
            resolved = context.get(key)  # type: ignore[attr-defined]
            if isinstance(resolved, str):
                return resolved
    return _expected_text(context)


def _coerce_float(value: object) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (str, int, float)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, sort_keys=True)


def _coerce_message_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
        return "".join(text_parts)
    return ""


def _openai_message_payload(message: dict[str, object]) -> dict[str, object]:
    content = message.get("content")
    if not isinstance(content, list):
        return dict(message)
    converted_parts: list[dict[str, object]] = []
    for part in content:
        if not isinstance(part, dict):
            raise InferenceError(
                code=ErrorCode.PLUGIN_INCOMPATIBLE,
                message="Prompt message content parts must be objects.",
            )
        part_type = part.get("type")
        if part_type == "text":
            converted_parts.append({"type": "text", "text": str(part.get("text", ""))})
            continue
        if part_type == "image_url":
            converted_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": str(part.get("image_url", ""))},
                }
            )
            continue
        raise InferenceError(
            code=ErrorCode.PLUGIN_INCOMPATIBLE,
            message=f"Unsupported prompt content part type '{part_type}'.",
        )
    payload = dict(message)
    payload["content"] = converted_parts
    return payload


def _openai_response_input_message(message: dict[str, object]) -> dict[str, object]:
    content = message.get("content")
    if not isinstance(content, list):
        return {
            "role": message.get("role"),
            "content": [
                {"type": "input_text", "text": "" if content is None else str(content)}
            ],
        }
    converted_parts: list[dict[str, object]] = []
    for part in content:
        if not isinstance(part, dict):
            raise InferenceError(
                code=ErrorCode.PLUGIN_INCOMPATIBLE,
                message="Prompt message content parts must be objects.",
            )
        part_type = part.get("type")
        if part_type == "text":
            converted_parts.append(
                {"type": "input_text", "text": str(part.get("text", ""))}
            )
            continue
        if part_type == "image_url":
            image_url = part.get("image_url")
            if isinstance(image_url, dict):
                url = str(image_url.get("url", ""))
            else:
                url = str(image_url or "")
            converted_parts.append({"type": "input_image", "image_url": url})
            continue
        raise InferenceError(
            code=ErrorCode.PLUGIN_INCOMPATIBLE,
            message=f"Unsupported prompt content part type '{part_type}'.",
        )
    return {"role": message.get("role"), "content": converted_parts}


def _runtime_secret(runtime, key: str) -> str | None:
    secrets = getattr(runtime, "secrets", {}) or {}
    value = secrets.get(key)
    if value is None:
        return None
    if hasattr(value, "get_secret_value"):
        return value.get_secret_value()
    return str(value)


def _extract_lpfqa_reference_answer(text: str) -> str:
    match = _LPFQA_REFERENCE_PATTERN.search(text)
    if match is None:
        return text
    return match.group("answer").strip()


def _extract_hle_answer(text: str) -> str | None:
    match = _HLE_ANSWER_PATTERN.search(text)
    if match is None:
        return None
    return match.group("value").strip()


def _extract_hle_confidence(text: str) -> int:
    match = _HLE_CONFIDENCE_PATTERN.search(text)
    if match is None:
        return 100
    return int(match.group("value"))


def _coerce_json_dict(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return {str(key): value[key] for key in value}
    return {}


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


def _coerce_usage_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, str)):
        return int(value)
    return None


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
            events.append(
                ToolCallEvent(
                    role=PromptRole.ASSISTANT,
                    payload=ToolCallPayload(
                        tool_name=_response_mcp_tool_name(item),
                        tool_arguments=validate_json_dict(
                            _coerce_json_dict(
                                _maybe_json_loads(
                                    _response_item_attr(item, "arguments")
                                )
                            ),
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


def _prompt_messages_with_optional_system(
    system_prompt: str,
    user_prompt: str,
) -> list[PromptMessage]:
    messages: list[PromptMessage] = []
    if system_prompt:
        messages.append(PromptMessage(role=PromptRole.SYSTEM, content=system_prompt))
    messages.append(PromptMessage(role=PromptRole.USER, content=user_prompt))
    return messages


def _parse_simpleqa_grade(
    text: str,
) -> Literal["CORRECT", "INCORRECT", "NOT_ATTEMPTED"]:
    match = _SIMPLEQA_GRADE_PATTERN.search(text)
    if match is None:
        return "NOT_ATTEMPTED"
    token = match.group(1).upper().replace(" ", "_")
    if token in {"A", "CORRECT"}:
        return "CORRECT"
    if token in {"B", "INCORRECT"}:
        return "INCORRECT"
    if token in {"C", "NOT_ATTEMPTED"}:
        return "NOT_ATTEMPTED"
    return "NOT_ATTEMPTED"


def _simpleqa_demo_grade(question: str, target: str, predicted_answer: str) -> str:
    del question
    normalized_target = _normalize_text(target)
    normalized_predicted = _normalize_text(predicted_answer)
    if not normalized_predicted:
        return "C"
    if normalized_predicted == normalized_target:
        return "A"
    return "B"
