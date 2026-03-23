"""OpenAI transport helpers for catalog runtime."""

from __future__ import annotations

from time import perf_counter
from typing import cast

from themis._optional import import_optional
from themis.contracts.protocols import InferenceResult
from themis.errors import InferenceError, RetryableProviderError
from themis.records import InferenceRecord, TokenUsage
from themis.types.enums import ErrorCode, ResponseFormat

from ..datasets._prompts import _prompt_messages_from_context
from ._coercion import _coerce_message_text, _context_item_id
from ._responses import (
    _openai_mcp_tool_payload,
    _response_conversation,
    _response_output_text,
    _response_token_usage,
    _runtime_secret,
)


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
        _runtime_secret(runtime, "OPENAI_API_KEY") or extras.get("api_key") or "dummy"
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
        _runtime_secret(runtime, "OPENAI_API_KEY") or extras.get("api_key") or "dummy"
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
