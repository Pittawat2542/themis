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
from themis.errors import InferenceError, RetryableProviderError
from themis.extractors.builtin import _normalize_text
from themis.records import InferenceRecord, TokenUsage
from themis.specs.foundational import JudgeInferenceSpec
from themis.types.enums import ErrorCode, PromptRole, ResponseFormat
from themis.types.json_types import JSONDict

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
