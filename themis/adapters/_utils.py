"""Shared adapter helpers."""

from __future__ import annotations

import hashlib
import asyncio
import inspect
import json
from collections.abc import Mapping

from themis.core.base import JSONValue
from themis.core.models import ProviderTelemetry, SeedCapability

_ALLOWED_RESPONSE_HEADERS = {
    "request-id",
    "retry-after",
    "x-request-id",
    "x-ratelimit-limit",
    "x-ratelimit-limit-requests",
    "x-ratelimit-remaining",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-reset",
    "x-ratelimit-reset-requests",
    "ratelimit-limit-requests",
}


def stable_fingerprint(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


async def call_maybe_sync(function, /, **kwargs: object) -> object:
    """Invoke sync SDK methods off-loop while preserving async SDK methods."""

    if inspect.iscoroutinefunction(function):
        return await function(**kwargs)
    result = await asyncio.to_thread(function, **kwargs)
    return await result if inspect.isawaitable(result) else result


async def close_maybe_sync(client: object) -> None:
    close = getattr(client, "aclose", None) or getattr(client, "close", None)
    if close is None:
        return
    if inspect.iscoroutinefunction(close):
        await close()
        return
    result = await asyncio.to_thread(close)
    if inspect.isawaitable(result):
        await result


def normalize_json_value(value: object) -> JSONValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): normalize_json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [normalize_json_value(item) for item in value]
    return repr(value)


def dump_response(response: object) -> dict[str, JSONValue]:
    if hasattr(response, "model_dump"):
        return normalize_json_value(getattr(response, "model_dump")(mode="json"))  # type: ignore[return-value]
    if isinstance(response, Mapping):
        return {
            str(key): normalize_json_value(value) for key, value in response.items()
        }
    return {"repr": repr(response)}


def extract_token_usage(usage: object | None) -> dict[str, int] | None:
    if usage is None:
        return None
    usage_mapping = usage if isinstance(usage, Mapping) else None
    prompt_tokens = _read_usage_value(usage, usage_mapping, "input_tokens")
    completion_tokens = _read_usage_value(usage, usage_mapping, "output_tokens")
    if prompt_tokens is None:
        prompt_tokens = _read_usage_value(usage, usage_mapping, "prompt_tokens")
    if completion_tokens is None:
        completion_tokens = _read_usage_value(
            usage, usage_mapping, "completion_tokens"
        )
    if prompt_tokens is None:
        prompt_tokens = _read_usage_value(usage, usage_mapping, "inputTokens")
    if completion_tokens is None:
        completion_tokens = _read_usage_value(usage, usage_mapping, "outputTokens")
    if prompt_tokens is None:
        prompt_tokens = _read_usage_value(usage, usage_mapping, "prompt_eval_count")
    if completion_tokens is None:
        completion_tokens = _read_usage_value(usage, usage_mapping, "eval_count")
    if prompt_tokens is None and completion_tokens is None:
        return None
    return {
        "prompt_tokens": _coerce_token_count(prompt_tokens),
        "completion_tokens": _coerce_token_count(completion_tokens),
    }


def extract_headers(response: object) -> dict[str, JSONValue] | None:
    headers = getattr(response, "headers", None)
    if headers is None:
        headers = getattr(response, "response_headers", None)
    if headers is None:
        return None
    if isinstance(headers, Mapping):
        return {
            str(key).lower(): normalize_json_value(value)
            for key, value in headers.items()
            if str(key).lower() in _ALLOWED_RESPONSE_HEADERS
        }
    return None


def extract_rate_limit(
    headers: Mapping[str, JSONValue] | None,
) -> dict[str, JSONValue] | None:
    if headers is None:
        return None
    for key in (
        "x-ratelimit-limit-requests",
        "ratelimit-limit-requests",
        "x-ratelimit-limit",
    ):
        value = headers.get(key)
        if isinstance(value, str) and value.isdigit():
            return {"requests_per_minute": int(value)}
        if isinstance(value, int):
            return {"requests_per_minute": value}
    return None


def extract_provider_telemetry(
    response: object,
    *,
    seed_requested: int | None = None,
    seed_applied: int | None = None,
    seed_capability: SeedCapability = SeedCapability.UNSUPPORTED,
) -> ProviderTelemetry:
    """Extract common telemetry fields from provider-specific response shapes."""

    raw_response = dump_response(response)
    headers = extract_headers(response)
    return ProviderTelemetry(
        request_id=_extract_request_id(response, raw_response),
        token_usage=extract_token_usage(_extract_usage(response, raw_response)),
        raw_response=raw_response,
        headers=headers,
        rate_limit=extract_rate_limit(headers),
        seed_requested=seed_requested,
        seed_applied=seed_applied,
        seed_capability=seed_capability,
    )


def provider_artifacts(telemetry: ProviderTelemetry) -> dict[str, JSONValue]:
    artifacts: dict[str, JSONValue] = {
        "provider_request_id": telemetry.request_id,
        "raw_response": telemetry.raw_response,
        "response_headers": telemetry.headers or {},
        "seed_requested": telemetry.seed_requested,
        "seed_applied": telemetry.seed_applied,
        "seed_capability": telemetry.seed_capability.value,
    }
    if telemetry.rate_limit is not None:
        artifacts["rate_limit"] = telemetry.rate_limit
    return artifacts


def _read_usage_value(
    usage: object,
    usage_mapping: Mapping[object, object] | None,
    key: str,
) -> object | None:
    if usage_mapping is not None and key in usage_mapping:
        return usage_mapping[key]
    return getattr(usage, key, None)


def _coerce_token_count(value: object | None) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float | str):
        return int(value)
    return 0


def _extract_usage(response: object, raw_response: Mapping[str, JSONValue]) -> object:
    usage = getattr(response, "usage", None)
    if usage is not None:
        return usage
    if "usage" in raw_response:
        return raw_response["usage"]
    return raw_response


def _extract_request_id(
    response: object, raw_response: Mapping[str, JSONValue]
) -> str | None:
    for attr in ("id", "response_id", "request_id"):
        value = getattr(response, attr, None)
        if value is not None:
            return str(value)
    for key in ("id", "response_id", "request_id"):
        value = raw_response.get(key)
        if value is not None:
            return str(value)
    metadata = raw_response.get("ResponseMetadata")
    if isinstance(metadata, Mapping):
        request_id = metadata.get("RequestId") or metadata.get("requestId")
        if request_id is not None:
            return str(request_id)
    return None
