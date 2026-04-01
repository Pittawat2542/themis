"""Shared adapter helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

from themis.core.base import JSONValue


def stable_fingerprint(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


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
    prompt_tokens = getattr(usage, "input_tokens", None)
    completion_tokens = getattr(usage, "output_tokens", None)
    if prompt_tokens is None:
        prompt_tokens = getattr(usage, "prompt_tokens", None)
    if completion_tokens is None:
        completion_tokens = getattr(usage, "completion_tokens", None)
    if prompt_tokens is None and completion_tokens is None:
        return None
    return {
        "prompt_tokens": int(prompt_tokens or 0),
        "completion_tokens": int(completion_tokens or 0),
    }


def extract_headers(response: object) -> dict[str, JSONValue] | None:
    headers = getattr(response, "headers", None)
    if headers is None:
        headers = getattr(response, "response_headers", None)
    if headers is None:
        return None
    if isinstance(headers, Mapping):
        return {str(key): normalize_json_value(value) for key, value in headers.items()}
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
