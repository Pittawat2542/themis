"""Shared adapter helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


def stable_fingerprint(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def dump_response(response: object) -> dict[str, object]:
    if hasattr(response, "model_dump"):
        return getattr(response, "model_dump")(mode="json")
    if isinstance(response, Mapping):
        return dict(response)
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


def extract_headers(response: object) -> dict[str, object] | None:
    headers = getattr(response, "headers", None)
    if headers is None:
        headers = getattr(response, "response_headers", None)
    if headers is None:
        return None
    if isinstance(headers, Mapping):
        return dict(headers)
    return None
