"""Helpers for sanitizing persisted snapshot-facing configuration."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import cast
from urllib.parse import urlsplit, urlunsplit

from themis.core.base import JSONValue

_SECRET_KEY_MARKERS = (
    "api_key",
    "apikey",
    "token",
    "secret",
    "password",
    "passwd",
    "access_key",
    "private_key",
    "connection_string",
    "dsn",
)
_REFERENCE_PATTERNS = (
    re.compile(r"^\$\{[A-Z0-9_]+\}$"),
    re.compile(r"^env:[A-Z0-9_][A-Z0-9_]*$"),
    re.compile(r"^secret://.+$"),
)
_SECRET_VALUE_PATTERNS = (
    re.compile(r"^sk-[A-Za-z0-9._-]+$"),
    re.compile(r"^sk-ant-[A-Za-z0-9._-]+$"),
    re.compile(r"^gh[pousr]_[A-Za-z0-9]+$"),
    re.compile(r"^hf_[A-Za-z0-9]+$"),
    re.compile(r"^xox[baprs]-[A-Za-z0-9-]+$"),
    re.compile(r"^-----BEGIN [A-Z ]+-----$"),
)


def is_secret_reference(value: str) -> bool:
    stripped = value.strip()
    return any(pattern.match(stripped) for pattern in _REFERENCE_PATTERNS)


def sanitize_persisted_json_value(value: JSONValue, *, field_path: str) -> JSONValue:
    return _sanitize_value(value, path=(field_path,))


def sanitize_persisted_string_mapping(
    value: Mapping[str, str],
    *,
    field_path: str,
) -> dict[str, str]:
    sanitized = sanitize_persisted_json_value(dict(value), field_path=field_path)
    if not isinstance(sanitized, dict) or any(
        not isinstance(item, str) for item in sanitized.values()
    ):
        raise TypeError(f"Expected sanitized string mapping at {field_path}")
    return cast(dict[str, str], sanitized)


def _sanitize_value(value: JSONValue, *, path: tuple[str, ...]) -> JSONValue:
    if isinstance(value, dict):
        return {
            key: _sanitize_value(item, path=path + (str(key),))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _sanitize_value(item, path=path + (str(index),))
            for index, item in enumerate(value)
        ]
    if isinstance(value, str):
        return _sanitize_string(value, path=path)
    return value


def _sanitize_string(value: str, *, path: tuple[str, ...]) -> str:
    if value == "" or value == "<redacted>":
        return value
    redacted_url = _redact_url_credentials(value)
    if redacted_url is not None:
        return redacted_url
    if is_secret_reference(value):
        return value
    if _path_is_secret_like(path) or _looks_like_secret_value(value):
        raise ValueError(
            f"Refusing to persist literal secret at {'.'.join(path)}; use an env or secret reference instead."
        )
    return value


def _normalize_segment(segment: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", segment.lower()).strip("_")


def _path_is_secret_like(path: tuple[str, ...]) -> bool:
    normalized_path = [_normalize_segment(segment) for segment in path]
    return any(
        marker in segment
        for segment in normalized_path
        for marker in _SECRET_KEY_MARKERS
    )


def _looks_like_secret_value(value: str) -> bool:
    stripped = value.strip()
    return any(pattern.match(stripped) for pattern in _SECRET_VALUE_PATTERNS)


def _redact_url_credentials(value: str) -> str | None:
    parts = urlsplit(value)
    if not parts.scheme or parts.hostname is None:
        return None
    if parts.username is None and parts.password is None:
        return None

    host = parts.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    if parts.port is not None:
        host = f"{host}:{parts.port}"

    if parts.username is not None and parts.password is not None:
        auth = f"{parts.username}:<redacted>@"
    elif parts.username is not None:
        auth = f"{parts.username}@"
    else:
        auth = ""

    return urlunsplit(
        (parts.scheme, f"{auth}{host}", parts.path, parts.query, parts.fragment)
    )
