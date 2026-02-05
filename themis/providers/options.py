"""Provider option normalization helpers."""

from __future__ import annotations

from typing import Any, Mapping


def normalize_provider_options(
    provider_options: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Normalize provider kwargs across API/session entry points.

    Rules:
    - map `base_url` -> `api_base` when `api_base` is not already set
    - if both are present, keep `api_base` and drop `base_url`
    """
    options = dict(provider_options or {})
    if "base_url" in options and "api_base" not in options:
        options["api_base"] = options.pop("base_url")
    elif "base_url" in options:
        options.pop("base_url")
    return options


__all__ = ["normalize_provider_options"]
