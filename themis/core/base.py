"""Shared immutable model and hashing helpers for Themis."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict

type JSONPrimitive = str | int | float | bool | None
type JSONValue = JSONPrimitive | dict[str, JSONValue] | list[JSONValue]


def _canonicalize(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if all(hasattr(value, attr) for attr in ("component_id", "version")) and hasattr(
        value, "fingerprint"
    ):
        fingerprint = (
            value.fingerprint() if callable(value.fingerprint) else value.fingerprint
        )
        return {
            "component_id": value.component_id,
            "version": value.version,
            "fingerprint": fingerprint,
        }
    if isinstance(value, BaseModel):
        if isinstance(value, HashableModel):
            return value.canonical_data()
        return {key: _canonicalize(item) for key, item in value.model_dump().items()}
    if isinstance(value, dict):
        return {key: _canonicalize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_canonicalize(item) for item in value]
    return value


class FrozenModel(BaseModel):
    """Base Pydantic model used by the immutable core."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)


class HashableModel(FrozenModel):
    """Immutable model with stable content-addressable hashing."""

    def canonical_data(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        for field_name, field_info in self.__class__.model_fields.items():
            if field_info.exclude:
                continue
            value = getattr(self, field_name)
            data[field_name] = _canonicalize(value)
        return data

    def _canonical_json(self) -> str:
        return json.dumps(
            self.canonical_data(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    def compute_hash(self) -> str:
        return hashlib.sha256(self._canonical_json().encode("utf-8")).hexdigest()
