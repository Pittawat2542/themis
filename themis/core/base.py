"""Shared immutable model and hashing helpers for Themis."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime
from enum import Enum
from typing import Any, Never, Self

from pydantic import BaseModel, ConfigDict, model_validator

type JSONPrimitive = str | int | float | bool | None
type JSONValue = JSONPrimitive | dict[str, JSONValue] | list[JSONValue]


def _immutable(*_args: object, **_kwargs: object) -> Never:
    raise TypeError("Frozen model collections cannot be mutated")


class FrozenList(list):
    """JSON-compatible list that rejects mutation."""

    __setitem__ = _immutable
    __delitem__ = _immutable
    __iadd__ = _immutable  # type: ignore[assignment]
    __imul__ = _immutable  # type: ignore[assignment]
    append = _immutable
    clear = _immutable
    extend = _immutable
    insert = _immutable
    pop = _immutable
    remove = _immutable
    reverse = _immutable
    sort = _immutable


class FrozenDict(dict):
    """JSON-compatible dictionary that rejects mutation."""

    __setitem__ = _immutable
    __delitem__ = _immutable
    __ior__ = _immutable  # type: ignore[assignment]
    clear = _immutable
    pop = _immutable
    popitem = _immutable  # type: ignore[assignment]
    setdefault = _immutable
    update = _immutable


def deep_freeze(value: Any) -> Any:
    """Recursively freeze collection values without changing JSON serialization."""

    if isinstance(value, (FrozenList, FrozenDict)):
        return value
    if isinstance(value, list):
        return FrozenList(deep_freeze(item) for item in value)
    if isinstance(value, dict):
        return FrozenDict(
            (deep_freeze(key), deep_freeze(item)) for key, item in value.items()
        )
    if isinstance(value, tuple):
        return tuple(deep_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(deep_freeze(item) for item in value)
    if isinstance(value, frozenset):
        return frozenset(deep_freeze(item) for item in value)
    return value


def _canonicalize(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, BaseModel):
        if isinstance(value, HashableModel):
            return value.canonical_data()
        return {key: _canonicalize(item) for key, item in value.model_dump().items()}
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
    if isinstance(value, dict):
        return {key: _canonicalize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_canonicalize(item) for item in value]
    return value


class FrozenModel(BaseModel):
    """Base Pydantic model used by the immutable core."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _freeze_collections(self) -> Self:
        return self._apply_deep_freeze()

    def _apply_deep_freeze(self) -> Self:
        for field_name in self.__class__.model_fields:
            object.__setattr__(self, field_name, deep_freeze(getattr(self, field_name)))
        if self.__pydantic_extra__ is not None:
            object.__setattr__(
                self,
                "__pydantic_extra__",
                deep_freeze(self.__pydantic_extra__),
            )
        return self

    def model_copy(
        self, *, update: Mapping[str, Any] | None = None, deep: bool = False
    ) -> Self:
        copied = super().model_copy(update=update, deep=deep)
        return copied._apply_deep_freeze()


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
