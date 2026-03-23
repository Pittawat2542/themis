"""Shared types for catalog dataset helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from themis.types.json_types import JSONDict

type CatalogRow = dict[str, object]
type CatalogPromptMessage = dict[str, object]
type CatalogMetadataLoader = Callable[[str, str | None], JSONDict]
type CatalogRowLoader = Callable[..., list[CatalogRow]]
type CatalogRowNormalizer = Callable[
    [list[CatalogRow], object], "CatalogNormalizedRows"
]


@dataclass(frozen=True, slots=True)
class CatalogNormalizedRows:
    rows: list[CatalogRow]
    stats: JSONDict = field(default_factory=dict)
