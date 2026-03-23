"""Dataset provider for openai/frontierscience."""

from __future__ import annotations

from ...datasets._normalizers import _normalize_frontierscience_rows
from ...datasets._providers import BuiltinDatasetProvider
from ...datasets._types import CatalogNormalizedRows


class BuiltinFrontierScienceDatasetProvider(BuiltinDatasetProvider):
    def normalize_loaded_rows(
        self,
        rows: list[dict[str, object]],
        dataset: object,
    ) -> CatalogNormalizedRows:
        return _normalize_frontierscience_rows(rows, dataset)
