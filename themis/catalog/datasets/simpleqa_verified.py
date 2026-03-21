"""Dataset provider for google/simpleqa-verified."""

from __future__ import annotations

from themis.specs.foundational import DatasetSpec

from .common import (
    BuiltinDatasetProvider,
    CatalogNormalizedRows,
    _normalize_simpleqa_rows,
    load_huggingface_rows,
)


class BuiltinSimpleQAVerifiedDatasetProvider(BuiltinDatasetProvider):
    def __init__(self, *, huggingface_loader=None) -> None:
        super().__init__(huggingface_loader=huggingface_loader or load_huggingface_rows)

    def normalize_loaded_rows(
        self,
        rows: list[dict[str, object]],
        dataset: DatasetSpec,
    ) -> CatalogNormalizedRows:
        return _normalize_simpleqa_rows(rows, dataset)
