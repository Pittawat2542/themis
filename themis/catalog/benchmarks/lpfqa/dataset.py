"""Dataset provider for m-a-p/LPFQA."""

from __future__ import annotations

from themis.specs.foundational import DatasetSpec

from ...datasets.common import (
    BuiltinDatasetProvider,
    CatalogNormalizedRows,
    _normalize_lpfqa_rows,
    load_huggingface_rows,
)


class BuiltinLPFQADatasetProvider(BuiltinDatasetProvider):
    def __init__(self, *, huggingface_loader=None) -> None:
        super().__init__(huggingface_loader=huggingface_loader or load_huggingface_rows)

    def normalize_loaded_rows(
        self,
        rows: list[dict[str, object]],
        dataset: DatasetSpec,
    ) -> CatalogNormalizedRows:
        return _normalize_lpfqa_rows(rows, dataset)
