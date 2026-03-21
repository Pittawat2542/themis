"""Dataset provider for cais/hle."""

from __future__ import annotations

from themis.specs.foundational import DatasetSpec

from .common import (
    BuiltinDatasetProvider,
    StarterNormalizedRows,
    _normalize_hle_rows,
    load_hle_rows,
)


class BuiltinHLEDatasetProvider(BuiltinDatasetProvider):
    def __init__(self, *, huggingface_loader=None) -> None:
        super().__init__(huggingface_loader=huggingface_loader or load_hle_rows)

    def normalize_loaded_rows(
        self,
        rows: list[dict[str, object]],
        dataset: DatasetSpec,
    ) -> StarterNormalizedRows:
        return _normalize_hle_rows(rows, dataset)
