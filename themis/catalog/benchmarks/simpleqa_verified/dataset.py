"""Dataset provider for google/simpleqa-verified."""

from __future__ import annotations

from ...datasets._loaders import load_huggingface_rows
from ...datasets._normalizers import _normalize_simpleqa_rows
from ...datasets._providers import BuiltinDatasetProvider
from ...datasets._types import CatalogNormalizedRows


class BuiltinSimpleQAVerifiedDatasetProvider(BuiltinDatasetProvider):
    def __init__(self, *, huggingface_loader=None) -> None:
        super().__init__(huggingface_loader=huggingface_loader or load_huggingface_rows)

    def normalize_loaded_rows(
        self,
        rows: list[dict[str, object]],
        dataset: object,
    ) -> CatalogNormalizedRows:
        return _normalize_simpleqa_rows(rows, dataset)
