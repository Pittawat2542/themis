"""Dataset provider for ifujisawa/procbench."""

from __future__ import annotations

from ...datasets._normalizers import _normalize_procbench_rows
from ...datasets._providers import BuiltinDatasetProvider
from ...datasets._types import CatalogNormalizedRows


class BuiltinProcbenchDatasetProvider(BuiltinDatasetProvider):
    def normalize_loaded_rows(
        self,
        rows: list[dict[str, object]],
        dataset: object,
    ) -> CatalogNormalizedRows:
        return _normalize_procbench_rows(rows, dataset)
