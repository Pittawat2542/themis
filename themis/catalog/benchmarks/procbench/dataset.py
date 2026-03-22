"""Dataset provider for ifujisawa/procbench."""

from __future__ import annotations

from ...datasets.common import (
    BuiltinDatasetProvider,
    CatalogNormalizedRows,
    _normalize_procbench_rows,
)


class BuiltinProcbenchDatasetProvider(BuiltinDatasetProvider):
    def normalize_loaded_rows(
        self,
        rows: list[dict[str, object]],
        dataset: object,
    ) -> CatalogNormalizedRows:
        return _normalize_procbench_rows(rows, dataset)
