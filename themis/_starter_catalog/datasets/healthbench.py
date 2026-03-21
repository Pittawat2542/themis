"""Dataset provider for openai/healthbench."""

from __future__ import annotations

from themis.specs.foundational import DatasetSpec

from .common import (
    BuiltinDatasetProvider,
    StarterNormalizedRows,
    _normalize_healthbench_rows,
    load_healthbench_rows,
)


class BuiltinHealthBenchDatasetProvider(BuiltinDatasetProvider):
    def __init__(self, *, huggingface_loader=None) -> None:
        super().__init__(huggingface_loader=huggingface_loader or load_healthbench_rows)

    def normalize_loaded_rows(
        self,
        rows: list[dict[str, object]],
        dataset: DatasetSpec,
    ) -> StarterNormalizedRows:
        return _normalize_healthbench_rows(rows, dataset)
