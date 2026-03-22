"""Dataset providers for built-in short-answer math benchmarks."""

from __future__ import annotations

from themis.specs.foundational import DatasetSpec

from .common import (
    BuiltinDatasetProvider,
    CatalogNormalizedRows,
    _normalize_imo_answerbench_rows,
    _normalize_math_short_answer_rows,
)


class BuiltinMathArenaDatasetProvider(BuiltinDatasetProvider):
    def normalize_loaded_rows(
        self,
        rows: list[dict[str, object]],
        dataset: DatasetSpec,
    ) -> CatalogNormalizedRows:
        return _normalize_math_short_answer_rows(rows, dataset)


class BuiltinBeyondAIMEDatasetProvider(BuiltinDatasetProvider):
    def normalize_loaded_rows(
        self,
        rows: list[dict[str, object]],
        dataset: DatasetSpec,
    ) -> CatalogNormalizedRows:
        return _normalize_math_short_answer_rows(rows, dataset)


class BuiltinIMOAnswerBenchDatasetProvider(BuiltinDatasetProvider):
    def normalize_loaded_rows(
        self,
        rows: list[dict[str, object]],
        dataset: DatasetSpec,
    ) -> CatalogNormalizedRows:
        return _normalize_imo_answerbench_rows(rows, dataset)
