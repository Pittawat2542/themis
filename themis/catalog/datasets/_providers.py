"""Dataset provider base classes and query helpers for the catalog."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import random

from themis import PromptMessage
from themis.prompting import render_prompt_messages
from themis.specs.foundational import DatasetSpec
from themis.types.enums import DatasetSource, PromptRole, SamplingKind
from themis.types.json_types import JSONDict

from ._types import (
    CatalogNormalizedRows,
    CatalogRow,
    CatalogRowLoader,
    CatalogRowNormalizer,
)


class CatalogDatasetProvider:
    """Dataset provider covering inline, local-file, and HuggingFace catalogs."""

    def __init__(
        self,
        *,
        memory_rows: list[CatalogRow] | None = None,
        huggingface_loader: CatalogRowLoader | None = None,
        local_loader: Callable[[Path], list[CatalogRow]] | None = None,
        row_normalizer: CatalogRowNormalizer | None = None,
    ) -> None:
        from ._loaders import load_huggingface_rows, load_local_rows

        self._memory_rows = list(memory_rows or [])
        self._huggingface_loader = huggingface_loader or load_huggingface_rows
        self._local_loader = local_loader or load_local_rows
        self._row_normalizer = row_normalizer
        self._last_scan_stats: JSONDict = {}

    def scan(self, slice_spec, query):
        from ._loaders import _invoke_huggingface_loader

        dataset = slice_spec.dataset
        if dataset.source == DatasetSource.MEMORY:
            rows = list(self._memory_rows)
        elif dataset.source == DatasetSource.LOCAL:
            dataset_path = dataset.dataset_id or dataset.data_dir
            if dataset_path is None:
                raise ValueError("Local catalog datasets require a dataset path.")
            rows = self._local_loader(Path(dataset_path))
        elif dataset.source == DatasetSource.HUGGINGFACE:
            if dataset.dataset_id is None:
                raise ValueError("HuggingFace catalog datasets require a dataset_id.")
            rows = _invoke_huggingface_loader(
                self._huggingface_loader,
                dataset.dataset_id,
                dataset.split,
                dataset.revision,
                config_name=dataset.config_name,
            )
        else:
            raise ValueError(f"Unsupported catalog dataset source '{dataset.source}'.")
        normalized = self.prepare_rows(rows, dataset)
        filtered = _apply_query(normalized.rows, query)
        self._last_scan_stats = {
            **normalized.stats,
            "loaded_count": len(rows),
            "normalized_count": len(normalized.rows),
            "returned_count": len(filtered),
        }
        return filtered

    def prepare_rows(
        self,
        rows: list[CatalogRow],
        dataset_or_slice: object,
    ) -> CatalogNormalizedRows:
        return _normalize_rows_for_provider(
            rows, dataset_or_slice, self._row_normalizer
        )

    def last_scan_stats(self) -> JSONDict:
        return dict(self._last_scan_stats)


class BuiltinDatasetProvider(CatalogDatasetProvider):
    """Benchmark-aware Hugging Face dataset provider base used by built-ins."""

    def __init__(self, *, huggingface_loader=None) -> None:
        super().__init__(huggingface_loader=huggingface_loader)

    def scan(self, slice_spec, query):
        dataset = slice_spec.dataset
        if dataset.source != DatasetSource.HUGGINGFACE or dataset.dataset_id is None:
            raise ValueError(
                "Built-in benchmark dataset providers require a HuggingFace dataset_id."
            )
        rows = self.load_rows(dataset)
        normalized = self.prepare_rows(rows, slice_spec)
        filtered = _apply_query(normalized.rows, query)
        self._last_scan_stats = {
            **normalized.stats,
            "loaded_count": len(rows),
            "normalized_count": len(normalized.rows),
            "returned_count": len(filtered),
        }
        return filtered

    def load_rows(self, dataset: DatasetSpec) -> list[dict[str, object]]:
        from ._loaders import _invoke_huggingface_loader

        if dataset.dataset_id is None:
            raise ValueError(
                "Built-in HuggingFace dataset providers require a dataset_id."
            )
        return _invoke_huggingface_loader(
            self._huggingface_loader,
            dataset.dataset_id,
            dataset.split,
            dataset.revision,
            config_name=dataset.config_name,
        )

    def normalize_loaded_rows(
        self,
        rows: list[CatalogRow],
        dataset: object,
    ) -> CatalogNormalizedRows:
        return CatalogNormalizedRows(rows=[dict(row) for row in rows])

    def prepare_rows(
        self,
        rows: list[CatalogRow],
        dataset_or_slice: object,
    ) -> CatalogNormalizedRows:
        return _normalize_rows_for_provider(
            rows, dataset_or_slice, self.normalize_loaded_rows
        )


class BuiltinMCQDatasetProvider(BuiltinDatasetProvider):
    def __init__(
        self,
        *,
        metadata_keys: list[str],
        huggingface_loader=None,
    ) -> None:
        from ._loaders import load_huggingface_rows

        super().__init__(huggingface_loader=huggingface_loader or load_huggingface_rows)
        self._metadata_keys = list(metadata_keys)

    def normalize_loaded_rows(
        self,
        rows: list[CatalogRow],
        dataset: object,
    ) -> CatalogNormalizedRows:
        from ._normalizers import _normalize_mcq_rows

        return _normalize_mcq_rows(rows, dataset, metadata_keys=self._metadata_keys)


def _apply_query(rows: list[CatalogRow], query) -> list[CatalogRow]:
    filtered = list(rows)
    if query.metadata_filters:
        filtered = [
            row
            for row in filtered
            if all(
                _row_metadata_value(row, key) == value
                for key, value in query.metadata_filters.items()
            )
        ]
    if query.item_ids:
        wanted = set(query.item_ids)
        filtered = [row for row in filtered if str(row.get("item_id")) in wanted]
    if query.kind == SamplingKind.ALL:
        return filtered
    count = query.count or 0
    if query.kind == SamplingKind.SUBSET:
        if query.seed is None:
            return filtered[:count]
        if count >= len(filtered):
            return filtered
        return random.Random(query.seed).sample(filtered, count)
    if query.kind == SamplingKind.STRATIFIED:
        field = query.strata_field
        if not field:
            return filtered
        buckets: dict[str, list[CatalogRow]] = {}
        for row in filtered:
            buckets.setdefault(_row_metadata_value(row, field), []).append(row)
        randomizer = random.Random(query.seed)
        samples: list[CatalogRow] = []
        for bucket_rows in buckets.values():
            if len(bucket_rows) <= count:
                samples.extend(bucket_rows)
            else:
                samples.extend(randomizer.sample(bucket_rows, count))
        return samples
    return filtered


def _assign_missing_item_ids(rows: list[CatalogRow]) -> list[CatalogRow]:
    normalized: list[CatalogRow] = []
    for index, row in enumerate(rows, start=1):
        payload = dict(row)
        payload.setdefault("item_id", payload.get("id", f"item-{index}"))
        normalized.append(payload)
    return normalized


def _render_string_template(template: str, payload: CatalogRow) -> str:
    message = PromptMessage(role=PromptRole.USER, content=template)
    rendered = render_prompt_messages([message], payload, strict=True)[0]["content"]
    if not isinstance(rendered, str):
        raise ValueError("Catalog dataset transforms require string prompt content.")
    return rendered


def _apply_dataset_transforms(
    rows: list[CatalogRow],
    dataset: DatasetSpec,
) -> list[CatalogRow]:
    transformed_rows = [dict(row) for row in rows]
    for transform in dataset.transforms:
        if transform.kind == "rename":
            for row in transformed_rows:
                row[transform.field] = row.get(transform.source_field)
            continue
        if transform.kind == "jinja":
            for row in transformed_rows:
                row[transform.field] = _render_string_template(transform.template, row)
            continue
        if transform.kind == "python":
            raise ValueError(
                "DatasetSpec python transforms are not supported by CatalogDatasetProvider."
            )
    return transformed_rows


def _normalize_rows_for_provider(
    rows: list[CatalogRow],
    dataset_or_slice: object,
    row_normalizer: CatalogRowNormalizer | None,
) -> CatalogNormalizedRows:
    dataset = getattr(dataset_or_slice, "dataset", dataset_or_slice)
    if not isinstance(dataset, DatasetSpec):
        raise ValueError("Catalog dataset normalization requires a DatasetSpec.")
    assigned = _assign_missing_item_ids(rows)
    normalized = (
        row_normalizer(assigned, dataset_or_slice)
        if row_normalizer is not None
        else CatalogNormalizedRows(rows=assigned)
    )
    transformed = _apply_dataset_transforms(normalized.rows, dataset)
    return CatalogNormalizedRows(rows=transformed, stats=normalized.stats)


def _row_metadata_value(row: CatalogRow, key: str) -> str:
    metadata = row.get("metadata")
    if isinstance(metadata, dict) and key in metadata:
        return str(metadata[key])
    return str(row.get(key, ""))
