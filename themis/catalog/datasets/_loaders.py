"""Local/Hugging Face dataset loading helpers for the catalog."""

from __future__ import annotations

import csv
import inspect
import json
from pathlib import Path

from themis._optional import import_optional

from ._providers import _assign_missing_item_ids
from ._types import CatalogRow, CatalogRowLoader


def load_local_rows(path: Path) -> list[CatalogRow]:
    """Load catalog dataset rows from JSONL or CSV."""

    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, object]] = []
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"JSONL row {line_number} in {path.name} must be an object."
                )
            rows.append(dict(payload))
        return _assign_missing_item_ids(rows)
    if path.suffix.lower() == ".csv":
        with path.open(newline="") as fh:
            return _assign_missing_item_ids([dict(row) for row in csv.DictReader(fh)])
    raise ValueError("Catalog local datasets must use .jsonl or .csv.")


def load_huggingface_rows(
    dataset_id: str,
    split: str,
    revision: str | None = None,
    config_name: str | None = None,
    *,
    datasets_module=None,
) -> list[CatalogRow]:
    """Load catalog dataset rows from a HuggingFace dataset identifier."""

    datasets = datasets_module or import_optional("datasets", extra="datasets")
    try:
        dataset = _datasets_load_dataset(
            datasets,
            dataset_id,
            split=split,
            revision=revision,
            config_name=config_name,
        )
    except Exception as exc:
        if _should_retry_huggingface_streaming(dataset_id, exc, datasets):
            dataset = _datasets_load_dataset(
                datasets,
                dataset_id,
                split=split,
                revision=revision,
                config_name=config_name,
                streaming=True,
            )
        else:
            raise
    dataset = _prepare_huggingface_dataset_for_iteration(dataset, datasets)
    return _assign_missing_item_ids([dict(row) for row in dataset])


def load_hle_rows(
    dataset_id: str,
    split: str,
    revision: str | None = None,
    config_name: str | None = None,
    *,
    datasets_module=None,
) -> list[CatalogRow]:
    """Load HLE rows without forcing image decoding during iteration."""

    datasets = datasets_module or import_optional("datasets", extra="datasets")
    dataset = _datasets_load_dataset(
        datasets,
        dataset_id,
        split=split,
        revision=revision,
        config_name=config_name,
    )
    dataset = _prepare_huggingface_dataset_for_iteration(dataset, datasets)
    return _assign_missing_item_ids([dict(row) for row in dataset])


def load_healthbench_rows(
    dataset_id: str,
    split: str,
    revision: str | None = None,
    config_name: str | None = None,
    *,
    datasets_module=None,
) -> list[CatalogRow]:
    """Load HealthBench rows with a dataset-specific streaming fallback."""

    datasets = datasets_module or import_optional("datasets", extra="datasets")
    try:
        dataset = _datasets_load_dataset(
            datasets,
            dataset_id,
            split=split,
            revision=revision,
            config_name=config_name,
        )
    except Exception as exc:
        if _should_retry_huggingface_streaming(dataset_id, exc, datasets):
            dataset = _datasets_load_dataset(
                datasets,
                dataset_id,
                split=split,
                revision=revision,
                config_name=config_name,
                streaming=True,
            )
        else:
            raise
    return _assign_missing_item_ids([dict(row) for row in dataset])


def _datasets_load_dataset(
    datasets_module,
    dataset_id: str,
    *,
    split: str,
    revision: str | None,
    config_name: str | None,
    streaming: bool = False,
):
    kwargs: dict[str, object] = {
        "split": split,
        "revision": revision,
    }
    if streaming:
        kwargs["streaming"] = True
    if config_name is None:
        return datasets_module.load_dataset(dataset_id, **kwargs)
    return datasets_module.load_dataset(dataset_id, config_name, **kwargs)


def _invoke_huggingface_loader(
    loader: CatalogRowLoader,
    dataset_id: str,
    split: str,
    revision: str | None,
    *,
    config_name: str | None,
) -> list[CatalogRow]:
    signature = inspect.signature(loader)
    if "config_name" in signature.parameters:
        return loader(
            dataset_id,
            split,
            revision,
            config_name=config_name,
        )
    return loader(dataset_id, split, revision)


def _prepare_huggingface_dataset_for_iteration(dataset, datasets_module):
    image_type = getattr(datasets_module, "Image", None)
    features = getattr(dataset, "features", None)
    if (
        image_type is None
        or features is None
        or not hasattr(features, "items")
        or not hasattr(dataset, "cast_column")
    ):
        return dataset
    prepared = dataset
    for column_name, feature in features.items():
        if isinstance(feature, image_type) and bool(getattr(feature, "decode", False)):
            prepared = prepared.cast_column(column_name, image_type(decode=False))
    return prepared


def _should_retry_huggingface_streaming(
    dataset_id: str,
    exc: Exception,
    datasets_module,
) -> bool:
    if dataset_id != "openai/healthbench":
        return False
    dataset_generation_error = getattr(datasets_module, "DatasetGenerationError", None)
    if dataset_generation_error is not None and isinstance(
        exc, dataset_generation_error
    ):
        return True
    exceptions_module = getattr(datasets_module, "exceptions", None)
    nested_dataset_generation_error = getattr(
        exceptions_module, "DatasetGenerationError", None
    )
    if nested_dataset_generation_error is None:
        return False
    return isinstance(exc, nested_dataset_generation_error)


def _is_dataset_generation_error(exc: Exception, datasets_module) -> bool:
    dataset_generation_error = getattr(datasets_module, "DatasetGenerationError", None)
    if dataset_generation_error is not None and isinstance(
        exc, dataset_generation_error
    ):
        return True
    exceptions_module = getattr(datasets_module, "exceptions", None)
    nested_dataset_generation_error = getattr(
        exceptions_module, "DatasetGenerationError", None
    )
    return bool(
        nested_dataset_generation_error is not None
        and isinstance(exc, nested_dataset_generation_error)
    )
