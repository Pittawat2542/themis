"""Catalog dataset loading helpers."""

from __future__ import annotations

from pathlib import Path

from themis.types.json_types import JSONDict

from .. import datasets as _datasets


def load_local_rows(path: Path) -> list[dict[str, object]]:
    return _datasets.load_local_rows(path)


def load_huggingface_rows(
    dataset_id: str,
    split: str,
    revision: str | None = None,
    config_name: str | None = None,
) -> list[dict[str, object]]:
    return _datasets.load_huggingface_rows(
        dataset_id,
        split,
        revision,
        config_name,
    )


def inspect_huggingface_dataset(
    dataset_id: str,
    *,
    config_name: str | None = None,
    split: str = "test",
    revision: str | None = None,
    metadata_loader=None,
    row_loader=None,
    max_samples: int = 3,
) -> JSONDict:
    return _datasets.inspect_huggingface_dataset(
        dataset_id,
        config_name=config_name,
        split=split,
        revision=revision,
        metadata_loader=metadata_loader,
        row_loader=row_loader,
        max_samples=max_samples,
    )
