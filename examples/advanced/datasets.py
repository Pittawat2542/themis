"""Dataset helpers for the advanced experiment."""

from __future__ import annotations

from typing import List

from experiments.example import datasets as base_datasets
from experiments.example.config import DatasetConfig


def load_all_datasets(dataset_configs: List[DatasetConfig]) -> List[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for config in dataset_configs:
        rows.extend(base_datasets.load_dataset(config))
    return rows


__all__ = ["load_all_datasets"]
