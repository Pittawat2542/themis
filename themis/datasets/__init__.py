"""Dataset helpers for Themis experiments."""

from __future__ import annotations

from typing import Any

from . import competition_math, math500, mmlu_pro, super_gpqa
from .registry import (
    create_dataset,
    is_dataset_registered,
    list_datasets,
    register_dataset,
    unregister_dataset,
)

# Factory functions for built-in datasets


def _create_math500(options: dict[str, Any]) -> list[dict[str, Any]]:
    """Factory for MATH-500 dataset."""
    samples = math500.load_math500(
        source=options.get("source", "huggingface"),
        data_dir=options.get("data_dir"),
        split=options.get("split", "test"),
        limit=options.get("limit"),
        subjects=options.get("subjects"),
    )
    return [sample.to_generation_example() for sample in samples]


def _create_competition_math(options: dict[str, Any]) -> list[dict[str, Any]]:
    """Factory for competition math datasets (AIME, AMC, etc.)."""
    # Get dataset and subset from options
    dataset = options.get("dataset")
    if not dataset:
        raise ValueError(
            "Competition math requires 'dataset' option "
            "(e.g., 'math-ai/aime24', 'math-ai/amc23')"
        )

    samples = competition_math.load_competition_math(
        dataset=dataset,
        subset=options.get("subset"),
        source=options.get("source", "huggingface"),
        data_dir=options.get("data_dir"),
        split=options.get("split", "test"),
        limit=options.get("limit"),
        subjects=options.get("subjects"),
    )
    return [sample.to_generation_example() for sample in samples]


def _create_super_gpqa(options: dict[str, Any]) -> list[dict[str, Any]]:
    """Factory for SuperGPQA dataset."""
    samples = super_gpqa.load_super_gpqa(
        source=options.get("source", "huggingface"),
        data_dir=options.get("data_dir"),
        split=options.get("split", "test"),
        limit=options.get("limit"),
        subjects=options.get("subjects"),
    )
    return [sample.to_generation_example() for sample in samples]


def _create_mmlu_pro(options: dict[str, Any]) -> list[dict[str, Any]]:
    """Factory for MMLU-Pro dataset."""
    samples = mmlu_pro.load_mmlu_pro(
        source=options.get("source", "huggingface"),
        data_dir=options.get("data_dir"),
        split=options.get("split", "test"),
        limit=options.get("limit"),
        subjects=options.get("subjects"),
    )
    return [sample.to_generation_example() for sample in samples]


# Auto-register built-in datasets
register_dataset("math500", _create_math500)
register_dataset("competition_math", _create_competition_math)
register_dataset("supergpqa", _create_super_gpqa)
register_dataset("mmlu-pro", _create_mmlu_pro)

# Also register specific competition datasets as aliases
def _create_aime24(options: dict[str, Any]) -> list[dict[str, Any]]:
    return _create_competition_math({**options, "dataset": "math-ai/aime24"})


def _create_aime25(options: dict[str, Any]) -> list[dict[str, Any]]:
    return _create_competition_math({**options, "dataset": "math-ai/aime25"})


def _create_amc23(options: dict[str, Any]) -> list[dict[str, Any]]:
    return _create_competition_math({**options, "dataset": "math-ai/amc23"})


def _create_olympiadbench(options: dict[str, Any]) -> list[dict[str, Any]]:
    return _create_competition_math({**options, "dataset": "math-ai/olympiadbench"})


def _create_beyondaime(options: dict[str, Any]) -> list[dict[str, Any]]:
    return _create_competition_math({**options, "dataset": "ByteDance-Seed/BeyondAIME"})


register_dataset("aime24", _create_aime24)
register_dataset("aime25", _create_aime25)
register_dataset("amc23", _create_amc23)
register_dataset("olympiadbench", _create_olympiadbench)
register_dataset("beyondaime", _create_beyondaime)

__all__ = [
    # Legacy module exports
    "competition_math",
    "math500",
    "mmlu_pro",
    "super_gpqa",
    # Registry functions
    "register_dataset",
    "unregister_dataset",
    "create_dataset",
    "list_datasets",
    "is_dataset_registered",
]
