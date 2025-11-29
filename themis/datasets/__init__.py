"""Dataset helpers for Themis experiments."""

from __future__ import annotations

from typing import Any

from . import (
    competition_math,
    commonsense_qa,
    coqa,
    gpqa,
    gsm_symbolic,
    gsm8k,
    math500,
    med_qa,
    medmcqa,
    mmlu_pro,
    piqa,
    sciq,
    social_i_qa,
    super_gpqa,
)
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


def _create_gsm8k(options: dict[str, Any]) -> list[dict[str, Any]]:
    """Factory for GSM8K dataset."""
    samples = gsm8k.load_gsm8k(
        source=options.get("source", "huggingface"),
        data_dir=options.get("data_dir"),
        split=options.get("split", "test"),
        limit=options.get("limit"),
        subset=options.get("subset", "main"),
    )
    return [sample.to_generation_example() for sample in samples]


def _create_gpqa(options: dict[str, Any]) -> list[dict[str, Any]]:
    """Factory for GPQA dataset."""
    samples = gpqa.load_gpqa(
        source=options.get("source", "huggingface"),
        data_dir=options.get("data_dir"),
        split=options.get("split", "test"),
        limit=options.get("limit"),
        subset=options.get("subset", "gpqa_diamond"),
    )
    return [sample.to_generation_example() for sample in samples]


def _create_gsm_symbolic(options: dict[str, Any]) -> list[dict[str, Any]]:
    """Factory for GSM-Symbolic dataset."""
    samples = gsm_symbolic.load_gsm_symbolic(
        source=options.get("source", "huggingface"),
        data_dir=options.get("data_dir"),
        split=options.get("split", "test"),
        limit=options.get("limit"),
        subset=options.get("subset", "main"),
    )
    return [sample.to_generation_example() for sample in samples]


def _create_medmcqa(options: dict[str, Any]) -> list[dict[str, Any]]:
    """Factory for MedMCQA dataset."""
    samples = medmcqa.load_medmcqa(
        source=options.get("source", "huggingface"),
        data_dir=options.get("data_dir"),
        split=options.get("split", "test"),
        limit=options.get("limit"),
        subset=options.get("subset"),
    )
    return [sample.to_generation_example() for sample in samples]


def _create_med_qa(options: dict[str, Any]) -> list[dict[str, Any]]:
    """Factory for MedQA dataset."""
    samples = med_qa.load_med_qa(
        source=options.get("source", "huggingface"),
        data_dir=options.get("data_dir"),
        split=options.get("split", "test"),
        limit=options.get("limit"),
        subset=options.get("subset", "med_qa_en_bigbio_qa"),
    )
    return [sample.to_generation_example() for sample in samples]


def _create_sciq(options: dict[str, Any]) -> list[dict[str, Any]]:
    """Factory for SciQ dataset."""
    samples = sciq.load_sciq(
        source=options.get("source", "huggingface"),
        data_dir=options.get("data_dir"),
        split=options.get("split", "test"),
        limit=options.get("limit"),
    )
    return [sample.to_generation_example() for sample in samples]


def _create_commonsense_qa(options: dict[str, Any]) -> list[dict[str, Any]]:
    """Factory for CommonsenseQA dataset."""
    samples = commonsense_qa.load_commonsense_qa(
        source=options.get("source", "huggingface"),
        data_dir=options.get("data_dir"),
        split=options.get("split", "validation"),
        limit=options.get("limit"),
    )
    return [sample.to_generation_example() for sample in samples]


def _create_piqa(options: dict[str, Any]) -> list[dict[str, Any]]:
    """Factory for PIQA dataset."""
    samples = piqa.load_piqa(
        source=options.get("source", "huggingface"),
        data_dir=options.get("data_dir"),
        split=options.get("split", "validation"),
        limit=options.get("limit"),
    )
    return [sample.to_generation_example() for sample in samples]


def _create_social_i_qa(options: dict[str, Any]) -> list[dict[str, Any]]:
    """Factory for Social IQA dataset."""
    samples = social_i_qa.load_social_i_qa(
        source=options.get("source", "huggingface"),
        data_dir=options.get("data_dir"),
        split=options.get("split", "validation"),
        limit=options.get("limit"),
    )
    return [sample.to_generation_example() for sample in samples]


def _create_coqa(options: dict[str, Any]) -> list[dict[str, Any]]:
    """Factory for CoQA dataset."""
    samples = coqa.load_coqa(
        source=options.get("source", "huggingface"),
        data_dir=options.get("data_dir"),
        split=options.get("split", "validation"),
        limit=options.get("limit"),
    )
    return [sample.to_generation_example() for sample in samples]


# Auto-register built-in datasets
register_dataset("math500", _create_math500)
register_dataset("competition_math", _create_competition_math)
register_dataset("supergpqa", _create_super_gpqa)
register_dataset("mmlu-pro", _create_mmlu_pro)
register_dataset("gsm8k", _create_gsm8k)
register_dataset("gpqa", _create_gpqa)
register_dataset("gsm-symbolic", _create_gsm_symbolic)
register_dataset("medmcqa", _create_medmcqa)
register_dataset("med_qa", _create_med_qa)
register_dataset("sciq", _create_sciq)
register_dataset("commonsense_qa", _create_commonsense_qa)
register_dataset("piqa", _create_piqa)
register_dataset("social_i_qa", _create_social_i_qa)
register_dataset("coqa", _create_coqa)


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
    "commonsense_qa",
    "coqa",
    "gpqa",
    "gsm_symbolic",
    "gsm8k",
    "math500",
    "med_qa",
    "medmcqa",
    "mmlu_pro",
    "piqa",
    "sciq",
    "social_i_qa",
    "super_gpqa",
    # Registry functions
    "register_dataset",
    "unregister_dataset",
    "create_dataset",
    "list_datasets",
    "is_dataset_registered",
]
