"""Dataset helpers for Themis experiments."""

from __future__ import annotations

from typing import Any

from themis.exceptions import DatasetError
from .registry import (
    create_dataset,
    is_dataset_registered,
    list_datasets,
    register_dataset,
    unregister_dataset,
)

# ---------------------------------------------------------------------------
# Factory functions for built-in datasets
# All dataset module imports are deferred to inside the factory so that
# `import themis` does not trigger heavy HuggingFace / network IO.
# ---------------------------------------------------------------------------


def _create_math500(options: dict[str, Any]) -> list[dict[str, Any]]:
    """Factory for MATH-500 dataset."""
    from themis.datasets.math500 import load_math500

    samples = load_math500(
        source=options.get("source", "huggingface"),
        data_dir=options.get("data_dir"),
        split=options.get("split", "test"),
        limit=options.get("limit"),
        subjects=options.get("subjects"),
    )
    return [sample.to_generation_example() for sample in samples]


def _create_competition_math(options: dict[str, Any]) -> list[dict[str, Any]]:
    """Factory for competition math datasets (AIME, AMC, etc.)."""
    from themis.datasets.competition_math import load_competition_math

    # Get dataset and subset from options
    dataset = options.get("dataset")
    if not dataset:
        raise DatasetError(
            "Competition math requires 'dataset' option "
            "(e.g., 'math-ai/aime24', 'math-ai/amc23')"
        )

    samples = load_competition_math(
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
    from themis.datasets.super_gpqa import load_super_gpqa

    samples = load_super_gpqa(
        source=options.get("source", "huggingface"),
        data_dir=options.get("data_dir"),
        split=options.get("split", "test"),
        limit=options.get("limit"),
        subjects=options.get("subjects"),
    )
    return [sample.to_generation_example() for sample in samples]


def _create_mmlu_pro(options: dict[str, Any]) -> list[dict[str, Any]]:
    """Factory for MMLU-Pro dataset."""
    from themis.datasets.mmlu_pro import load_mmlu_pro

    samples = load_mmlu_pro(
        source=options.get("source", "huggingface"),
        data_dir=options.get("data_dir"),
        split=options.get("split", "test"),
        limit=options.get("limit"),
        subjects=options.get("subjects"),
    )
    return [sample.to_generation_example() for sample in samples]


def _create_gsm8k(options: dict[str, Any]) -> list[dict[str, Any]]:
    """Factory for GSM8K dataset."""
    from themis.datasets.gsm8k import load_gsm8k

    samples = load_gsm8k(
        source=options.get("source", "huggingface"),
        data_dir=options.get("data_dir"),
        split=options.get("split", "test"),
        limit=options.get("limit"),
        subset=options.get("subset", "main"),
    )
    return [sample.to_generation_example() for sample in samples]


def _create_gpqa(options: dict[str, Any]) -> list[dict[str, Any]]:
    """Factory for GPQA dataset."""
    from themis.datasets.gpqa import load_gpqa

    samples = load_gpqa(
        source=options.get("source", "huggingface"),
        data_dir=options.get("data_dir"),
        split=options.get("split", "test"),
        limit=options.get("limit"),
        subset=options.get("subset", "gpqa_diamond"),
    )
    return [sample.to_generation_example() for sample in samples]


def _create_gsm_symbolic(options: dict[str, Any]) -> list[dict[str, Any]]:
    """Factory for GSM-Symbolic dataset."""
    from themis.datasets.gsm_symbolic import load_gsm_symbolic

    samples = load_gsm_symbolic(
        source=options.get("source", "huggingface"),
        data_dir=options.get("data_dir"),
        split=options.get("split", "test"),
        limit=options.get("limit"),
        subset=options.get("subset", "main"),
    )
    return [sample.to_generation_example() for sample in samples]


def _create_medmcqa(options: dict[str, Any]) -> list[dict[str, Any]]:
    """Factory for MedMCQA dataset."""
    from themis.datasets.medmcqa import load_medmcqa

    samples = load_medmcqa(
        source=options.get("source", "huggingface"),
        data_dir=options.get("data_dir"),
        split=options.get("split", "test"),
        limit=options.get("limit"),
        subset=options.get("subset"),
    )
    return [sample.to_generation_example() for sample in samples]


def _create_med_qa(options: dict[str, Any]) -> list[dict[str, Any]]:
    """Factory for MedQA dataset."""
    from themis.datasets.med_qa import load_med_qa

    samples = load_med_qa(
        source=options.get("source", "huggingface"),
        data_dir=options.get("data_dir"),
        split=options.get("split", "test"),
        limit=options.get("limit"),
        subset=options.get("subset", "med_qa_en_bigbio_qa"),
    )
    return [sample.to_generation_example() for sample in samples]


def _create_sciq(options: dict[str, Any]) -> list[dict[str, Any]]:
    """Factory for SciQ dataset."""
    from themis.datasets.sciq import load_sciq

    samples = load_sciq(
        source=options.get("source", "huggingface"),
        data_dir=options.get("data_dir"),
        split=options.get("split", "test"),
        limit=options.get("limit"),
    )
    return [sample.to_generation_example() for sample in samples]


def _create_commonsense_qa(options: dict[str, Any]) -> list[dict[str, Any]]:
    """Factory for CommonsenseQA dataset."""
    from themis.datasets.commonsense_qa import load_commonsense_qa

    samples = load_commonsense_qa(
        source=options.get("source", "huggingface"),
        data_dir=options.get("data_dir"),
        split=options.get("split", "validation"),
        limit=options.get("limit"),
    )
    return [sample.to_generation_example() for sample in samples]


def _create_piqa(options: dict[str, Any]) -> list[dict[str, Any]]:
    """Factory for PIQA dataset."""
    from themis.datasets.piqa import load_piqa

    samples = load_piqa(
        source=options.get("source", "huggingface"),
        data_dir=options.get("data_dir"),
        split=options.get("split", "validation"),
        limit=options.get("limit"),
    )
    return [sample.to_generation_example() for sample in samples]


def _create_social_i_qa(options: dict[str, Any]) -> list[dict[str, Any]]:
    """Factory for Social IQA dataset."""
    from themis.datasets.social_i_qa import load_social_i_qa

    samples = load_social_i_qa(
        source=options.get("source", "huggingface"),
        data_dir=options.get("data_dir"),
        split=options.get("split", "validation"),
        limit=options.get("limit"),
    )
    return [sample.to_generation_example() for sample in samples]


def _create_coqa(options: dict[str, Any]) -> list[dict[str, Any]]:
    """Factory for CoQA dataset."""
    from themis.datasets.coqa import load_coqa

    samples = load_coqa(
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
    # Registry functions
    "register_dataset",
    "unregister_dataset",
    "create_dataset",
    "list_datasets",
    "is_dataset_registered",
]
