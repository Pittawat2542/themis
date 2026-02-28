"""Benchmark preset configurations.

This module provides pre-configured settings for popular benchmarks,
including prompts, metrics, extractors, and data loaders.
"""

from __future__ import annotations

import string
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from themis.generation.templates import PromptTemplate
from themis.exceptions import ConfigurationError
from themis.interfaces import Extractor, Metric


@dataclass
class BenchmarkPreset:
    """Configuration preset for a benchmark.

    Attributes:
        name: Benchmark name
        prompt_template: Default prompt template
        metrics: List of metric instances
        extractor: Output extractor
        dataset_loader: Function to load the dataset
        metadata_fields: Fields to include in task metadata
        reference_field: Field containing the reference answer
        dataset_id_field: Field containing the sample ID
        description: Human-readable description
    """

    name: str
    prompt_template: PromptTemplate
    metrics: list[Metric]
    extractor: Extractor
    dataset_loader: Callable[[int | None], Sequence[dict[str, Any]]]
    metadata_fields: tuple[str, ...] = field(default_factory=tuple)
    reference_field: str = "answer"
    dataset_id_field: str = "id"
    description: str = ""

    def load_dataset(self, limit: int | None = None) -> Sequence[dict[str, Any]]:
        """Load the benchmark dataset.

        Args:
            limit: Maximum number of samples to load

        Returns:
            List of dataset samples
        """
        return self.dataset_loader(limit)


# Registry of benchmark presets
_BENCHMARK_REGISTRY: dict[str, BenchmarkPreset] = {}
_REGISTRY_INITIALIZED = False


def _to_dict_samples(samples: Sequence[Any]) -> list[dict[str, Any]]:
    return [
        sample.to_generation_example()
        if hasattr(sample, "to_generation_example")
        else dict(sample)
        for sample in samples
    ]


def _format_mcq_options(choices: Sequence[str], labels: Sequence[str]) -> str:
    return "\n".join(f"{label}. {choice}" for label, choice in zip(labels, choices))


def _normalize_mcq_answer(
    answer: Any,
    choices: Sequence[str],
    labels: Sequence[str],
) -> str:
    if answer is None:
        return ""
    if isinstance(answer, bool):
        return str(answer)
    if isinstance(answer, (int, float)):
        index = int(answer)
        if 1 <= index <= len(choices):
            return labels[index - 1]
        if 0 <= index < len(choices):
            return labels[index]
    text = str(answer).strip()
    if not text:
        return ""
    lowered = text.lower()
    if lowered.startswith("option "):
        text = text.split(" ", 1)[-1].strip()
    if lowered.startswith("choice "):
        text = text.split(" ", 1)[-1].strip()
    if len(text) >= 2 and text[1] in {".", ")", ":", "-"}:
        text = text[0]
    if len(text) == 1 and text.isalpha():
        letter = text.upper()
        if letter in labels:
            return letter
    for idx, choice in enumerate(choices):
        if text == str(choice).strip():
            return labels[idx]
    for idx, choice in enumerate(choices):
        if text.lower() == str(choice).strip().lower():
            return labels[idx]
    return text


def _normalize_mcq_samples(samples: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for sample in samples:
        row = dict(sample)
        choices = row.get("choices") or row.get("options")
        if not isinstance(choices, (list, tuple)):
            normalized.append(row)
            continue
        choices_list = [str(choice) for choice in choices]
        labels = row.get("choice_labels")
        if isinstance(labels, (list, tuple)) and labels:
            labels_list = [str(label) for label in labels][: len(choices_list)]
        else:
            labels_list = list(string.ascii_uppercase[: len(choices_list)])
        row["choices"] = choices_list
        row["choice_labels"] = labels_list
        row["options"] = _format_mcq_options(choices_list, labels_list)
        row["answer"] = _normalize_mcq_answer(
            row.get("answer"),
            choices_list,
            labels_list,
        )
        normalized.append(row)
    return normalized


def _ensure_registry_initialized() -> None:
    """Initialize benchmark registry on first use (lazy loading)."""
    global _REGISTRY_INITIALIZED
    if not _REGISTRY_INITIALIZED:
        _register_all_benchmarks()
        _REGISTRY_INITIALIZED = True


def register_benchmark(preset: BenchmarkPreset) -> None:
    """Register a benchmark preset.

    Args:
        preset: Benchmark preset configuration
    """
    _BENCHMARK_REGISTRY[preset.name.lower()] = preset


def get_benchmark_preset(name: str) -> BenchmarkPreset:
    """Get a benchmark preset by name.

    Args:
        name: Benchmark name (case-insensitive)

    Returns:
        Benchmark preset

    Raises:
        ConfigurationError: If benchmark is not found
    """
    _ensure_registry_initialized()

    name_lower = name.lower()
    if name_lower not in _BENCHMARK_REGISTRY:
        available = ", ".join(sorted(_BENCHMARK_REGISTRY.keys()))
        raise ConfigurationError(
            f"Unknown benchmark: {name}. Available benchmarks: {available}"
        )
    return _BENCHMARK_REGISTRY[name_lower]


def list_benchmarks() -> list[str]:
    """List all registered benchmark names.

    Returns:
        Sorted list of benchmark names
    """
    _ensure_registry_initialized()
    return sorted(_BENCHMARK_REGISTRY.keys())


# ============================================================================
# Generic Benchmark Factories
# ============================================================================


def _create_math_preset(
    name: str,
    dataset_name: str,
    prompt_template_str: str,
    description: str,
    dataset_kwargs: dict[str, Any] | None = None,
    metadata_fields: tuple[str, ...] = ("subject", "level"),
) -> BenchmarkPreset:
    """Create a generic math benchmark preset."""
    from themis.datasets.competition_math import load_competition_math
    from themis.evaluation.extractors.math_verify_extractor import MathVerifyExtractor
    from themis.evaluation.metrics.math_verify_accuracy import MathVerifyAccuracy

    kwargs = dataset_kwargs or {}

    def load_dataset(limit: int | None = None) -> Sequence[dict[str, Any]]:
        # Some loaders like gsm8k have specific loader functions, but for
        # generic competition math we use load_competition_math. We handle
        # specific hardcoded loaders below via dataset_name mapping if needed,
        # otherwise default to load_competition_math.
        if dataset_name == "math500":
            from themis.datasets.math500 import load_math500

            samples = load_math500(limit=limit, **kwargs)
        elif dataset_name == "gsm8k":
            from themis.datasets.gsm8k import load_gsm8k

            samples = load_gsm8k(limit=limit, **kwargs)
        elif dataset_name == "gsm-symbolic":
            from themis.datasets.gsm_symbolic import load_gsm_symbolic

            samples = load_gsm_symbolic(limit=limit, **kwargs)
        else:
            samples = load_competition_math(
                dataset=dataset_name,
                limit=limit,
                **kwargs,
            )
        return _to_dict_samples(samples)

    prompt_template = PromptTemplate(
        name=f"{name}-zero-shot",
        template=prompt_template_str,
    )

    return BenchmarkPreset(
        name=name,
        prompt_template=prompt_template,
        metrics=[MathVerifyAccuracy()],
        extractor=MathVerifyExtractor(),
        dataset_loader=load_dataset,
        metadata_fields=metadata_fields,
        reference_field="answer" if name != "math500" else "solution",
        dataset_id_field="unique_id",
        description=description,
    )


def _create_mcq_preset(
    name: str,
    dataset_name: str,
    prompt_template_str: str,
    description: str,
    dataset_kwargs: dict[str, Any] | None = None,
    metadata_fields: tuple[str, ...] = ("subject",),
    reference_field: str = "answer",
) -> BenchmarkPreset:
    """Create a generic MCQ benchmark preset."""
    from themis.evaluation.extractors.identity_extractor import IdentityExtractor
    from themis.evaluation.metrics.exact_match import ExactMatch

    kwargs = dataset_kwargs or {}

    def load_dataset(limit: int | None = None) -> Sequence[dict[str, Any]]:
        if dataset_name == "mmlu-pro":
            from themis.datasets.mmlu_pro import load_mmlu_pro

            samples = load_mmlu_pro(limit=limit, **kwargs)
        elif dataset_name == "super_gpqa":
            from themis.datasets.super_gpqa import load_super_gpqa

            samples = load_super_gpqa(limit=limit, **kwargs)
        elif dataset_name == "gpqa":
            from themis.datasets.gpqa import load_gpqa

            samples = load_gpqa(limit=limit, **kwargs)
        elif dataset_name == "medmcqa":
            from themis.datasets.medmcqa import load_medmcqa

            samples = load_medmcqa(limit=limit, **kwargs)
        elif dataset_name == "med_qa":
            from themis.datasets.med_qa import load_med_qa

            samples = load_med_qa(limit=limit, **kwargs)
        elif dataset_name == "sciq":
            from themis.datasets.sciq import load_sciq

            samples = load_sciq(limit=limit, **kwargs)
        elif dataset_name == "commonsense_qa":
            from themis.datasets.commonsense_qa import load_commonsense_qa

            samples = load_commonsense_qa(limit=limit, **kwargs)
        elif dataset_name == "piqa":
            from themis.datasets.piqa import load_piqa

            samples = load_piqa(limit=limit, **kwargs)
        elif dataset_name == "social_i_qa":
            from themis.datasets.social_i_qa import load_social_i_qa

            samples = load_social_i_qa(limit=limit, **kwargs)
        elif dataset_name == "coqa":
            from themis.datasets.coqa import load_coqa

            samples = load_coqa(limit=limit, **kwargs)
            return _to_dict_samples(samples)  # CoQA doesn't need _normalize_mcq_samples
        else:
            raise ConfigurationError(f"Unsupported MCQ dataset loader: {dataset_name}")

        return _normalize_mcq_samples(_to_dict_samples(samples))

    prompt_template = PromptTemplate(
        name=f"{name}-zero-shot",
        template=prompt_template_str,
    )

    return BenchmarkPreset(
        name=name,
        prompt_template=prompt_template,
        metrics=[ExactMatch()],
        extractor=IdentityExtractor(),
        dataset_loader=load_dataset,
        metadata_fields=metadata_fields,
        reference_field=reference_field,
        dataset_id_field="unique_id",
        description=description,
    )


# ============================================================================
# Math Benchmarks
# ============================================================================


def _create_math500_preset() -> BenchmarkPreset:
    """Create MATH-500 benchmark preset."""
    return _create_math_preset(
        name="math500",
        dataset_name="math500",
        prompt_template_str=(
            "Solve the following math problem step by step. "
            "Put your final answer in \\boxed{{}}.\n\n"
            "Problem: {problem}\n\n"
            "Solution:"
        ),
        description="MATH-500 dataset with 500 competition math problems",
    )


def _create_gsm8k_preset() -> BenchmarkPreset:
    """Create GSM8K benchmark preset."""
    return _create_math_preset(
        name="gsm8k",
        dataset_name="gsm8k",
        prompt_template_str="Solve this math problem step by step.\n\nQ: {question}\nA:",
        description="GSM8K dataset with grade school math word problems",
        dataset_kwargs={"source": "huggingface", "split": "test"},
        metadata_fields=(),
    )


def _create_aime24_preset() -> BenchmarkPreset:
    """Create AIME 2024 benchmark preset."""
    return _create_math_preset(
        name="aime24",
        dataset_name="math-ai/aime24",
        prompt_template_str=(
            "Solve the following AIME problem. "
            "Your answer should be a number between 000 and 999.\n\n"
            "Problem: {problem}\n\n"
            "Solution:"
        ),
        description="AIME 2024 competition math problems",
        dataset_kwargs={"source": "huggingface", "split": "test"},
        metadata_fields=("subject",),
    )


def _create_gsm_symbolic_preset() -> BenchmarkPreset:
    """Create GSM-Symbolic benchmark preset."""
    return _create_math_preset(
        name="gsm-symbolic",
        dataset_name="gsm-symbolic",
        prompt_template_str="Solve this math problem step by step.\n\nQ: {question}\nA:",
        description="GSM-Symbolic dataset for algebraic word problems",
        dataset_kwargs={"source": "huggingface", "split": "test"},
        metadata_fields=(),
    )


def _create_aime25_preset() -> BenchmarkPreset:
    """Create AIME 2025 benchmark preset."""
    return _create_math_preset(
        name="aime25",
        dataset_name="math-ai/aime25",
        prompt_template_str=(
            "Solve the following AIME problem. "
            "Your answer should be a number between 000 and 999.\n\n"
            "Problem: {problem}\n\n"
            "Solution:"
        ),
        description="AIME 2025 competition math problems",
        dataset_kwargs={"source": "huggingface", "split": "test"},
    )


def _create_amc23_preset() -> BenchmarkPreset:
    """Create AMC 2023 benchmark preset."""
    return _create_math_preset(
        name="amc23",
        dataset_name="math-ai/amc23",
        prompt_template_str=(
            "Solve the following AMC problem. "
            "Give only the final answer.\n\n"
            "Problem: {problem}\n\n"
            "Answer:"
        ),
        description="AMC 2023 competition math problems",
        dataset_kwargs={"source": "huggingface", "split": "test"},
    )


def _create_olympiadbench_preset() -> BenchmarkPreset:
    """Create OlympiadBench benchmark preset."""
    return _create_math_preset(
        name="olympiadbench",
        dataset_name="math-ai/olympiadbench",
        prompt_template_str=(
            "Solve the following olympiad-style math problem. "
            "Show reasoning briefly, then give the final answer.\n\n"
            "Problem: {problem}\n\n"
            "Solution:"
        ),
        description="OlympiadBench competition math benchmark",
        dataset_kwargs={"source": "huggingface", "split": "test"},
    )


def _create_beyondaime_preset() -> BenchmarkPreset:
    """Create BeyondAIME benchmark preset."""
    return _create_math_preset(
        name="beyondaime",
        dataset_name="ByteDance-Seed/BeyondAIME",
        prompt_template_str=(
            "Solve the following advanced contest math problem. "
            "Provide the final answer clearly.\n\n"
            "Problem: {problem}\n\n"
            "Answer:"
        ),
        description="BeyondAIME advanced competition math problems",
        dataset_kwargs={"source": "huggingface", "split": "test"},
    )


# ============================================================================
# MCQ Benchmarks
# ============================================================================


def _create_mmlu_pro_preset() -> BenchmarkPreset:
    """Create MMLU-Pro benchmark preset."""
    return _create_mcq_preset(
        name="mmlu-pro",
        dataset_name="mmlu-pro",
        prompt_template_str=(
            "Answer the following multiple choice question.\n\n"
            "Question: {question}\n\n"
            "Options:\n{options}\n\n"
            "Answer (letter):"
        ),
        description="MMLU-Pro professional-level multiple choice questions",
        dataset_kwargs={"source": "huggingface", "split": "test"},
    )


def _create_supergpqa_preset() -> BenchmarkPreset:
    """Create SuperGPQA benchmark preset."""
    return _create_mcq_preset(
        name="supergpqa",
        dataset_name="super_gpqa",
        prompt_template_str=(
            "Answer the following science question.\n\n"
            "Question: {question}\n\n"
            "Choices:\n{options}\n\n"
            "Answer (letter):"
        ),
        description="SuperGPQA graduate-level science questions",
        dataset_kwargs={"source": "huggingface", "split": "test"},
    )


def _create_gpqa_preset() -> BenchmarkPreset:
    """Create GPQA benchmark preset."""
    return _create_mcq_preset(
        name="gpqa",
        dataset_name="gpqa",
        prompt_template_str=(
            "Answer the following question.\n\n"
            "Question: {question}\n\n"
            "Choices:\n{options}\n\n"
            "Answer (letter):"
        ),
        description="GPQA graduate-level science questions",
        dataset_kwargs={"source": "huggingface", "split": "test", "subset": "default"},
    )


def _create_medmcqa_preset() -> BenchmarkPreset:
    """Create MedMCQA benchmark preset."""
    return _create_mcq_preset(
        name="medmcqa",
        dataset_name="medmcqa",
        prompt_template_str=(
            "Answer the following medical multiple choice question.\n\n"
            "Question: {question}\n\n"
            "Options:\n{options}\n\n"
            "Answer (letter):"
        ),
        description="MedMCQA medical entrance exam questions",
        dataset_kwargs={"source": "huggingface", "split": "test"},
    )


def _create_med_qa_preset() -> BenchmarkPreset:
    """Create MedQA benchmark preset."""
    return _create_mcq_preset(
        name="med_qa",
        dataset_name="med_qa",
        prompt_template_str=(
            "Answer the following medical multiple choice question.\n\n"
            "Question: {question}\n\n"
            "Options:\n{options}\n\n"
            "Answer (letter):"
        ),
        description="MedQA multiple choice medical QA benchmark",
        dataset_kwargs={"source": "huggingface", "split": "test"},
    )


def _create_sciq_preset() -> BenchmarkPreset:
    """Create SciQ benchmark preset."""
    return _create_mcq_preset(
        name="sciq",
        dataset_name="sciq",
        prompt_template_str=(
            "Answer the following science question.\n\n"
            "Question: {question}\n\n"
            "Options:\n{options}\n\n"
            "Answer (letter):"
        ),
        description="SciQ science multiple choice questions",
        dataset_kwargs={"source": "huggingface", "split": "test"},
        metadata_fields=(),
    )


def _create_commonsense_qa_preset() -> BenchmarkPreset:
    """Create CommonsenseQA benchmark preset."""
    return _create_mcq_preset(
        name="commonsense_qa",
        dataset_name="commonsense_qa",
        prompt_template_str=(
            "Answer the following commonsense question.\n\n"
            "Question: {question}\n\n"
            "Options:\n{options}\n\n"
            "Answer (letter):"
        ),
        description="CommonsenseQA multiple choice reasoning benchmark",
        dataset_kwargs={"source": "huggingface", "split": "validation"},
        metadata_fields=("concept",),
    )


def _create_piqa_preset() -> BenchmarkPreset:
    """Create PIQA benchmark preset."""
    return _create_mcq_preset(
        name="piqa",
        dataset_name="piqa",
        prompt_template_str=(
            "Choose the best answer for the goal.\n\n"
            "Goal: {goal}\n\n"
            "Options:\n{options}\n\n"
            "Answer (letter):"
        ),
        description="PIQA physical commonsense reasoning benchmark",
        dataset_kwargs={"source": "huggingface", "split": "validation"},
        metadata_fields=(),
    )


def _create_social_i_qa_preset() -> BenchmarkPreset:
    """Create Social IQA benchmark preset."""
    return _create_mcq_preset(
        name="social_i_qa",
        dataset_name="social_i_qa",
        prompt_template_str=(
            "Answer the question based on the social context.\n\n"
            "Context: {context}\n"
            "Question: {question}\n\n"
            "Options:\n{options}\n\n"
            "Answer (letter):"
        ),
        description="Social IQA commonsense reasoning benchmark",
        dataset_kwargs={"source": "huggingface", "split": "validation"},
        metadata_fields=(),
    )


def _create_coqa_preset() -> BenchmarkPreset:
    """Create CoQA benchmark preset."""
    return _create_mcq_preset(
        name="coqa",
        dataset_name="coqa",
        prompt_template_str=(
            "Answer the question based on the passage.\n\n"
            "Passage: {story}\n\n"
            "Question: {question}\n"
            "Answer:"
        ),
        description="CoQA conversational question answering benchmark",
        dataset_kwargs={"source": "huggingface", "split": "validation"},
        metadata_fields=("turn",),
    )


# ============================================================================
# Demo/Test Benchmarks
# ============================================================================


def _create_demo_preset() -> BenchmarkPreset:
    """Create demo benchmark preset for testing."""
    from themis.evaluation.extractors.identity_extractor import IdentityExtractor
    from themis.evaluation.metrics.exact_match import ExactMatch

    def load_demo(limit: int | None = None) -> Sequence[dict[str, Any]]:
        samples = [
            {"id": "demo-1", "question": "What is 2 + 2?", "answer": "4"},
            {
                "id": "demo-2",
                "question": "What is the capital of France?",
                "answer": "Paris",
            },
            {"id": "demo-3", "question": "What is 10 * 5?", "answer": "50"},
        ]
        if limit is not None:
            samples = samples[:limit]
        return samples

    prompt_template = PromptTemplate(
        name="demo",
        template="Q: {question}\nA:",
    )

    return BenchmarkPreset(
        name="demo",
        prompt_template=prompt_template,
        metrics=[ExactMatch()],
        extractor=IdentityExtractor(),
        dataset_loader=load_demo,
        metadata_fields=(),
        reference_field="answer",
        dataset_id_field="id",
        description="Demo benchmark for testing",
    )


# ============================================================================
# Register all benchmarks (lazy initialization)
# ============================================================================


def _register_all_benchmarks() -> None:
    """Register all built-in benchmarks.

    This is called lazily on first use to avoid importing heavy dependencies
    (datasets, models, etc.) until actually needed.
    """
    # Math benchmarks
    register_benchmark(_create_math500_preset())
    register_benchmark(_create_gsm8k_preset())
    register_benchmark(_create_aime24_preset())
    register_benchmark(_create_aime25_preset())
    register_benchmark(_create_amc23_preset())
    register_benchmark(_create_olympiadbench_preset())
    register_benchmark(_create_beyondaime_preset())
    register_benchmark(_create_gsm_symbolic_preset())

    # MCQ benchmarks
    register_benchmark(_create_mmlu_pro_preset())
    register_benchmark(_create_supergpqa_preset())
    register_benchmark(_create_gpqa_preset())
    register_benchmark(_create_medmcqa_preset())
    register_benchmark(_create_med_qa_preset())
    register_benchmark(_create_sciq_preset())
    register_benchmark(_create_commonsense_qa_preset())
    register_benchmark(_create_piqa_preset())
    register_benchmark(_create_social_i_qa_preset())
    register_benchmark(_create_coqa_preset())

    # Demo
    register_benchmark(_create_demo_preset())


__all__ = [
    "BenchmarkPreset",
    "register_benchmark",
    "get_benchmark_preset",
    "list_benchmarks",
]
