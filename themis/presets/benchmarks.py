"""Benchmark preset configurations.

This module provides pre-configured settings for popular benchmarks,
including prompts, metrics, extractors, and data loaders.
"""

from __future__ import annotations

import string
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from themis.generation.templates import PromptTemplate
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
        ValueError: If benchmark is not found
    """
    _ensure_registry_initialized()

    name_lower = name.lower()
    if name_lower not in _BENCHMARK_REGISTRY:
        available = ", ".join(sorted(_BENCHMARK_REGISTRY.keys()))
        raise ValueError(
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
# Math Benchmarks
# ============================================================================


def _create_math500_preset() -> BenchmarkPreset:
    """Create MATH-500 benchmark preset."""
    from themis.datasets.math500 import load_math500 as load_math500_dataset
    from themis.evaluation.extractors.math_verify_extractor import MathVerifyExtractor
    from themis.evaluation.metrics.math_verify_accuracy import MathVerifyAccuracy

    def load_math500(limit: int | None = None) -> Sequence[dict[str, Any]]:
        samples = load_math500_dataset(source="huggingface", limit=limit)
        return _to_dict_samples(samples)

    prompt_template = PromptTemplate(
        name="math500-zero-shot",
        template=(
            "Solve the following math problem step by step. "
            "Put your final answer in \\boxed{{}}.\n\n"
            "Problem: {problem}\n\n"
            "Solution:"
        ),
    )

    return BenchmarkPreset(
        name="math500",
        prompt_template=prompt_template,
        metrics=[MathVerifyAccuracy()],
        extractor=MathVerifyExtractor(),
        dataset_loader=load_math500,
        metadata_fields=("subject", "level"),
        reference_field="solution",
        dataset_id_field="unique_id",
        description="MATH-500 dataset with 500 competition math problems",
    )


def _create_gsm8k_preset() -> BenchmarkPreset:
    """Create GSM8K benchmark preset."""
    from themis.datasets.gsm8k import load_gsm8k as load_gsm8k_dataset
    from themis.evaluation.extractors.math_verify_extractor import MathVerifyExtractor
    from themis.evaluation.metrics.math_verify_accuracy import MathVerifyAccuracy

    def load_gsm8k(limit: int | None = None) -> Sequence[dict[str, Any]]:
        samples = load_gsm8k_dataset(source="huggingface", split="test", limit=limit)
        return _to_dict_samples(samples)

    prompt_template = PromptTemplate(
        name="gsm8k-zero-shot",
        template=("Solve this math problem step by step.\n\nQ: {question}\nA:"),
    )

    return BenchmarkPreset(
        name="gsm8k",
        prompt_template=prompt_template,
        metrics=[MathVerifyAccuracy()],
        extractor=MathVerifyExtractor(),
        dataset_loader=load_gsm8k,
        metadata_fields=(),
        reference_field="answer",
        dataset_id_field="unique_id",
        description="GSM8K dataset with grade school math word problems",
    )


def _create_aime24_preset() -> BenchmarkPreset:
    """Create AIME 2024 benchmark preset."""
    from themis.datasets.competition_math import load_competition_math
    from themis.evaluation.extractors.math_verify_extractor import MathVerifyExtractor
    from themis.evaluation.metrics.math_verify_accuracy import MathVerifyAccuracy

    def load_aime24(limit: int | None = None) -> Sequence[dict[str, Any]]:
        samples = load_competition_math(
            dataset="math-ai/aime24",
            source="huggingface",
            split="test",
            limit=limit,
        )
        return _to_dict_samples(samples)

    prompt_template = PromptTemplate(
        name="aime24-zero-shot",
        template=(
            "Solve the following AIME problem. "
            "Your answer should be a number between 000 and 999.\n\n"
            "Problem: {problem}\n\n"
            "Solution:"
        ),
    )

    return BenchmarkPreset(
        name="aime24",
        prompt_template=prompt_template,
        metrics=[MathVerifyAccuracy()],
        extractor=MathVerifyExtractor(),
        dataset_loader=load_aime24,
        metadata_fields=("subject",),
        reference_field="answer",
        dataset_id_field="unique_id",
        description="AIME 2024 competition math problems",
    )


def _create_gsm_symbolic_preset() -> BenchmarkPreset:
    """Create GSM-Symbolic benchmark preset."""
    from themis.datasets.gsm_symbolic import (
        load_gsm_symbolic as load_gsm_symbolic_dataset,
    )
    from themis.evaluation.extractors.math_verify_extractor import MathVerifyExtractor
    from themis.evaluation.metrics.math_verify_accuracy import MathVerifyAccuracy

    def load_gsm_symbolic(limit: int | None = None) -> Sequence[dict[str, Any]]:
        samples = load_gsm_symbolic_dataset(
            source="huggingface",
            split="test",
            limit=limit,
        )
        return _to_dict_samples(samples)

    prompt_template = PromptTemplate(
        name="gsm-symbolic-zero-shot",
        template=("Solve this math problem step by step.\n\nQ: {question}\nA:"),
    )

    return BenchmarkPreset(
        name="gsm-symbolic",
        prompt_template=prompt_template,
        metrics=[MathVerifyAccuracy()],
        extractor=MathVerifyExtractor(),
        dataset_loader=load_gsm_symbolic,
        metadata_fields=(),
        reference_field="answer",
        dataset_id_field="unique_id",
        description="GSM-Symbolic dataset for algebraic word problems",
    )


def _create_aime25_preset() -> BenchmarkPreset:
    """Create AIME 2025 benchmark preset."""
    from themis.datasets.competition_math import load_competition_math
    from themis.evaluation.extractors.math_verify_extractor import MathVerifyExtractor
    from themis.evaluation.metrics.math_verify_accuracy import MathVerifyAccuracy

    def load_aime25(limit: int | None = None) -> Sequence[dict[str, Any]]:
        samples = load_competition_math(
            dataset="math-ai/aime25",
            source="huggingface",
            split="test",
            limit=limit,
        )
        return _to_dict_samples(samples)

    prompt_template = PromptTemplate(
        name="aime25-zero-shot",
        template=(
            "Solve the following AIME problem. "
            "Your answer should be a number between 000 and 999.\n\n"
            "Problem: {problem}\n\n"
            "Solution:"
        ),
    )

    return BenchmarkPreset(
        name="aime25",
        prompt_template=prompt_template,
        metrics=[MathVerifyAccuracy()],
        extractor=MathVerifyExtractor(),
        dataset_loader=load_aime25,
        metadata_fields=("subject", "level"),
        reference_field="answer",
        dataset_id_field="unique_id",
        description="AIME 2025 competition math problems",
    )


def _create_amc23_preset() -> BenchmarkPreset:
    """Create AMC 2023 benchmark preset."""
    from themis.datasets.competition_math import load_competition_math
    from themis.evaluation.extractors.math_verify_extractor import MathVerifyExtractor
    from themis.evaluation.metrics.math_verify_accuracy import MathVerifyAccuracy

    def load_amc23(limit: int | None = None) -> Sequence[dict[str, Any]]:
        samples = load_competition_math(
            dataset="math-ai/amc23",
            source="huggingface",
            split="test",
            limit=limit,
        )
        return _to_dict_samples(samples)

    prompt_template = PromptTemplate(
        name="amc23-zero-shot",
        template=(
            "Solve the following AMC problem. "
            "Give only the final answer.\n\n"
            "Problem: {problem}\n\n"
            "Answer:"
        ),
    )

    return BenchmarkPreset(
        name="amc23",
        prompt_template=prompt_template,
        metrics=[MathVerifyAccuracy()],
        extractor=MathVerifyExtractor(),
        dataset_loader=load_amc23,
        metadata_fields=("subject", "level"),
        reference_field="answer",
        dataset_id_field="unique_id",
        description="AMC 2023 competition math problems",
    )


def _create_olympiadbench_preset() -> BenchmarkPreset:
    """Create OlympiadBench benchmark preset."""
    from themis.datasets.competition_math import load_competition_math
    from themis.evaluation.extractors.math_verify_extractor import MathVerifyExtractor
    from themis.evaluation.metrics.math_verify_accuracy import MathVerifyAccuracy

    def load_olympiadbench(limit: int | None = None) -> Sequence[dict[str, Any]]:
        samples = load_competition_math(
            dataset="math-ai/olympiadbench",
            source="huggingface",
            split="test",
            limit=limit,
        )
        return _to_dict_samples(samples)

    prompt_template = PromptTemplate(
        name="olympiadbench-zero-shot",
        template=(
            "Solve the following olympiad-style math problem. "
            "Show reasoning briefly, then give the final answer.\n\n"
            "Problem: {problem}\n\n"
            "Solution:"
        ),
    )

    return BenchmarkPreset(
        name="olympiadbench",
        prompt_template=prompt_template,
        metrics=[MathVerifyAccuracy()],
        extractor=MathVerifyExtractor(),
        dataset_loader=load_olympiadbench,
        metadata_fields=("subject", "level"),
        reference_field="answer",
        dataset_id_field="unique_id",
        description="OlympiadBench competition math benchmark",
    )


def _create_beyondaime_preset() -> BenchmarkPreset:
    """Create BeyondAIME benchmark preset."""
    from themis.datasets.competition_math import load_competition_math
    from themis.evaluation.extractors.math_verify_extractor import MathVerifyExtractor
    from themis.evaluation.metrics.math_verify_accuracy import MathVerifyAccuracy

    def load_beyondaime(limit: int | None = None) -> Sequence[dict[str, Any]]:
        samples = load_competition_math(
            dataset="ByteDance-Seed/BeyondAIME",
            source="huggingface",
            split="test",
            limit=limit,
        )
        return _to_dict_samples(samples)

    prompt_template = PromptTemplate(
        name="beyondaime-zero-shot",
        template=(
            "Solve the following advanced contest math problem. "
            "Provide the final answer clearly.\n\n"
            "Problem: {problem}\n\n"
            "Answer:"
        ),
    )

    return BenchmarkPreset(
        name="beyondaime",
        prompt_template=prompt_template,
        metrics=[MathVerifyAccuracy()],
        extractor=MathVerifyExtractor(),
        dataset_loader=load_beyondaime,
        metadata_fields=("subject", "level"),
        reference_field="answer",
        dataset_id_field="unique_id",
        description="BeyondAIME advanced competition math problems",
    )


# ============================================================================
# MCQ Benchmarks
# ============================================================================


def _create_mmlu_pro_preset() -> BenchmarkPreset:
    """Create MMLU-Pro benchmark preset."""
    from themis.datasets.mmlu_pro import load_mmlu_pro as load_mmlu_pro_dataset
    from themis.evaluation.extractors.identity_extractor import IdentityExtractor
    from themis.evaluation.metrics.exact_match import ExactMatch

    def load_mmlu_pro(limit: int | None = None) -> Sequence[dict[str, Any]]:
        samples = load_mmlu_pro_dataset(source="huggingface", split="test", limit=limit)
        return _normalize_mcq_samples(_to_dict_samples(samples))

    prompt_template = PromptTemplate(
        name="mmlu-pro-zero-shot",
        template=(
            "Answer the following multiple choice question.\n\n"
            "Question: {question}\n\n"
            "Options:\n{options}\n\n"
            "Answer (letter):"
        ),
    )

    return BenchmarkPreset(
        name="mmlu-pro",
        prompt_template=prompt_template,
        metrics=[ExactMatch()],
        extractor=IdentityExtractor(),
        dataset_loader=load_mmlu_pro,
        metadata_fields=("subject",),
        reference_field="answer",
        dataset_id_field="unique_id",
        description="MMLU-Pro professional-level multiple choice questions",
    )


def _create_supergpqa_preset() -> BenchmarkPreset:
    """Create SuperGPQA benchmark preset."""
    from themis.datasets.super_gpqa import load_super_gpqa as load_supergpqa_dataset
    from themis.evaluation.extractors.identity_extractor import IdentityExtractor
    from themis.evaluation.metrics.exact_match import ExactMatch

    def load_supergpqa(limit: int | None = None) -> Sequence[dict[str, Any]]:
        samples = load_supergpqa_dataset(
            source="huggingface",
            split="test",
            limit=limit,
        )
        return _normalize_mcq_samples(_to_dict_samples(samples))

    prompt_template = PromptTemplate(
        name="supergpqa-zero-shot",
        template=(
            "Answer the following science question.\n\n"
            "Question: {question}\n\n"
            "Choices:\n{options}\n\n"
            "Answer (letter):"
        ),
    )

    return BenchmarkPreset(
        name="supergpqa",
        prompt_template=prompt_template,
        metrics=[ExactMatch()],
        extractor=IdentityExtractor(),
        dataset_loader=load_supergpqa,
        metadata_fields=("subject",),
        reference_field="answer",
        dataset_id_field="unique_id",
        description="SuperGPQA graduate-level science questions",
    )


def _create_gpqa_preset() -> BenchmarkPreset:
    """Create GPQA benchmark preset."""
    from themis.datasets.gpqa import load_gpqa as load_gpqa_dataset
    from themis.evaluation.extractors.identity_extractor import IdentityExtractor
    from themis.evaluation.metrics.exact_match import ExactMatch

    def load_gpqa(limit: int | None = None) -> Sequence[dict[str, Any]]:
        samples = load_gpqa_dataset(
            source="huggingface",
            split="test",
            limit=limit,
            subset="default",
        )
        return _normalize_mcq_samples(_to_dict_samples(samples))

    prompt_template = PromptTemplate(
        name="gpqa-zero-shot",
        template=(
            "Answer the following question.\n\n"
            "Question: {question}\n\n"
            "Choices:\n{options}\n\n"
            "Answer (letter):"
        ),
    )

    return BenchmarkPreset(
        name="gpqa",
        prompt_template=prompt_template,
        metrics=[ExactMatch()],
        extractor=IdentityExtractor(),
        dataset_loader=load_gpqa,
        metadata_fields=("subject",),
        reference_field="answer",
        dataset_id_field="unique_id",
        description="GPQA graduate-level science questions",
    )


def _create_medmcqa_preset() -> BenchmarkPreset:
    """Create MedMCQA benchmark preset."""
    from themis.datasets.medmcqa import load_medmcqa as load_medmcqa_dataset
    from themis.evaluation.extractors.identity_extractor import IdentityExtractor
    from themis.evaluation.metrics.exact_match import ExactMatch

    def load_medmcqa(limit: int | None = None) -> Sequence[dict[str, Any]]:
        samples = load_medmcqa_dataset(
            source="huggingface",
            split="test",
            limit=limit,
        )
        return _normalize_mcq_samples(_to_dict_samples(samples))

    prompt_template = PromptTemplate(
        name="medmcqa-zero-shot",
        template=(
            "Answer the following medical multiple choice question.\n\n"
            "Question: {question}\n\n"
            "Options:\n{options}\n\n"
            "Answer (letter):"
        ),
    )

    return BenchmarkPreset(
        name="medmcqa",
        prompt_template=prompt_template,
        metrics=[ExactMatch()],
        extractor=IdentityExtractor(),
        dataset_loader=load_medmcqa,
        metadata_fields=("subject",),
        reference_field="answer",
        dataset_id_field="unique_id",
        description="MedMCQA medical entrance exam questions",
    )


def _create_med_qa_preset() -> BenchmarkPreset:
    """Create MedQA benchmark preset."""
    from themis.datasets.med_qa import load_med_qa as load_med_qa_dataset
    from themis.evaluation.extractors.identity_extractor import IdentityExtractor
    from themis.evaluation.metrics.exact_match import ExactMatch

    def load_med_qa(limit: int | None = None) -> Sequence[dict[str, Any]]:
        samples = load_med_qa_dataset(
            source="huggingface",
            split="test",
            limit=limit,
        )
        return _normalize_mcq_samples(_to_dict_samples(samples))

    prompt_template = PromptTemplate(
        name="med-qa-zero-shot",
        template=(
            "Answer the following medical multiple choice question.\n\n"
            "Question: {question}\n\n"
            "Options:\n{options}\n\n"
            "Answer (letter):"
        ),
    )

    return BenchmarkPreset(
        name="med_qa",
        prompt_template=prompt_template,
        metrics=[ExactMatch()],
        extractor=IdentityExtractor(),
        dataset_loader=load_med_qa,
        metadata_fields=("subject",),
        reference_field="answer",
        dataset_id_field="unique_id",
        description="MedQA multiple choice medical QA benchmark",
    )


def _create_sciq_preset() -> BenchmarkPreset:
    """Create SciQ benchmark preset."""
    from themis.datasets.sciq import load_sciq as load_sciq_dataset
    from themis.evaluation.extractors.identity_extractor import IdentityExtractor
    from themis.evaluation.metrics.exact_match import ExactMatch

    def load_sciq(limit: int | None = None) -> Sequence[dict[str, Any]]:
        samples = load_sciq_dataset(
            source="huggingface",
            split="test",
            limit=limit,
        )
        return _normalize_mcq_samples(_to_dict_samples(samples))

    prompt_template = PromptTemplate(
        name="sciq-zero-shot",
        template=(
            "Answer the following science question.\n\n"
            "Question: {question}\n\n"
            "Options:\n{options}\n\n"
            "Answer (letter):"
        ),
    )

    return BenchmarkPreset(
        name="sciq",
        prompt_template=prompt_template,
        metrics=[ExactMatch()],
        extractor=IdentityExtractor(),
        dataset_loader=load_sciq,
        metadata_fields=(),
        reference_field="answer",
        dataset_id_field="unique_id",
        description="SciQ science multiple choice questions",
    )


def _create_commonsense_qa_preset() -> BenchmarkPreset:
    """Create CommonsenseQA benchmark preset."""
    from themis.datasets.commonsense_qa import (
        load_commonsense_qa as load_commonsense_qa_dataset,
    )
    from themis.evaluation.extractors.identity_extractor import IdentityExtractor
    from themis.evaluation.metrics.exact_match import ExactMatch

    def load_commonsense_qa(limit: int | None = None) -> Sequence[dict[str, Any]]:
        samples = load_commonsense_qa_dataset(
            source="huggingface",
            split="validation",
            limit=limit,
        )
        return _normalize_mcq_samples(_to_dict_samples(samples))

    prompt_template = PromptTemplate(
        name="commonsense-qa-zero-shot",
        template=(
            "Answer the following commonsense question.\n\n"
            "Question: {question}\n\n"
            "Options:\n{options}\n\n"
            "Answer (letter):"
        ),
    )

    return BenchmarkPreset(
        name="commonsense_qa",
        prompt_template=prompt_template,
        metrics=[ExactMatch()],
        extractor=IdentityExtractor(),
        dataset_loader=load_commonsense_qa,
        metadata_fields=("concept",),
        reference_field="answer",
        dataset_id_field="unique_id",
        description="CommonsenseQA multiple choice reasoning benchmark",
    )


def _create_piqa_preset() -> BenchmarkPreset:
    """Create PIQA benchmark preset."""
    from themis.datasets.piqa import load_piqa as load_piqa_dataset
    from themis.evaluation.extractors.identity_extractor import IdentityExtractor
    from themis.evaluation.metrics.exact_match import ExactMatch

    def load_piqa(limit: int | None = None) -> Sequence[dict[str, Any]]:
        samples = load_piqa_dataset(
            source="huggingface",
            split="validation",
            limit=limit,
        )
        return _normalize_mcq_samples(_to_dict_samples(samples))

    prompt_template = PromptTemplate(
        name="piqa-zero-shot",
        template=(
            "Choose the best answer for the goal.\n\n"
            "Goal: {goal}\n\n"
            "Options:\n{options}\n\n"
            "Answer (letter):"
        ),
    )

    return BenchmarkPreset(
        name="piqa",
        prompt_template=prompt_template,
        metrics=[ExactMatch()],
        extractor=IdentityExtractor(),
        dataset_loader=load_piqa,
        metadata_fields=(),
        reference_field="answer",
        dataset_id_field="unique_id",
        description="PIQA physical commonsense reasoning benchmark",
    )


def _create_social_i_qa_preset() -> BenchmarkPreset:
    """Create Social IQA benchmark preset."""
    from themis.datasets.social_i_qa import load_social_i_qa as load_social_i_qa_dataset
    from themis.evaluation.extractors.identity_extractor import IdentityExtractor
    from themis.evaluation.metrics.exact_match import ExactMatch

    def load_social_i_qa(limit: int | None = None) -> Sequence[dict[str, Any]]:
        samples = load_social_i_qa_dataset(
            source="huggingface",
            split="validation",
            limit=limit,
        )
        return _normalize_mcq_samples(_to_dict_samples(samples))

    prompt_template = PromptTemplate(
        name="social-iqa-zero-shot",
        template=(
            "Answer the question based on the social context.\n\n"
            "Context: {context}\n"
            "Question: {question}\n\n"
            "Options:\n{options}\n\n"
            "Answer (letter):"
        ),
    )

    return BenchmarkPreset(
        name="social_i_qa",
        prompt_template=prompt_template,
        metrics=[ExactMatch()],
        extractor=IdentityExtractor(),
        dataset_loader=load_social_i_qa,
        metadata_fields=(),
        reference_field="answer",
        dataset_id_field="unique_id",
        description="Social IQA commonsense reasoning benchmark",
    )


def _create_coqa_preset() -> BenchmarkPreset:
    """Create CoQA benchmark preset."""
    from themis.datasets.coqa import load_coqa as load_coqa_dataset
    from themis.evaluation.extractors.identity_extractor import IdentityExtractor
    from themis.evaluation.metrics.exact_match import ExactMatch

    def load_coqa(limit: int | None = None) -> Sequence[dict[str, Any]]:
        samples = load_coqa_dataset(
            source="huggingface",
            split="validation",
            limit=limit,
        )
        return _to_dict_samples(samples)

    prompt_template = PromptTemplate(
        name="coqa-zero-shot",
        template=(
            "Answer the question based on the passage.\n\n"
            "Passage: {story}\n\n"
            "Question: {question}\n"
            "Answer:"
        ),
    )

    return BenchmarkPreset(
        name="coqa",
        prompt_template=prompt_template,
        metrics=[ExactMatch()],
        extractor=IdentityExtractor(),
        dataset_loader=load_coqa,
        metadata_fields=("turn",),
        reference_field="answer",
        dataset_id_field="unique_id",
        description="CoQA conversational question answering benchmark",
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
