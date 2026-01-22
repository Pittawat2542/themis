"""Benchmark preset configurations.

This module provides pre-configured settings for popular benchmarks,
including prompts, metrics, extractors, and data loaders.
"""

from __future__ import annotations

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
            f"Unknown benchmark: {name}. "
            f"Available benchmarks: {available}"
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
        # Convert MathSample objects to dicts
        return [s.to_generation_example() if hasattr(s, 'to_generation_example') else dict(s) for s in samples]
    
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
        # Convert sample objects to dicts if needed
        return [dict(s) if not isinstance(s, dict) else s for s in samples]
    
    prompt_template = PromptTemplate(
        name="gsm8k-zero-shot",
        template=(
            "Solve this math problem step by step.\n\n"
            "Q: {question}\n"
            "A:"
        ),
    )
    
    return BenchmarkPreset(
        name="gsm8k",
        prompt_template=prompt_template,
        metrics=[MathVerifyAccuracy()],
        extractor=MathVerifyExtractor(),
        dataset_loader=load_gsm8k,
        metadata_fields=(),
        reference_field="answer",
        dataset_id_field="id",
        description="GSM8K dataset with grade school math word problems",
    )


def _create_aime24_preset() -> BenchmarkPreset:
    """Create AIME 2024 benchmark preset."""
    from themis.datasets.competition_math import load_competition_math
    from themis.evaluation.extractors.math_verify_extractor import MathVerifyExtractor
    from themis.evaluation.metrics.math_verify_accuracy import MathVerifyAccuracy
    
    def load_aime24(limit: int | None = None) -> Sequence[dict[str, Any]]:
        samples = load_competition_math(
            dataset_id="aime24",
            source="huggingface",
            split="test",
            limit=limit,
        )
        return [dict(s) if not isinstance(s, dict) else s for s in samples]
    
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
        dataset_id_field="id",
        description="AIME 2024 competition math problems",
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
        return [dict(s) if not isinstance(s, dict) else s for s in samples]
    
    prompt_template = PromptTemplate(
        name="mmlu-pro-zero-shot",
        template=(
            "Answer the following multiple choice question.\n\n"
            "Question: {question}\n\n"
            "Options:\n{options}\n\n"
            "Answer:"
        ),
    )
    
    return BenchmarkPreset(
        name="mmlu-pro",
        prompt_template=prompt_template,
        metrics=[ExactMatch()],
        extractor=IdentityExtractor(),
        dataset_loader=load_mmlu_pro,
        metadata_fields=("category",),
        reference_field="answer",
        dataset_id_field="id",
        description="MMLU-Pro professional-level multiple choice questions",
    )


def _create_supergpqa_preset() -> BenchmarkPreset:
    """Create SuperGPQA benchmark preset."""
    from themis.datasets.super_gpqa import load_super_gpqa as load_supergpqa_dataset
    from themis.evaluation.extractors.identity_extractor import IdentityExtractor
    from themis.evaluation.metrics.exact_match import ExactMatch
    
    def load_supergpqa(limit: int | None = None) -> Sequence[dict[str, Any]]:
        samples = load_supergpqa_dataset(source="huggingface", split="test", limit=limit)
        return [dict(s) if not isinstance(s, dict) else s for s in samples]
    
    prompt_template = PromptTemplate(
        name="supergpqa-zero-shot",
        template=(
            "Answer the following science question.\n\n"
            "Question: {question}\n\n"
            "Choices:\n{choices}\n\n"
            "Answer:"
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
        dataset_id_field="id",
        description="SuperGPQA graduate-level science questions",
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
            {"id": "demo-2", "question": "What is the capital of France?", "answer": "Paris"},
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
    
    # MCQ benchmarks
    register_benchmark(_create_mmlu_pro_preset())
    register_benchmark(_create_supergpqa_preset())
    
    # Demo
    register_benchmark(_create_demo_preset())


__all__ = [
    "BenchmarkPreset",
    "register_benchmark",
    "get_benchmark_preset",
    "list_benchmarks",
]
