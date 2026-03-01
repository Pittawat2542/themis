"""Benchmark preset configurations (Facade module).

This module provides pre-configured settings for popular benchmarks,
including prompts, metrics, extractors, and data loaders.

Note: This module has been refactored into `core.py`, `math_benchmarks.py`,
and `mcq_benchmarks.py`. This file serves as a facade to maintain backward
compatibility.
"""

from __future__ import annotations

from typing import Any, Sequence

from themis.evaluation.extractors.identity_extractor import IdentityExtractor
from themis.evaluation.metrics.exact_match import ExactMatch
from themis.generation.templates import PromptTemplate
from themis.presets.core import (
    BenchmarkPreset,
    register_benchmark,
    get_benchmark_preset,
    list_benchmarks,
)


def _create_demo_preset() -> BenchmarkPreset:
    """Create demo benchmark preset for testing."""

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


def _register_demo_benchmark() -> None:
    register_benchmark(_create_demo_preset())


__all__ = [
    "BenchmarkPreset",
    "register_benchmark",
    "get_benchmark_preset",
    "list_benchmarks",
]
