"""Math benchmark presets and factories."""

from __future__ import annotations

from typing import Any, Sequence

from themis.generation.templates import PromptTemplate
from themis.presets.core import BenchmarkPreset, register_benchmark
from themis.presets.mcq_benchmarks import _to_dict_samples


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


def _create_math500_preset() -> BenchmarkPreset:
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
    return _create_math_preset(
        name="gsm8k",
        dataset_name="gsm8k",
        prompt_template_str="Solve this math problem step by step.\n\nQ: {question}\nA:",
        description="GSM8K dataset with grade school math word problems",
        dataset_kwargs={"source": "huggingface", "split": "test"},
        metadata_fields=(),
    )


def _create_aime24_preset() -> BenchmarkPreset:
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
    return _create_math_preset(
        name="gsm-symbolic",
        dataset_name="gsm-symbolic",
        prompt_template_str="Solve this math problem step by step.\n\nQ: {question}\nA:",
        description="GSM-Symbolic dataset for algebraic word problems",
        dataset_kwargs={"source": "huggingface", "split": "test"},
        metadata_fields=(),
    )


def _create_aime25_preset() -> BenchmarkPreset:
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


def _register_math_benchmarks() -> None:
    """Register all built-in math benchmarks."""
    register_benchmark(_create_math500_preset())
    register_benchmark(_create_gsm8k_preset())
    register_benchmark(_create_aime24_preset())
    register_benchmark(_create_aime25_preset())
    register_benchmark(_create_amc23_preset())
    register_benchmark(_create_olympiadbench_preset())
    register_benchmark(_create_beyondaime_preset())
    register_benchmark(_create_gsm_symbolic_preset())
