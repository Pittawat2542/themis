"""Benchmark listing commands."""

from __future__ import annotations

from typing import Annotated

from cyclopts import Parameter

from themis.providers.registry import _REGISTRY


def list_providers(
    *,
    verbose: Annotated[
        bool, Parameter(help="Show detailed provider information")
    ] = False,
) -> int:
    """List available LLM providers."""
    providers = sorted(_REGISTRY._factories.keys())

    if not providers:
        print("No providers registered.")
        return 0

    print("Available Providers:")
    print("=" * 60)

    provider_info = {
        "fake": "Built-in fake provider for testing (no API required)",
        "openai-compatible": "OpenAI-compatible API (LM Studio, Ollama, vLLM, OpenAI)",
        "vllm": "vLLM server provider for local model hosting",
    }

    for provider in providers:
        status = "✓" if provider in provider_info else "·"
        print(f"{status} {provider}")
        if verbose and provider in provider_info:
            print(f"  {provider_info[provider]}")

    if not verbose:
        print("\nUse --verbose for more details")

    return 0


def list_benchmarks(
    *,
    verbose: Annotated[
        bool, Parameter(help="Show detailed benchmark information")
    ] = False,
) -> int:
    """List available datasets and benchmarks."""
    benchmarks = [
        {
            "name": "math500",
            "description": "MATH-500 dataset for mathematical reasoning",
            "source": "huggingface (default) or local",
            "subjects": [
                "algebra",
                "counting_and_probability",
                "geometry",
                "intermediate_algebra",
                "number_theory",
                "prealgebra",
                "precalculus",
            ],
            "command": "uv run python -m themis.cli math500",
        },
        {
            "name": "gsm8k",
            "description": "GSM8K dataset for grade school math word problems",
            "source": "huggingface (default) or local",
            "subjects": "math",
            "command": "uv run python -m themis.cli gsm8k",
        },
        {
            "name": "gpqa",
            "description": "GPQA dataset for graduate-level science questions",
            "source": "huggingface (default) or local",
            "subjects": "science",
            "command": "uv run python -m themis.cli gpqa",
        },
        {
            "name": "gsm-symbolic",
            "description": "GSM-Symbolic dataset for symbolic math reasoning",
            "source": "huggingface (default) or local",
            "subjects": "math",
            "command": "uv run python -m themis.cli gsm-symbolic",
        },
        {
            "name": "medmcqa",
            "description": "MedMCQA dataset for medical entrance exams",
            "source": "huggingface (default) or local",
            "subjects": "medicine",
            "command": "uv run python -m themis.cli medmcqa",
        },
        {
            "name": "med_qa",
            "description": "MedQA dataset for medical question answering",
            "source": "huggingface (default) or local",
            "subjects": "medicine",
            "command": "uv run python -m themis.cli med_qa",
        },
        {
            "name": "sciq",
            "description": "SciQ dataset for science questions",
            "source": "huggingface (default) or local",
            "subjects": "science",
            "command": "uv run python -m themis.cli sciq",
        },
        {
            "name": "commonsense_qa",
            "description": "CommonsenseQA dataset for commonsense reasoning",
            "source": "huggingface (default) or local",
            "subjects": "commonsense",
            "command": "uv run python -m themis.cli commonsense_qa",
        },
        {
            "name": "piqa",
            "description": "PIQA dataset for physical commonsense reasoning",
            "source": "huggingface (default) or local",
            "subjects": "commonsense",
            "command": "uv run python -m themis.cli piqa",
        },
        {
            "name": "social_i_qa",
            "description": "Social IQA dataset for social commonsense reasoning",
            "source": "huggingface (default) or local",
            "subjects": "commonsense",
            "command": "uv run python -m themis.cli social_i_qa",
        },
        {
            "name": "coqa",
            "description": "CoQA dataset for conversational question answering",
            "source": "huggingface (default) or local",
            "subjects": "conversational",
            "command": "uv run python -m themis.cli coqa",
        },
        {
            "name": "supergpqa",
            "description": "Graduate-level QA benchmark with multiple-choice questions",
            "source": "huggingface (default) or local",
            "subjects": "category filter via --subjects",
            "command": "uv run python -m themis.cli supergpqa",
        },
        {
            "name": "mmlu-pro",
            "description": "Professional-level MMLU benchmark with refined distractors",
            "source": "huggingface (default) or local",
            "subjects": "subject filter via --subjects",
            "command": "uv run python -m themis.cli mmlu-pro",
        },
        {
            "name": "aime24",
            "description": "AIME 2024 competition problems",
            "source": "huggingface (default) or local",
            "subjects": "problem set",
            "command": "uv run python -m themis.cli aime24",
        },
        {
            "name": "aime25",
            "description": "AIME 2025 competition problems",
            "source": "huggingface (default) or local",
            "subjects": "problem set",
            "command": "uv run python -m themis.cli aime25",
        },
        {
            "name": "amc23",
            "description": "AMC 2023 competition problems",
            "source": "huggingface (default) or local",
            "subjects": "problem set",
            "command": "uv run python -m themis.cli amc23",
        },
        {
            "name": "olympiadbench",
            "description": "Mixed Olympiad-style math benchmark",
            "source": "huggingface (default) or local",
            "subjects": "competition metadata",
            "command": "uv run python -m themis.cli olympiadbench",
        },
        {
            "name": "beyondaime",
            "description": "BeyondAIME advanced math competition set",
            "source": "huggingface (default) or local",
            "subjects": "problem set",
            "command": "uv run python -m themis.cli beyondaime",
        },
        {
            "name": "demo",
            "description": "Built-in demo with 2 math problems",
            "source": "inline",
            "subjects": ["precalculus", "arithmetic"],
            "command": "uv run python -m themis.cli demo",
        },
        {
            "name": "inline",
            "description": "Custom inline dataset (via config file)",
            "source": "config file",
            "subjects": "user-defined",
            "command": "uv run python -m themis.cli run-config --config your_config.yaml",
        },
    ]

    print("Available Datasets & Benchmarks:")
    print("=" * 60)

    for bench in benchmarks:
        print(f"\n📊 {bench['name']}")
        print(f"   {bench['description']}")
        if verbose:
            print(f"   Source: {bench['source']}")
            if isinstance(bench["subjects"], list):
                print(f"   Subjects: {', '.join(bench['subjects'])}")
            else:
                print(f"   Subjects: {bench['subjects']}")
            print(f"   Command: {bench['command']}")

    if not verbose:
        print("\nUse --verbose for more details and example commands")

    return 0
