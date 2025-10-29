"""Configuration models for the prompt engineering experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class PromptVariantConfig(BaseModel):
    """Configuration for a single prompt variation."""
    name: str
    template: str
    description: str
    metadata: dict[str, object] = Field(default_factory=dict)


class SamplingProfile(BaseModel):
    name: str = "default"
    temperature: float = 0.0
    top_p: float = 0.95
    max_tokens: int = 512

    def to_sampling_config(self):
        from themis.core import entities as core_entities

        return core_entities.SamplingConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )


class ModelConfig(BaseModel):
    name: str
    provider: str = "litellm"  # Using the universal LiteLLM provider
    provider_options: dict[str, object] = Field(default_factory=dict)
    description: str | None = None


class DatasetConfig(BaseModel):
    name: str
    kind: Literal["demo", "math500_local", "math500_hf", "inline"] = "demo"
    limit: int | None = None
    subjects: tuple[str, ...] = Field(default_factory=tuple)
    data_dir: Path | None = None
    # For inline dataset
    samples: list[dict[str, object]] = Field(default_factory=list)


class PromptEngineeringConfig(BaseModel):
    run_id: str = "prompt-eng-experiment"
    storage_dir: Path = Path(".cache/prompt-eng")
    resume: bool = True
    prompt_variants: list[PromptVariantConfig]
    models: list[ModelConfig]
    samplings: list[SamplingProfile] = Field(
        default_factory=lambda: [SamplingProfile()]
    )
    datasets: list[DatasetConfig]

    def apply_overrides(
        self,
        *,
        run_id: str | None = None,
        storage_dir: Path | None = None,
        resume: bool | None = None,
    ) -> "PromptEngineeringConfig":
        update: dict[str, object] = {}
        if run_id is not None:
            update["run_id"] = run_id
        if storage_dir is not None:
            update["storage_dir"] = storage_dir
        if resume is not None:
            update["resume"] = resume
        if not update:
            return self
        return self.model_copy(update=update)


# Default config with multiple prompt variations
DEFAULT_PROMPT_VARIANTS = [
    {
        "name": "zero-shot",
        "template": """
You are an expert mathematician. Solve the problem below and respond with a JSON object containing `answer` and `reasoning` keys only.

Problem:
{problem}
        """.strip(),
        "description": "Direct problem solving without examples",
        "metadata": {"strategy": "zero-shot"}
    },
    {
        "name": "few-shot",
        "template": """
You are an expert mathematician. Solve the problem below and respond with a JSON object containing `answer` and `reasoning` keys only.

Example:
Problem: What is 2+2?
Answer: {{"answer": "4", "reasoning": "Simple addition of 2 and 2"}}

Problem:
{problem}

Answer:
        """.strip(),
        "description": "Problem solving with one example",
        "metadata": {"strategy": "few-shot"}
    },
    {
        "name": "chain-of-thought",
        "template": """
You are an expert mathematician. Think through the problem step-by-step, then provide your final answer.

Problem: {problem}

Step-by-step reasoning:
        """.strip(),
        "description": "Chain-of-thought reasoning approach",
        "metadata": {"strategy": "chain-of-thought"}
    }
]

DEFAULT_CONFIG = PromptEngineeringConfig(
    prompt_variants=[PromptVariantConfig(**variant) for variant in DEFAULT_PROMPT_VARIANTS],
    models=[
        ModelConfig(
            name="openai/qwen/qwen3-vl-30b",
            provider="litellm",
            provider_options={
                "api_base": "http://localhost:1234/v1",
                "api_key": "sk-not-needed",
                "timeout": 300,
                "custom_llm_provider": "openai"
            },
            description="Local Qwen3-VL-30B model via OpenAI-compatible endpoint (requires local server running on port 1234)"
        )
    ],
    samplings=[SamplingProfile(name="default", temperature=0.7, top_p=0.95, max_tokens=-1)],
    datasets=[
        DatasetConfig(
            name="demo",
            kind="demo",
            limit=5
        )
    ]
)


def load_config(path: Path) -> PromptEngineeringConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    return PromptEngineeringConfig.model_validate(data)


__all__ = [
    "PromptVariantConfig",
    "SamplingProfile", 
    "ModelConfig",
    "DatasetConfig",
    "PromptEngineeringConfig",
    "DEFAULT_CONFIG",
    "DEFAULT_PROMPT_VARIANTS",
    "load_config",
]