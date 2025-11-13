"""Configuration for the judge evaluation example."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class JudgeModelConfig:
    """Configuration for a judge model."""

    name: str
    provider: Literal["fake", "litellm"] = "fake"
    provider_options: dict[str, object] | None = None
    description: str | None = None


@dataclass
class RubricConfig:
    """Scoring rubric specification."""

    name: str
    criteria: dict[str, str]


@dataclass
class JudgeExperimentConfig:
    """Configuration for judge-based evaluation experiment."""

    run_id: str
    storage_dir: Path | str
    resume: bool = False
    judge_models: list[JudgeModelConfig] | None = None
    rubrics: list[RubricConfig] | None = None


DEFAULT_JUDGE_CONFIG = JudgeExperimentConfig(
    run_id="judge-eval-demo",
    storage_dir=Path("./themis_output/judge_eval"),
    judge_models=[
        JudgeModelConfig(
            name="judge-gpt4",
            provider="fake",
            description="Simulated GPT-4 judge",
        )
    ],
    rubrics=[
        RubricConfig(
            name="math-rubric",
            criteria={
                "correctness": "Answer matches ground truth",
                "reasoning": "Clear step-by-step explanation",
                "formatting": "Proper mathematical notation",
            },
        )
    ],
)
