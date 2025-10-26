"""Configuration for the agentic experiment."""

from __future__ import annotations

from pathlib import Path

from experiments.example.config import (
    DatasetConfig,
    ExampleExperimentConfig,
    ModelConfig,
    SamplingProfile,
    load_config as load_base_config,
)


class AgenticExperimentConfig(ExampleExperimentConfig):
    planner_prompt: str = (
        "Break the problem into numbered steps before answering: {problem}"
    )
    final_prompt_prefix: str = "Use the above plan to craft the final answer."

    def apply_overrides(
        self,
        *,
        run_id: str | None = None,
        storage_dir: Path | None = None,
        resume: bool | None = None,
        planner_prompt: str | None = None,
        final_prompt_prefix: str | None = None,
    ) -> "AgenticExperimentConfig":
        base = super().apply_overrides(
            run_id=run_id, storage_dir=storage_dir, resume=resume
        )
        data = base.model_dump()
        if planner_prompt is not None:
            data["planner_prompt"] = planner_prompt
        if final_prompt_prefix is not None:
            data["final_prompt_prefix"] = final_prompt_prefix
        return AgenticExperimentConfig.model_validate(data)


AGENTIC_DEFAULT_CONFIG = AgenticExperimentConfig(
    run_id="agentic-demo",
    storage_dir=Path(".cache/agentic-demo"),
    models=[ModelConfig(name="fake-math-llm", provider="fake")],
    samplings=[SamplingProfile(name="agent", temperature=0.3, top_p=0.9)],
    datasets=[DatasetConfig(name="demo", kind="demo", limit=2)],
)


def load_config(path: Path) -> AgenticExperimentConfig:
    base = load_base_config(path)
    return AgenticExperimentConfig.model_validate(base.model_dump())


__all__ = [
    "AgenticExperimentConfig",
    "AGENTIC_DEFAULT_CONFIG",
    "load_config",
]
