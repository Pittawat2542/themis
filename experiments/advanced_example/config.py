"""Configuration for the advanced example experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Literal


from experiments.example.config import (
    DatasetConfig,
    ExampleExperimentConfig,
    ModelConfig,
    SamplingProfile,
    load_config as load_base_config,
)


class AdvancedExperimentConfig(ExampleExperimentConfig):
    enable_subject_breakdown: bool = True
    prompt_style: Literal["cot", "concise"] = "cot"
    test_time_attempts: int = 1

    def apply_overrides(
        self,
        *,
        run_id: str | None = None,
        storage_dir: Path | None = None,
        resume: bool | None = None,
        enable_subject_breakdown: bool | None = None,
        prompt_style: str | None = None,
        test_time_attempts: int | None = None,
    ) -> "AdvancedExperimentConfig":
        base = super().apply_overrides(
            run_id=run_id, storage_dir=storage_dir, resume=resume
        )
        data = base.model_dump()
        data.setdefault("enable_subject_breakdown", self.enable_subject_breakdown)
        data.setdefault("prompt_style", self.prompt_style)
        data.setdefault("test_time_attempts", self.test_time_attempts)
        if enable_subject_breakdown is not None:
            data["enable_subject_breakdown"] = enable_subject_breakdown
        if prompt_style is not None:
            data["prompt_style"] = prompt_style
        if test_time_attempts is not None:
            data["test_time_attempts"] = test_time_attempts
        return AdvancedExperimentConfig.model_validate(data)


ADVANCED_DEFAULT_CONFIG = AdvancedExperimentConfig(
    run_id="advanced-demo",
    storage_dir=Path(".cache/advanced-demo"),
    models=[ModelConfig(name="fake-math-llm", provider="fake")],
    samplings=[SamplingProfile(name="cot", temperature=0.2, top_p=0.9)],
    datasets=[DatasetConfig(name="demo", kind="demo", limit=2)],
    test_time_attempts=2,
)


def load_config(path: Path) -> AdvancedExperimentConfig:
    base = load_base_config(path)
    return AdvancedExperimentConfig.model_validate(base.model_dump())


__all__ = [
    "AdvancedExperimentConfig",
    "ADVANCED_DEFAULT_CONFIG",
    "load_config",
    "DatasetConfig",
    "SamplingProfile",
    "ModelConfig",
]
