"""Configuration models for the example experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


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
    provider: str = "fake"
    description: str | None = None
    provider_options: dict[str, object] = Field(default_factory=dict)


class DatasetConfig(BaseModel):
    name: str
    kind: Literal["demo", "math500_local", "math500_hf"]
    limit: int | None = None
    subjects: tuple[str, ...] = Field(default_factory=tuple)
    data_dir: Path | None = None


class ExampleExperimentConfig(BaseModel):
    run_id: str = "example-demo"
    storage_dir: Path = Path(".cache/example")
    resume: bool = True
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
    ) -> "ExampleExperimentConfig":
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


DEFAULT_CONFIG = ExampleExperimentConfig(
    models=[
        ModelConfig(
            name="fake-math-llm",
            provider="fake",
            description="Built-in heuristic model",
        )
    ],
    samplings=[SamplingProfile(name="zero-shot", temperature=0.0)],
    datasets=[DatasetConfig(name="demo", kind="demo", limit=2)],
)


def load_config(path: Path) -> ExampleExperimentConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    return ExampleExperimentConfig.model_validate(data)


__all__ = [
    "DatasetConfig",
    "ExampleExperimentConfig",
    "ModelConfig",
    "SamplingProfile",
    "DEFAULT_CONFIG",
    "load_config",
]
