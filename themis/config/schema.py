"""Structured configuration definitions for Hydra/OmegaConf."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class ProviderConfig:
    name: str = "fake"
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunnerConfig:
    max_parallel: int = 1
    max_retries: int = 3
    retry_initial_delay: float = 0.5
    retry_backoff_multiplier: float = 2.0
    retry_max_delay: float | None = 2.0


@dataclass
class SamplingConfig:
    temperature: float = 0.0
    top_p: float = 0.95
    max_tokens: int = 512


@dataclass
class GenerationConfig:
    model_identifier: str = "fake-math-llm"
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    runner: RunnerConfig = field(default_factory=RunnerConfig)


@dataclass
class DatasetConfig:
    source: str = "huggingface"
    data_dir: str | None = None
    limit: int | None = None
    split: str = "test"
    subjects: list[str] = field(default_factory=list)
    inline_samples: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class StorageConfig:
    path: str | None = None
    default_path: str | None = None  # New field for default storage path


@dataclass
class WandbConfig:
    enable: bool = False
    project: str | None = None
    entity: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class HuggingFaceHubConfig:
    enable: bool = False
    repository: str | None = None


@dataclass
class IntegrationsConfig:
    wandb: WandbConfig = field(default_factory=WandbConfig)
    huggingface_hub: HuggingFaceHubConfig = field(default_factory=HuggingFaceHubConfig)


@dataclass
class ExperimentConfig:
    name: str = "math500_zero_shot"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    integrations: IntegrationsConfig = field(default_factory=IntegrationsConfig)
    max_samples: int | None = None
    run_id: str | None = None
    resume: bool = True
