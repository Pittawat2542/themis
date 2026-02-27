"""Structured configuration definitions for Hydra/OmegaConf."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from themis.core.entities import SamplingConfig


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
class ExtractorConfig:
    name: str = "identity"
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricConfig:
    name: str
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    extractor: ExtractorConfig | None = None
    metrics: list[MetricConfig] = field(default_factory=list)


@dataclass
class GenerationConfig:
    model_identifier: str = ""
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    runner: RunnerConfig = field(default_factory=RunnerConfig)


@dataclass
class DatasetConfig:
    source: str = "huggingface"
    dataset_id: str | None = None
    data_dir: str | None = None
    limit: int | None = None
    split: str = "test"
    subjects: list[str] = field(default_factory=list)
    inline_samples: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class StorageConfig:
    path: str | None = None
    default_path: str | None = None


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
class LangfuseConfig:
    enable: bool = False
    public_key: str | None = None
    secret_key: str | None = None
    base_url: str | None = None
    tags: list[str] = field(default_factory=list)
    trace_name: str | None = None
    enable_tracing: bool = True


@dataclass
class IntegrationsConfig:
    wandb: WandbConfig = field(default_factory=WandbConfig)
    huggingface_hub: HuggingFaceHubConfig = field(default_factory=HuggingFaceHubConfig)
    langfuse: LangfuseConfig = field(default_factory=LangfuseConfig)


@dataclass
class ExperimentConfig:
    name: str = "math500_zero_shot"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    pipeline: PipelineConfig | None = None  # Added generic pipeline config
    storage: StorageConfig = field(default_factory=StorageConfig)
    integrations: IntegrationsConfig = field(default_factory=IntegrationsConfig)
    max_samples: int | None = None
    run_id: str | None = None
    resume: bool = True
    task: str | None = None
    task_options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str | Path) -> ExperimentConfig:
        """Load configuration from a file."""
        from .loader import load_experiment_config

        return load_experiment_config(Path(path))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        """Create configuration from a dictionary."""
        from omegaconf import OmegaConf

        base = OmegaConf.structured(cls)
        merged = OmegaConf.merge(base, OmegaConf.create(data))
        return OmegaConf.to_object(merged)  # type: ignore

    def to_file(self, path: str | Path) -> None:
        """Save configuration to a file."""
        from omegaconf import OmegaConf

        conf = OmegaConf.structured(self)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(conf, Path(path))
