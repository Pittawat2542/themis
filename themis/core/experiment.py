"""Experiment authoring model and snapshot compilation for Phase 1."""

from __future__ import annotations

from pydantic import Field

from themis.core.base import FrozenModel
from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.models import Dataset
from themis.core.snapshot import (
    ComponentRefs,
    DatasetRef,
    RunIdentity,
    RunProvenance,
    RunSnapshot,
    component_ref_from_value,
)


class Experiment(FrozenModel):
    """Compiled-input experiment definition for Themis v4."""

    generation: GenerationConfig
    evaluation: EvaluationConfig
    storage: StorageConfig
    datasets: list[Dataset] = Field(default_factory=list)
    seeds: list[int] = Field(default_factory=list)
    environment_metadata: dict[str, str] = Field(default_factory=dict)
    themis_version: str = "4.0.0a0"
    python_version: str = "3.12"
    platform: str = "unknown"

    def compile(self) -> RunSnapshot:
        component_refs = ComponentRefs(
            generator=component_ref_from_value(self.generation.generator),
            reducer=component_ref_from_value(self.generation.reducer)
            if self.generation.reducer is not None
            else None,
            parsers=[component_ref_from_value(parser) for parser in self.evaluation.parsers],
            metrics=[component_ref_from_value(metric) for metric in self.evaluation.metrics],
        )
        identity = RunIdentity(
            dataset_refs=[
                DatasetRef(
                    dataset_id=dataset.dataset_id,
                    revision=dataset.revision,
                    fingerprint=dataset.compute_hash(),
                )
                for dataset in self.datasets
            ],
            generator_ref=component_refs.generator,
            reducer_ref=component_refs.reducer,
            parser_refs=component_refs.parsers,
            metric_refs=component_refs.metrics,
            candidate_policy=self.generation.candidate_policy,
            judge_config=self.evaluation.judge_config,
            seeds=self.seeds,
        )
        provenance = RunProvenance(
            themis_version=self.themis_version,
            python_version=self.python_version,
            platform=self.platform,
            storage=self.storage,
            environment_metadata=self.environment_metadata,
        )
        return RunSnapshot(
            identity=identity,
            provenance=provenance,
            component_refs=component_refs,
        )
