"""Benchmark catalog loader and definition types."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field

from themis.catalog.loaders import load_toml
from themis.core.base import FrozenModel
from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.experiment import Experiment
from themis.core.models import Case, Dataset
from themis.core.store import RunStore


class BenchmarkDefinition(FrozenModel):
    benchmark_id: str
    base_benchmark_id: str
    dataset_id: str
    split: str
    variant: str | None = None
    metric_ids: list[str] = Field(default_factory=lambda: ["builtin/exact_match"])
    parser_ids: list[str] = Field(default_factory=lambda: ["builtin/json_identity"])
    judge_model_ids: list[str] = Field(default_factory=list)
    reducer_id: str = "builtin/majority_vote"
    generator_id: str = "builtin/demo_generator"

    def build_experiment(
        self,
        *,
        model: object | None = None,
        storage: StorageConfig | None = None,
    ) -> Experiment:
        generator = model if model is not None else self.generator_id
        return Experiment(
            generation=GenerationConfig(
                generator=generator,
                candidate_policy={"num_samples": 1},
                reducer=self.reducer_id,
            ),
            evaluation=EvaluationConfig(
                metrics=self.metric_ids,
                parsers=self.parser_ids,
                judge_models=self.judge_model_ids,
            ),
            storage=storage or StorageConfig(store="memory"),
            datasets=[
                Dataset(
                    dataset_id=self.dataset_id,
                    revision=self.variant or self.split,
                    metadata={"split": self.split, "benchmark_id": self.benchmark_id},
                    cases=[
                        Case(
                            case_id=f"{self.base_benchmark_id}-sample-1",
                            input={"prompt": f"sample prompt for {self.benchmark_id}"},
                            expected_output={"answer": f"sample answer for {self.base_benchmark_id}"},
                        )
                    ],
                )
            ],
            seeds=[7],
        )


@lru_cache(maxsize=1)
def benchmark_specs() -> dict[str, dict[str, object]]:
    manifest_path = Path(__file__).with_name("manifests") / "benchmarks.toml"
    payload = load_toml(manifest_path)
    return {
        benchmark_id: dict(entry)
        for benchmark_id, entry in payload.get("benchmarks", {}).items()
    }


def load_benchmark(name: str) -> BenchmarkDefinition:
    base_name, separator, variant = name.partition(":")
    try:
        spec = benchmark_specs()[base_name]
    except KeyError as exc:
        raise ValueError(f"Unknown catalog benchmark: {name}") from exc

    resolved_variant = _resolve_variant(base_name, variant if separator else None, spec)
    benchmark_id = base_name if resolved_variant is None else f"{base_name}:{resolved_variant}"
    return BenchmarkDefinition(
        benchmark_id=benchmark_id,
        base_benchmark_id=base_name,
        dataset_id=str(spec["dataset_id"]),
        split=str(spec["split"]),
        variant=resolved_variant,
        metric_ids=list(spec.get("metric_ids", ["builtin/exact_match"])),
        parser_ids=list(spec.get("parser_ids", ["builtin/json_identity"])),
        judge_model_ids=list(spec.get("judge_model_ids", [])),
        reducer_id=str(spec.get("reducer_id", "builtin/majority_vote")),
        generator_id=str(spec.get("generator_id", "builtin/demo_generator")),
    )


def run_benchmark(name: str, *, model: object | None = None, store: RunStore | None = None):
    definition = load_benchmark(name)
    storage = StorageConfig(store="memory") if store is None else None
    experiment = definition.build_experiment(model=model, storage=storage)
    return experiment.run(store=store)


def _resolve_variant(base_name: str, variant: str | None, spec: dict[str, object]) -> str | None:
    allowed = list(spec.get("variants", []))
    variant_mode = str(spec.get("variant_mode", "none"))
    if variant is None:
        return None
    if variant_mode == "open":
        if not variant:
            raise ValueError(f"Invalid variant for {base_name}: {variant}")
        return variant
    if variant in allowed:
        return variant
    raise ValueError(f"Invalid variant for {base_name}: {variant}")
