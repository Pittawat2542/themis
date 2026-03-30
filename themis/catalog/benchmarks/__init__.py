"""Benchmark catalog loader and definition types."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import cast

from pydantic import Field

from themis.catalog.benchmarks.adapters import apply_benchmark_adapter
from themis.catalog.loaders import load_toml
from themis.core.base import FrozenModel, JSONValue
from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.experiment import Experiment
from themis.core.models import Case, Dataset
from themis.core.protocols import Generator
from themis.core.store import RunStore


def _default_candidate_policy() -> dict[str, JSONValue]:
    return {"num_samples": 1}


class BenchmarkDefinition(FrozenModel):
    benchmark_id: str
    base_benchmark_id: str
    dataset_id: str
    dataset_revision: str | None = None
    split: str
    variant: str | None = None
    requires_code_execution: bool = False
    supported_execution_backends: list[str] = Field(default_factory=list)
    metric_ids: list[str] = Field(default_factory=lambda: ["builtin/exact_match"])
    parser_ids: list[str] = Field(default_factory=lambda: ["builtin/json_identity"])
    judge_model_ids: list[str] = Field(default_factory=list)
    reducer_id: str = "builtin/majority_vote"
    generator_id: str = "builtin/demo_generator"
    candidate_policy: dict[str, JSONValue] = Field(default_factory=_default_candidate_policy)
    workflow_overrides: dict[str, JSONValue] = Field(default_factory=dict)
    dataset_metadata: dict[str, str] = Field(default_factory=dict)
    sample_case_id: str = "sample-1"
    sample_case_input: JSONValue = Field(default_factory=dict)
    sample_case_expected_output: JSONValue | None = None
    sample_case_metadata: dict[str, str] = Field(default_factory=dict)

    def build_experiment(
        self,
        *,
        model: object | None = None,
        storage: StorageConfig | None = None,
    ) -> Experiment:
        generator: Generator | str = cast(Generator | str, model) if model is not None else self.generator_id
        seeds = list(range(7, 7 + _candidate_count(self.candidate_policy)))
        return Experiment(
            generation=GenerationConfig(
                generator=generator,
                candidate_policy=self.candidate_policy,
                reducer=self.reducer_id,
            ),
            evaluation=EvaluationConfig(
                metrics=[*self.metric_ids],
                parsers=[*self.parser_ids],
                judge_models=[*self.judge_model_ids],
                workflow_overrides=self.workflow_overrides,
            ),
            storage=storage or StorageConfig(store="memory"),
            datasets=[
                Dataset(
                    dataset_id=self.dataset_id,
                    revision=self.variant or self.dataset_revision or self.split,
                    metadata={
                        "split": self.split,
                        "benchmark_id": self.benchmark_id,
                        "dataset_revision": self.dataset_revision or "",
                        "requires_code_execution": str(self.requires_code_execution).lower(),
                        "supported_execution_backends": ",".join(self.supported_execution_backends),
                        **self.dataset_metadata,
                    },
                    cases=[
                        Case(
                            case_id=self.sample_case_id,
                            input=self.sample_case_input,
                            expected_output=self.sample_case_expected_output,
                            metadata=self.sample_case_metadata,
                        )
                    ],
                )
            ],
            seeds=seeds,
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
    adapter_payload = apply_benchmark_adapter(
        _optional_str(spec.get("adapter")),
        base_name=base_name,
        benchmark_id=benchmark_id,
        spec=spec,
        variant=resolved_variant,
    )
    return BenchmarkDefinition(
        benchmark_id=benchmark_id,
        base_benchmark_id=base_name,
        dataset_id=_required_str(spec, "dataset_id"),
        dataset_revision=_optional_str(spec.get("dataset_revision")),
        split=_required_str(spec, "split"),
        variant=resolved_variant,
        requires_code_execution=bool(spec.get("requires_code_execution", False)),
        supported_execution_backends=_string_list_from_value(spec.get("supported_execution_backends", [])),
        metric_ids=_string_list_from_value(spec.get("metric_ids", adapter_payload.get("metric_ids", ["builtin/exact_match"]))),
        parser_ids=_string_list_from_value(spec.get("parser_ids", adapter_payload.get("parser_ids", ["builtin/json_identity"]))),
        judge_model_ids=_string_list_from_value(spec.get("judge_model_ids", adapter_payload.get("judge_model_ids", []))),
        reducer_id=_string_from_value(spec.get("reducer_id", adapter_payload.get("reducer_id", "builtin/majority_vote"))),
        generator_id=_string_from_value(spec.get("generator_id", adapter_payload.get("generator_id", "builtin/demo_generator"))),
        candidate_policy=_json_mapping_from_value(adapter_payload.get("candidate_policy", {"num_samples": 1})),
        workflow_overrides=_json_mapping_from_value(adapter_payload.get("workflow_overrides", {})),
        dataset_metadata=_string_mapping_from_value(adapter_payload.get("dataset_metadata", {})),
        sample_case_id=_string_from_value(adapter_payload.get("sample_case_id", f"{base_name}-sample-1")),
        sample_case_input=_json_value_from_value(
            adapter_payload.get("sample_case_input", {"prompt": f"sample prompt for {benchmark_id}"})
        ),
        sample_case_expected_output=_json_optional_value_from_value(
            adapter_payload.get(
                "sample_case_expected_output",
                {"answer": f"sample answer for {base_name}"},
            )
        ),
        sample_case_metadata=_string_mapping_from_value(adapter_payload.get("sample_case_metadata", {})),
    )


def run_benchmark(name: str, *, model: object | None = None, store: RunStore | None = None):
    definition = load_benchmark(name)
    storage = StorageConfig(store="memory") if store is None else None
    experiment = definition.build_experiment(model=model, storage=storage)
    return experiment.run(store=store)


def _resolve_variant(base_name: str, variant: str | None, spec: dict[str, object]) -> str | None:
    allowed = _string_list_from_value(spec.get("variants", []))
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


def _candidate_count(candidate_policy: dict[str, JSONValue]) -> int:
    count = candidate_policy.get("num_samples", 1)
    return int(count) if isinstance(count, int) and count > 0 else 1


def _required_str(payload: dict[str, object], key: str) -> str:
    return _string_from_value(payload.get(key))


def _optional_str(value: object | None) -> str | None:
    if value is None:
        return None
    return _string_from_value(value)


def _string_from_value(value: object | None) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Expected string value, got {type(value).__name__}")
    return value


def _string_list_from_value(value: object) -> list[str]:
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    raise ValueError(f"Expected list[str], got {type(value).__name__}")


def _string_mapping_from_value(value: object) -> dict[str, str]:
    if isinstance(value, dict) and all(isinstance(key, str) and isinstance(item, str) for key, item in value.items()):
        return dict(value)
    raise ValueError(f"Expected dict[str, str], got {type(value).__name__}")


def _json_mapping_from_value(value: object) -> dict[str, JSONValue]:
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return cast(dict[str, JSONValue], dict(value))
    raise ValueError(f"Expected JSON object, got {type(value).__name__}")


def _json_value_from_value(value: object) -> JSONValue:
    return cast(JSONValue, value)


def _json_optional_value_from_value(value: object) -> JSONValue | None:
    return cast(JSONValue | None, value)
