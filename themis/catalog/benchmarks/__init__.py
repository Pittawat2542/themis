"""Benchmark catalog loader and definition types."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import cast

from pydantic import Field

from themis.catalog.benchmarks.adapters import apply_benchmark_adapter
from themis.catalog.benchmarks.materializers import materialize_benchmark_dataset
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
    source_kind: str = "huggingface_dataset"
    source_files: list[str] = Field(default_factory=list)
    source_file_map: dict[str, str] = Field(default_factory=dict)
    support_tier: str = "ready"
    version_notes: str | None = None
    requires_code_execution: bool = False
    supported_execution_backends: list[str] = Field(default_factory=list)
    metric_ids: list[str] = Field(default_factory=lambda: ["builtin/exact_match"])
    parser_ids: list[str] = Field(default_factory=lambda: ["builtin/json_identity"])
    judge_model_ids: list[str] = Field(default_factory=list)
    selector_id: str | None = None
    reducer_id: str | None = "builtin/majority_vote"
    generator_id: str = "builtin/demo_generator"
    candidate_policy: dict[str, JSONValue] = Field(
        default_factory=_default_candidate_policy
    )
    workflow_overrides: dict[str, JSONValue] = Field(default_factory=dict)
    dataset_metadata: dict[str, str] = Field(default_factory=dict)
    sample_case_id: str = "sample-1"
    sample_case_input: JSONValue = Field(default_factory=dict)
    sample_case_expected_output: JSONValue | None = None
    sample_case_metadata: dict[str, str] = Field(default_factory=dict)

    def materialize_dataset(self, *, loader=None) -> Dataset:
        return materialize_benchmark_dataset(self, loader=loader)

    def build_experiment(
        self,
        *,
        dataset: Dataset | None = None,
        model: object | None = None,
        storage: StorageConfig | None = None,
    ) -> Experiment:
        generator: Generator | str = (
            cast(Generator | str, model) if model is not None else self.generator_id
        )
        seeds = list(range(7, 7 + _candidate_count(self.candidate_policy)))
        resolved_dataset = dataset or Dataset(
            dataset_id=self.dataset_id,
            revision=self.variant or self.dataset_revision or self.split,
            metadata={
                "split": self.split,
                "benchmark_id": self.benchmark_id,
                "dataset_revision": self.dataset_revision or "",
                "requires_code_execution": str(self.requires_code_execution).lower(),
                "supported_execution_backends": ",".join(
                    self.supported_execution_backends
                ),
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
        return Experiment(
            generation=GenerationConfig(
                generator=generator,
                candidate_policy=self.candidate_policy,
                selector=self.selector_id,
                reducer=self.reducer_id,
            ),
            evaluation=EvaluationConfig(
                metrics=[*self.metric_ids],
                parsers=[*self.parser_ids],
                judge_models=[*self.judge_model_ids],
                workflow_overrides=self.workflow_overrides,
            ),
            storage=storage or StorageConfig(store="memory"),
            datasets=[resolved_dataset],
            seeds=seeds,
        )


class BenchmarkCatalogEntry(FrozenModel):
    benchmark_id: str
    base_benchmark_id: str
    variant_mode: str = "none"
    declared_variants: list[str] = Field(default_factory=list)
    dataset_id: str
    split: str
    dataset_revision: str | None = None
    source_kind: str = "huggingface_dataset"
    requires_code_execution: bool = False
    supported_execution_backends: list[str] = Field(default_factory=list)
    parser_ids: list[str] = Field(default_factory=list)
    metric_ids: list[str] = Field(default_factory=list)
    support_tier: str = "ready"
    version_notes: str | None = None


class BenchmarkValidationCheck(FrozenModel):
    status: str
    message: str | None = None


class BenchmarkValidationResult(FrozenModel):
    benchmark_id: str
    support_tier: str
    checks: dict[str, BenchmarkValidationCheck]
    issues: list[str] = Field(default_factory=list)


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
    benchmark_id = (
        base_name if resolved_variant is None else f"{base_name}:{resolved_variant}"
    )
    adapter_payload = apply_benchmark_adapter(
        _optional_str(spec.get("adapter")),
        base_name=base_name,
        benchmark_id=benchmark_id,
        spec=spec,
        variant=resolved_variant,
    )
    recipe_payload = _recipe_defaults(base_name, variant=resolved_variant)
    resolved_payload = {**recipe_payload, **adapter_payload}
    return BenchmarkDefinition(
        benchmark_id=benchmark_id,
        base_benchmark_id=base_name,
        dataset_id=_required_str(spec, "dataset_id"),
        dataset_revision=_optional_str(spec.get("dataset_revision")),
        split=_required_str(spec, "split"),
        variant=resolved_variant,
        source_kind=_string_from_value(spec.get("source_kind", "huggingface_dataset")),
        source_files=_string_list_from_value(spec.get("source_files", [])),
        source_file_map=_string_mapping_from_value(spec.get("source_file_map", {})),
        support_tier=_string_from_value(spec.get("support_tier", "ready")),
        version_notes=_optional_str(spec.get("version_notes")),
        requires_code_execution=bool(spec.get("requires_code_execution", False)),
        supported_execution_backends=_string_list_from_value(
            spec.get("supported_execution_backends", [])
        ),
        metric_ids=_string_list_from_value(
            spec.get(
                "metric_ids",
                resolved_payload.get("metric_ids", ["builtin/exact_match"]),
            )
        ),
        parser_ids=_string_list_from_value(
            spec.get(
                "parser_ids",
                resolved_payload.get("parser_ids", ["builtin/json_identity"]),
            )
        ),
        judge_model_ids=_string_list_from_value(
            spec.get("judge_model_ids", resolved_payload.get("judge_model_ids", []))
        ),
        selector_id=_optional_str(
            spec.get("selector_id", resolved_payload.get("selector_id"))
        ),
        reducer_id=_optional_str(
            spec.get(
                "reducer_id",
                resolved_payload.get(
                    "reducer_id",
                    None
                    if "selector_id" in resolved_payload
                    else "builtin/majority_vote",
                ),
            )
        ),
        generator_id=_string_from_value(
            spec.get(
                "generator_id",
                resolved_payload.get("generator_id", "builtin/demo_generator"),
            )
        ),
        candidate_policy=_json_mapping_from_value(
            resolved_payload.get("candidate_policy", {"num_samples": 1})
        ),
        workflow_overrides=_json_mapping_from_value(
            resolved_payload.get("workflow_overrides", {})
        ),
        dataset_metadata=_string_mapping_from_value(
            resolved_payload.get("dataset_metadata", {})
        ),
        sample_case_id=_string_from_value(
            resolved_payload.get("sample_case_id", f"{base_name}-sample-1")
        ),
        sample_case_input=_json_value_from_value(
            resolved_payload.get(
                "sample_case_input", {"prompt": f"sample prompt for {benchmark_id}"}
            )
        ),
        sample_case_expected_output=_json_optional_value_from_value(
            resolved_payload.get(
                "sample_case_expected_output",
                {"answer": f"sample answer for {base_name}"},
            )
        ),
        sample_case_metadata=_string_mapping_from_value(
            resolved_payload.get("sample_case_metadata", {})
        ),
    )


def run_benchmark(
    name: str, *, model: object | None = None, store: RunStore | None = None
):
    definition = load_benchmark(name)
    dataset = definition.materialize_dataset()
    storage = StorageConfig(store="memory") if store is None else None
    experiment = definition.build_experiment(
        dataset=dataset, model=model, storage=storage
    )
    return experiment.run(store=store)


def list_benchmark_ids() -> list[str]:
    return sorted(benchmark_specs())


def list_benchmarks() -> list[BenchmarkCatalogEntry]:
    return [get_benchmark(benchmark_id) for benchmark_id in list_benchmark_ids()]


def get_benchmark(name: str) -> BenchmarkCatalogEntry:
    definition = load_benchmark(name)
    spec = benchmark_specs()[definition.base_benchmark_id]
    return BenchmarkCatalogEntry(
        benchmark_id=definition.base_benchmark_id,
        base_benchmark_id=definition.base_benchmark_id,
        variant_mode=_string_from_value(spec.get("variant_mode", "none")),
        declared_variants=_string_list_from_value(spec.get("variants", [])),
        dataset_id=definition.dataset_id,
        split=definition.split,
        dataset_revision=definition.dataset_revision,
        source_kind=definition.source_kind,
        requires_code_execution=definition.requires_code_execution,
        supported_execution_backends=definition.supported_execution_backends,
        parser_ids=definition.parser_ids,
        metric_ids=definition.metric_ids,
        support_tier=definition.support_tier,
        version_notes=definition.version_notes,
    )


def validate_benchmark(name: str) -> BenchmarkValidationResult:
    issues: list[str] = []
    checks: dict[str, BenchmarkValidationCheck] = {}

    try:
        definition = load_benchmark(name)
    except Exception as exc:
        message = str(exc)
        checks["load"] = BenchmarkValidationCheck(status="failed", message=message)
        checks["materialize"] = BenchmarkValidationCheck(
            status="skipped", message="Benchmark failed to load."
        )
        checks["score_smoke"] = BenchmarkValidationCheck(
            status="skipped", message="Benchmark failed to load."
        )
        return BenchmarkValidationResult(
            benchmark_id=name,
            support_tier="unsupported",
            checks=checks,
            issues=[message],
        )

    checks["load"] = BenchmarkValidationCheck(status="passed")
    dataset = None

    try:
        dataset = definition.materialize_dataset()
        if not dataset.cases:
            raise ValueError("Benchmark materialized an empty dataset.")
        checks["materialize"] = BenchmarkValidationCheck(status="passed")
    except Exception as exc:
        message = str(exc)
        issues.append(message)
        checks["materialize"] = BenchmarkValidationCheck(
            status="failed", message=message
        )
        checks["score_smoke"] = BenchmarkValidationCheck(
            status="skipped", message="Materialization failed."
        )
        return BenchmarkValidationResult(
            benchmark_id=definition.benchmark_id,
            support_tier=definition.support_tier,
            checks=checks,
            issues=issues,
        )

    if not definition.requires_code_execution:
        checks["score_smoke"] = BenchmarkValidationCheck(
            status="skipped", message="Benchmark does not require code execution."
        )
        return BenchmarkValidationResult(
            benchmark_id=definition.benchmark_id,
            support_tier=definition.support_tier,
            checks=checks,
            issues=issues,
        )

    if definition.support_tier != "ready":
        message = "Benchmark is not ready for execution score smoke validation."
        checks["score_smoke"] = BenchmarkValidationCheck(
            status="skipped",
            message=message,
        )
        return BenchmarkValidationResult(
            benchmark_id=definition.benchmark_id,
            support_tier=definition.support_tier,
            checks=checks,
            issues=issues,
        )

    try:
        assert dataset is not None
        _validate_execution_wiring(definition, dataset)
        _score_smoke(definition, dataset)
        checks["score_smoke"] = BenchmarkValidationCheck(status="passed")
    except Exception as exc:
        message = str(exc)
        issues.append(message)
        checks["score_smoke"] = BenchmarkValidationCheck(
            status="failed", message=message
        )

    return BenchmarkValidationResult(
        benchmark_id=definition.benchmark_id,
        support_tier=definition.support_tier,
        checks=checks,
        issues=issues,
    )


def _resolve_variant(
    base_name: str, variant: str | None, spec: dict[str, object]
) -> str | None:
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


def _recipe_defaults(base_name: str, *, variant: str | None) -> dict[str, object]:
    del variant
    if base_name in {
        "mmlu_pro",
        "supergpqa",
        "encyclo_k",
        "babe",
        "gpqa_diamond",
        "mmmlu",
        "superchem",
    }:
        return {
            "metric_ids": ["builtin/choice_accuracy"],
            "parser_ids": ["builtin/choice_letter"],
        }
    if base_name in {
        "aime_2025",
        "aime_2026",
        "apex_2025",
        "beyond_aime",
        "hmmt_feb_2025",
        "hmmt_nov_2025",
        "imo_answerbench",
        "phybench",
    }:
        return {
            "metric_ids": ["builtin/math_equivalence"],
            "parser_ids": ["builtin/math_answer"],
        }
    if base_name == "procbench":
        return {
            "metric_ids": ["builtin/procbench_final_accuracy"],
            "parser_ids": ["builtin/text"],
        }
    if base_name == "codeforces":
        return {
            "metric_ids": ["builtin/codeforces_pass_rate"],
            "parser_ids": ["builtin/code_text"],
        }
    if base_name == "aethercode":
        return {
            "metric_ids": ["builtin/aethercode_pass_rate"],
            "parser_ids": ["builtin/code_text"],
        }
    if base_name == "livecodebench":
        return {
            "metric_ids": ["builtin/livecodebench_pass_rate"],
            "parser_ids": ["builtin/code_text"],
        }
    if base_name in {"humaneval", "humaneval_plus"}:
        return {
            "metric_ids": ["builtin/humaneval_pass_rate"],
            "parser_ids": ["builtin/code_text"],
        }
    return {}


def _validate_execution_wiring(definition: BenchmarkDefinition, dataset: Dataset) -> None:
    if definition.parser_ids == ["builtin/json_identity"] and definition.metric_ids == [
        "builtin/exact_match"
    ]:
        raise ValueError(
            "Code-execution benchmark resolves to non-execution parser/metric wiring."
        )
    first_case = dataset.cases[0]
    expected_output = first_case.expected_output
    if not isinstance(expected_output, dict) or "official_tests" not in expected_output:
        raise ValueError(
            "Code-execution benchmark materializer did not produce an execution payload."
        )


def _score_smoke(definition: BenchmarkDefinition, dataset: Dataset) -> None:
    from themis.catalog import load as load_catalog
    from themis.core.contexts import ParseContext, ScoreContext
    from themis.core.models import ReducedCandidate
    from themis.core.protocols import Parser, PureMetric

    if not definition.parser_ids or not definition.metric_ids:
        raise ValueError("Benchmark is missing parser or metric configuration.")

    parser = cast(Parser, load_catalog(definition.parser_ids[0]))
    metric = cast(PureMetric, load_catalog(definition.metric_ids[0]))
    case = dataset.cases[0]
    candidate_output = "def solve():\n    return 'validation'\n"
    if isinstance(case.expected_output, dict):
        solution = case.expected_output.get("solution")
        if isinstance(solution, str) and solution.strip():
            candidate_output = solution
    parsed = parser.parse(
        ReducedCandidate(
            candidate_id="validation-candidate",
            final_output=candidate_output,
        ),
        ParseContext(
            run_id="benchmark-validation",
            case_id=case.case_id,
            candidate_id="validation-candidate",
        ),
    )
    metric.score(
        parsed,
        case,
        ScoreContext(
            run_id="benchmark-validation",
            case=case,
            parsed_output=parsed,
        ),
    )


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
    if isinstance(value, dict) and all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
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
