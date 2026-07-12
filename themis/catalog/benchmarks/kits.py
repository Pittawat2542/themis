"""Experiment-building kits for manifest-backed benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from themis.core.base import FrozenModel, JSONValue
from themis.core.config import (
    EvaluationConfig,
    GeneratorComponent,
    JudgeModelComponent,
    MetricComponent,
    ParserComponent,
    ParserView,
    ReducerComponent,
    RuntimeConfig,
    SelectorComponent,
    GenerationConfig,
    StorageConfig,
)
from themis.core.dataset_sources import catalog_dataset_source, inline_dataset_source
from themis.core.experiment import Experiment
from themis.core.models import Case, Dataset
from themis.core.prompts import PromptSpec


class BenchmarkExperimentDefaults(FrozenModel):
    """Defaults or overrides used when a Benchmark Kit builds an experiment."""

    generator: GeneratorComponent | None = None
    candidate_policy: dict[str, JSONValue] | None = None
    prompt_spec: PromptSpec | None = None
    selector: SelectorComponent | None = None
    reducer: ReducerComponent | None = None
    parser_views: list[ParserView | ParserComponent] | None = None
    metrics: list[MetricComponent] | None = None
    judge_models: list[JudgeModelComponent] | None = None
    evaluation_prompt_spec: PromptSpec | None = None
    judge_config: dict[str, JSONValue] | None = None
    workflow_overrides: dict[str, JSONValue] | None = None

    def merged_with(
        self, overrides: BenchmarkExperimentDefaults | None
    ) -> BenchmarkExperimentDefaults:
        """Return these defaults with explicitly supplied override values applied."""

        if overrides is None:
            return self
        payload = self.model_dump()
        for field_name, value in overrides:
            if value is not None:
                payload[field_name] = value
        return BenchmarkExperimentDefaults.model_validate(payload)


@dataclass(frozen=True)
class BenchmarkKit:
    """Themis-owned helper that turns one catalog benchmark into experiments."""

    kit_id: str
    benchmark_id: str
    description: str
    defaults: BenchmarkExperimentDefaults

    def build_experiment(
        self,
        *,
        storage: StorageConfig | None = None,
        runtime: RuntimeConfig | None = None,
        overrides: BenchmarkExperimentDefaults | None = None,
        dataset: Dataset | None = None,
    ) -> Experiment:
        """Build a complete experiment for this kit's benchmark."""

        from themis.catalog.benchmarks import load_benchmark

        definition = load_benchmark(self.benchmark_id)
        defaults = self.defaults.merged_with(overrides)
        generator = defaults.generator or definition.generator_id
        candidate_policy = defaults.candidate_policy or definition.candidate_policy
        parser_views = (
            defaults.parser_views
            if defaults.parser_views is not None
            else [*definition.parser_ids]
        )
        metrics = (
            defaults.metrics
            if defaults.metrics is not None
            else [*definition.metric_ids]
        )
        judge_models = (
            defaults.judge_models
            if defaults.judge_models is not None
            else [*definition.judge_model_ids]
        )
        resolved_dataset = dataset or _sample_dataset(definition)
        return Experiment(
            generation=GenerationConfig(
                generator=generator,
                candidate_policy=candidate_policy,
                prompt_spec=defaults.prompt_spec,
                selector=defaults.selector
                if defaults.selector is not None
                else definition.selector_id,
                reducer=defaults.reducer
                if defaults.reducer is not None
                else definition.reducer_id,
            ),
            evaluation=EvaluationConfig(
                metrics=cast(list[MetricComponent], metrics),
                parsers=cast(list[ParserView | ParserComponent], parser_views),
                judge_models=cast(list[JudgeModelComponent], judge_models),
                prompt_spec=defaults.evaluation_prompt_spec,
                judge_config=defaults.judge_config or {},
                workflow_overrides=defaults.workflow_overrides
                if defaults.workflow_overrides is not None
                else definition.workflow_overrides,
            ),
            storage=storage or StorageConfig(target="memory"),
            runtime=runtime or RuntimeConfig(),
            dataset_sources=[
                inline_dataset_source(resolved_dataset)
                if dataset is not None
                else catalog_dataset_source(
                    benchmark_name=definition.benchmark_id,
                    dataset_id=resolved_dataset.dataset_id,
                    revision=resolved_dataset.revision,
                    provenance_metadata=resolved_dataset.metadata,
                )
            ],
            seeds=list(range(7, 7 + _candidate_count(candidate_policy))),
        )


def list_benchmark_kits() -> list[str]:
    """List benchmark identifiers with Themis-native experiment kits."""

    from themis.catalog.benchmarks import _materialization_benchmark_ids

    return _materialization_benchmark_ids()


def get_benchmark_kit(benchmark_id: str) -> BenchmarkKit:
    """Return a kit for a manifest-backed benchmark."""

    from themis.catalog.benchmarks import load_benchmark

    definition = load_benchmark(benchmark_id)
    return BenchmarkKit(
        kit_id=f"benchmark/{definition.benchmark_id}",
        benchmark_id=definition.benchmark_id,
        description=(
            "Build a Themis Experiment from the catalog definition for "
            f"{definition.benchmark_id}."
        ),
        defaults=BenchmarkExperimentDefaults(
            generator=definition.generator_id,
            candidate_policy=definition.candidate_policy,
            selector=definition.selector_id,
            reducer=definition.reducer_id,
            parser_views=[*definition.parser_ids],
            metrics=[*definition.metric_ids],
            judge_models=[*definition.judge_model_ids],
            workflow_overrides=definition.workflow_overrides,
        ),
    )


def build_benchmark_experiment(
    benchmark_id: str,
    *,
    storage: StorageConfig | None = None,
    runtime: RuntimeConfig | None = None,
    overrides: BenchmarkExperimentDefaults | None = None,
    dataset: Dataset | None = None,
) -> Experiment:
    """Build a complete experiment from a benchmark kit."""

    return get_benchmark_kit(benchmark_id).build_experiment(
        storage=storage,
        runtime=runtime,
        overrides=overrides,
        dataset=dataset,
    )


def _sample_dataset(definition) -> Dataset:
    return Dataset(
        dataset_id=definition.dataset_id,
        revision=definition.variant or definition.dataset_revision or definition.split,
        metadata={
            "split": definition.split,
            "benchmark_id": definition.benchmark_id,
            "dataset_revision": definition.dataset_revision or "",
            "requires_code_execution": str(definition.requires_code_execution).lower(),
            "supported_execution_backends": ",".join(
                definition.supported_execution_backends
            ),
            **definition.dataset_metadata,
        },
        cases=[
            Case(
                case_id=definition.sample_case_id,
                input=definition.sample_case_input,
                expected_output=definition.sample_case_expected_output,
                metadata=definition.sample_case_metadata,
            )
        ],
    )


def _candidate_count(candidate_policy: dict[str, JSONValue]) -> int:
    count = candidate_policy.get("num_samples", 1)
    return int(count) if isinstance(count, int) and count > 0 else 1
