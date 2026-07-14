"""Experiment-building kits for manifest-backed benchmarks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated

from pydantic import Field

from themis.api import Evaluation, Experiment, Generation
from themis.core.base import FrozenModel, JSONValue
from themis.core.config import (
    GeneratorComponent,
    JudgeModelComponent,
    MetricComponent,
    ParserComponent,
    ReducerComponent,
    SelectorComponent,
)
from themis.core.dataset_sources import catalog_dataset_source, inline_dataset_source
from themis.core.models import Case, Dataset
from themis.core.prompts import PromptSpec


class BenchmarkExperimentDefaults(FrozenModel):
    """Defaults or overrides used when a Benchmark Kit builds an experiment."""

    generator: GeneratorComponent | None = None
    samples: Annotated[int, Field(ge=1)] | None = None
    generation_prompt: PromptSpec | None = None
    selector: SelectorComponent | None = None
    reducer: ReducerComponent | None = None
    parser: ParserComponent | None = None
    parser_views: Mapping[str, ParserComponent] | None = None
    metrics: list[MetricComponent] | None = None
    judge_models: list[JudgeModelComponent] | None = None
    evaluation_prompt: PromptSpec | None = None
    judge_options: dict[str, JSONValue] | None = None
    workflow_options: dict[str, JSONValue] | None = None

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
        overrides: BenchmarkExperimentDefaults | None = None,
        dataset: Dataset | None = None,
    ) -> Experiment:
        """Build a complete experiment for this kit's benchmark."""

        from themis.catalog.benchmarks import load_benchmark

        definition = load_benchmark(self.benchmark_id)
        defaults = self.defaults.merged_with(overrides)
        generator = defaults.generator or definition.generator_id
        samples = (
            defaults.samples
            if defaults.samples is not None
            else _candidate_count(definition.candidate_policy)
        )
        parser = defaults.parser
        parser_views = defaults.parser_views or {}
        if parser is None and not parser_views and definition.parser_ids:
            parser = definition.parser_ids[0]
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
            generation=Generation(
                generator=generator,
                samples=samples,
                prompt=defaults.generation_prompt,
                selector=defaults.selector
                if defaults.selector is not None
                else definition.selector_id,
                reducer=defaults.reducer
                if defaults.reducer is not None
                else definition.reducer_id,
            ),
            evaluation=Evaluation(
                metrics=list(metrics),
                parser=parser,
                parser_views=parser_views,
                judge_models=list(judge_models),
                prompt=defaults.evaluation_prompt,
                judge_options=defaults.judge_options or {},
                workflow_options=defaults.workflow_options
                if defaults.workflow_options is not None
                else definition.workflow_overrides,
            ),
            datasets=[
                inline_dataset_source(resolved_dataset)
                if dataset is not None
                else catalog_dataset_source(
                    benchmark_name=definition.benchmark_id,
                    dataset_id=resolved_dataset.dataset_id,
                    revision=resolved_dataset.revision,
                    provenance_metadata=resolved_dataset.metadata,
                )
            ],
            seeds=list(range(7, 7 + samples)),
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
            samples=_candidate_count(definition.candidate_policy),
            selector=definition.selector_id,
            reducer=definition.reducer_id,
            parser=definition.parser_ids[0] if definition.parser_ids else None,
            metrics=[*definition.metric_ids],
            judge_models=[*definition.judge_model_ids],
            workflow_options=definition.workflow_overrides,
        ),
    )


def build_benchmark_experiment(
    benchmark_id: str,
    *,
    overrides: BenchmarkExperimentDefaults | None = None,
    dataset: Dataset | None = None,
) -> Experiment:
    """Build a complete experiment from a benchmark kit."""

    return get_benchmark_kit(benchmark_id).build_experiment(
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
