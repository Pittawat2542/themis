from __future__ import annotations

from pathlib import Path

import pytest

from pydantic import ValidationError

from themis import (
    BenchmarkResult,
    BenchmarkSpec,
    DatasetQuerySpec,
    ExecutionPolicySpec,
    InferenceGridSpec,
    InferenceParamsSpec,
    ModelSpec,
    Orchestrator,
    PluginRegistry,
    ProjectSpec,
    PromptMessage,
    PromptVariantSpec,
    ScoreSpec,
    SliceSpec,
    StorageSpec,
)
from themis.contracts.protocols import InferenceResult
from themis.orchestration.run_manifest import RunHandle
from themis.records import InferenceRecord, MetricScore
from themis.specs.experiment import RuntimeContext
from themis.specs.foundational import DatasetSpec, GenerationSpec
from themis.types.enums import CompressionCodec, DatasetSource, PromptRole


class DemoDatasetProvider:
    def scan(self, slice_spec, query):
        del query
        return [{"item_id": "item-1", "question": "2 + 2", "answer": "4"}]


class DemoEngine:
    def infer(self, trial, context, runtime):
        del trial, runtime
        return InferenceResult(
            inference=InferenceRecord(
                spec_hash=f"inf_{context['item_id']}",
                raw_text=str(context["answer"]),
            )
        )


class ExactMatchMetric:
    def score(self, trial, candidate, context):
        del trial
        actual = candidate.inference.raw_text if candidate.inference else ""
        return MetricScore(
            metric_id="exact_match",
            value=float(actual == context["answer"]),
        )


def test_orchestrator_renders_benchmark_prompt_before_inference_and_preserves_prompt_metadata(
    tmp_path: Path,
) -> None:
    seen: dict[str, object] = {}

    class RenderingEngine:
        def infer(self, trial, context, runtime):
            del context, runtime
            seen["content"] = trial.prompt.messages[0].content
            seen["family"] = trial.prompt.family
            seen["variables"] = trial.prompt.variables
            return InferenceResult(
                inference=InferenceRecord(
                    spec_hash="inf_rendered",
                    raw_text="4",
                )
            )

    class RenderingMetric:
        def score(self, trial, candidate, context):
            del trial, candidate, context
            return MetricScore(metric_id="exact_match", value=1.0)

    project = ProjectSpec(
        project_name="bench-tests",
        researcher_id="tests",
        global_seed=7,
        storage=StorageSpec(
            root_dir=str(tmp_path / "store"),
            compression=CompressionCodec.NONE,
        ),
        execution_policy=ExecutionPolicySpec(),
    )
    registry = PluginRegistry()
    registry.register_inference_engine("demo", RenderingEngine())
    registry.register_metric("exact_match", RenderingMetric())

    orchestrator = Orchestrator.from_project_spec(
        project,
        registry=registry,
        dataset_provider=DemoDatasetProvider(),
    )
    benchmark = BenchmarkSpec(
        benchmark_id="demo-benchmark",
        models=[ModelSpec(model_id="demo-model", provider="demo")],
        slices=[
            SliceSpec(
                slice_id="qa",
                dataset=DatasetSpec(source=DatasetSource.MEMORY),
                dataset_query=DatasetQuerySpec.subset(1, seed=7),
                dimensions={"source": "synthetic", "format": "qa"},
                prompt_variant_ids=["qa-default"],
                generation=GenerationSpec(),
                scores=[ScoreSpec(name="default", metrics=["exact_match"])],
            )
        ],
        prompt_variants=[
            PromptVariantSpec(
                id="qa-default",
                family="qa",
                variables={"tone": "formal"},
                messages=[
                    PromptMessage(
                        role=PromptRole.USER,
                        content=(
                            "Solve: {item.question} "
                            "[{prompt.family}/{prompt.variables[tone]}] "
                            "{slice.dimensions[source]} "
                            "{runtime.run_labels[phase]}"
                        ),
                    )
                ],
            )
        ],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec(max_tokens=16)]),
    )

    result = orchestrator.run_benchmark(
        benchmark,
        runtime=RuntimeContext(run_labels={"phase": "smoke"}),
    )

    assert isinstance(result, BenchmarkResult)
    assert seen == {
        "content": "Solve: 2 + 2 [qa/formal] synthetic smoke",
        "family": "qa",
        "variables": {"tone": "formal"},
    }


def test_orchestrator_runs_benchmark_and_returns_benchmark_result(
    tmp_path: Path,
) -> None:
    project = ProjectSpec(
        project_name="bench-tests",
        researcher_id="tests",
        global_seed=7,
        storage=StorageSpec(
            root_dir=str(tmp_path / "store"),
            compression=CompressionCodec.NONE,
        ),
        execution_policy=ExecutionPolicySpec(),
    )
    registry = PluginRegistry()
    registry.register_inference_engine("demo", DemoEngine())
    registry.register_metric("exact_match", ExactMatchMetric())

    orchestrator = Orchestrator.from_project_spec(
        project,
        registry=registry,
        dataset_provider=DemoDatasetProvider(),
    )
    benchmark = BenchmarkSpec(
        benchmark_id="demo-benchmark",
        models=[ModelSpec(model_id="demo-model", provider="demo")],
        slices=[
            SliceSpec(
                slice_id="qa",
                dataset=DatasetSpec(source=DatasetSource.MEMORY),
                dataset_query=DatasetQuerySpec.subset(1, seed=7),
                dimensions={"source": "synthetic", "format": "qa"},
                prompt_variant_ids=["qa-default"],
                generation=GenerationSpec(),
                scores=[ScoreSpec(name="default", metrics=["exact_match"])],
            )
        ],
        prompt_variants=[
            PromptVariantSpec(
                id="qa-default",
                family="qa",
                messages=[
                    PromptMessage(
                        role=PromptRole.USER, content="Solve: {item.question}"
                    )
                ],
            )
        ],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec(max_tokens=16)]),
    )

    result = orchestrator.run_benchmark(benchmark)

    assert isinstance(result, BenchmarkResult)
    assert result.benchmark_id == "demo-benchmark"
    assert result.slice_ids == ["qa"]
    assert result.aggregate(
        group_by=["model_id", "slice_id", "metric_id", "source", "prompt_variant_id"]
    ) == [
        {
            "count": 1,
            "mean": 1.0,
            "metric_id": "exact_match",
            "model_id": "demo-model",
            "prompt_variant_id": "qa-default",
            "slice_id": "qa",
            "source": "synthetic",
        }
    ]


def _build_project(tmp_path: Path) -> ProjectSpec:
    return ProjectSpec(
        project_name="bench-tests",
        researcher_id="tests",
        global_seed=7,
        storage=StorageSpec(
            root_dir=str(tmp_path / "store"),
            compression=CompressionCodec.NONE,
        ),
        execution_policy=ExecutionPolicySpec(),
    )


def _build_benchmark() -> BenchmarkSpec:
    return BenchmarkSpec(
        benchmark_id="demo-benchmark",
        models=[ModelSpec(model_id="demo-model", provider="demo")],
        slices=[
            SliceSpec(
                slice_id="qa",
                dataset=DatasetSpec(source=DatasetSource.MEMORY),
                dataset_query=DatasetQuerySpec.subset(1, seed=7),
                dimensions={"source": "synthetic", "format": "qa"},
                prompt_variant_ids=["qa-default"],
                generation=GenerationSpec(),
                scores=[ScoreSpec(name="default", metrics=["exact_match"])],
            )
        ],
        prompt_variants=[
            PromptVariantSpec(
                id="qa-default",
                family="qa",
                messages=[
                    PromptMessage(
                        role=PromptRole.USER, content="Solve: {item.question}"
                    )
                ],
            )
        ],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec(max_tokens=16)]),
    )


def _build_orchestrator(tmp_path: Path) -> Orchestrator:
    registry = PluginRegistry()
    registry.register_inference_engine("demo", DemoEngine())
    registry.register_metric("exact_match", ExactMatchMetric())
    return Orchestrator.from_project_spec(
        _build_project(tmp_path),
        registry=registry,
        dataset_provider=DemoDatasetProvider(),
    )


def test_orchestrator_submit_resume_and_plan_are_benchmark_native(
    tmp_path: Path,
) -> None:
    orchestrator = _build_orchestrator(tmp_path)
    benchmark = _build_benchmark()

    manifest = orchestrator.plan(benchmark)
    handle = orchestrator.submit(benchmark, runtime=RuntimeContext())
    resumed = orchestrator.resume(handle.run_id)

    assert manifest.benchmark_spec is not None
    assert manifest.benchmark_spec.benchmark_id == "demo-benchmark"
    assert isinstance(handle, RunHandle)
    assert handle.status == "completed"
    assert isinstance(resumed, BenchmarkResult)
    assert resumed.benchmark_id == "demo-benchmark"
    assert resumed.slice_ids == ["qa"]
    stored_manifest = orchestrator._run_planning.manifest_repo.get_manifest(
        handle.run_id
    )
    assert stored_manifest is not None
    assert stored_manifest.source_kind == "benchmark"
    assert stored_manifest.benchmark_spec is not None
    assert stored_manifest.benchmark_spec.benchmark_id == "demo-benchmark"


def test_orchestrator_diff_specs_supports_benchmark_source_diffs(
    tmp_path: Path,
) -> None:
    orchestrator = _build_orchestrator(tmp_path)
    baseline = _build_benchmark()
    treatment = baseline.model_copy(
        update={
            "models": [
                *baseline.models,
                ModelSpec(model_id="demo-model-2", provider="demo"),
            ]
        }
    )

    diff = orchestrator.diff_specs(baseline, treatment)

    assert diff.source_kind == "benchmark"
    assert "models" in diff.changed_source_fields
    assert diff.added_trial_hashes
    assert diff.removed_trial_hashes == []


def test_orchestrator_diff_specs_rehashes_trials_for_query_and_render_fields(
    tmp_path: Path,
) -> None:
    orchestrator = _build_orchestrator(tmp_path)
    baseline = _build_benchmark()
    treatment_slice = baseline.slices[0].model_copy(
        update={
            "dataset_query": DatasetQuerySpec.subset(1, seed=11),
            "dimensions": {"source": "synthetic", "format": "cot"},
        }
    )
    treatment_prompt = baseline.prompt_variants[0].model_copy(
        update={"variables": {"tone": "formal"}}
    )
    treatment = baseline.model_copy(
        update={
            "slices": [treatment_slice],
            "prompt_variants": [treatment_prompt],
        }
    )

    diff = orchestrator.diff_specs(baseline, treatment)

    assert "slices" in diff.changed_source_fields
    assert "prompt_variants" in diff.changed_source_fields
    assert diff.added_trial_hashes
    assert diff.removed_trial_hashes


def test_benchmark_spec_rejects_duplicate_slice_ids() -> None:
    with pytest.raises(ValidationError, match="duplicate slice_id"):
        BenchmarkSpec(
            benchmark_id="demo-benchmark",
            models=[ModelSpec(model_id="demo-model", provider="demo")],
            slices=[
                SliceSpec(
                    slice_id="qa",
                    dataset=DatasetSpec(source=DatasetSource.MEMORY),
                    generation=GenerationSpec(),
                    scores=[ScoreSpec(name="default", metrics=["exact_match"])],
                ),
                SliceSpec(
                    slice_id="qa",
                    dataset=DatasetSpec(source=DatasetSource.MEMORY),
                    generation=GenerationSpec(),
                    scores=[ScoreSpec(name="default", metrics=["exact_match"])],
                ),
            ],
            prompt_variants=[
                PromptVariantSpec(
                    id="qa-default",
                    messages=[
                        PromptMessage(
                            role=PromptRole.USER,
                            content="Solve: {item.question}",
                        )
                    ],
                )
            ],
            inference_grid=InferenceGridSpec(
                params=[InferenceParamsSpec(max_tokens=16)]
            ),
        )


def test_benchmark_spec_rejects_duplicate_prompt_variant_ids() -> None:
    with pytest.raises(ValidationError, match="duplicate prompt variant"):
        BenchmarkSpec(
            benchmark_id="demo-benchmark",
            models=[ModelSpec(model_id="demo-model", provider="demo")],
            slices=[
                SliceSpec(
                    slice_id="qa",
                    dataset=DatasetSpec(source=DatasetSource.MEMORY),
                    generation=GenerationSpec(),
                    scores=[ScoreSpec(name="default", metrics=["exact_match"])],
                )
            ],
            prompt_variants=[
                PromptVariantSpec(
                    id="qa-default",
                    messages=[
                        PromptMessage(
                            role=PromptRole.USER,
                            content="Solve: {item.question}",
                        )
                    ],
                ),
                PromptVariantSpec(
                    id="qa-default",
                    messages=[
                        PromptMessage(
                            role=PromptRole.USER,
                            content="Score: {item.question}",
                        )
                    ],
                ),
            ],
            inference_grid=InferenceGridSpec(
                params=[InferenceParamsSpec(max_tokens=16)]
            ),
        )


def test_benchmark_spec_rejects_slices_with_unknown_prompt_variant_ids() -> None:
    with pytest.raises(ValidationError, match="missing-variant"):
        BenchmarkSpec(
            benchmark_id="demo-benchmark",
            models=[ModelSpec(model_id="demo-model", provider="demo")],
            slices=[
                SliceSpec(
                    slice_id="qa",
                    dataset=DatasetSpec(source=DatasetSource.MEMORY),
                    generation=GenerationSpec(),
                    prompt_variant_ids=["missing-variant"],
                    scores=[ScoreSpec(name="default", metrics=["exact_match"])],
                )
            ],
            prompt_variants=[
                PromptVariantSpec(
                    id="qa-default",
                    messages=[
                        PromptMessage(
                            role=PromptRole.USER,
                            content="Solve: {item.question}",
                        )
                    ],
                )
            ],
            inference_grid=InferenceGridSpec(
                params=[InferenceParamsSpec(max_tokens=16)]
            ),
        )
