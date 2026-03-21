"""run_benchmark_iter() and estimate() with trial matrix — streaming and planning.

Demonstrates:
- Orchestrator.run_benchmark_iter() to process trial records as they complete
- Orchestrator.estimate() with trial_count and trial_matrix breakdown
- RunDiff.has_invalidated_resume_work to detect spec changes that lose completed work
"""

from pathlib import Path

from themis import (
    BenchmarkSpec,
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
from themis.records import InferenceRecord, MetricScore
from themis.specs import DatasetSpec, GenerationSpec
from themis.types.enums import CompressionCodec, DatasetSource, PromptRole


class ArithmeticProvider:
    def scan(self, slice_spec, query):
        del slice_spec, query
        return [
            {"item_id": "item-1", "question": "2 + 2", "answer": "4"},
            {"item_id": "item-2", "question": "3 + 3", "answer": "6"},
        ]


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


def _make_benchmark(benchmark_id: str, models: list[ModelSpec]) -> BenchmarkSpec:
    return BenchmarkSpec(
        benchmark_id=benchmark_id,
        models=models,
        slices=[
            SliceSpec(
                slice_id="arithmetic",
                dataset=DatasetSpec(source=DatasetSource.MEMORY),
                prompt_variant_ids=["qa-default"],
                generation=GenerationSpec(),
                scores=[ScoreSpec(name="default", metrics=["exact_match"])],
            )
        ],
        prompt_variants=[
            PromptVariantSpec(
                id="qa-default",
                messages=[
                    PromptMessage(role=PromptRole.USER, content="{item.question}")
                ],
            )
        ],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec()]),
    )


def main() -> None:
    registry = PluginRegistry()
    registry.register_inference_engine("demo", DemoEngine())
    registry.register_metric("exact_match", ExactMatchMetric())

    project = ProjectSpec(
        project_name="iter-and-estimate",
        researcher_id="examples",
        global_seed=7,
        storage=StorageSpec(
            root_dir=str(Path(".cache/themis-examples/12-iter-and-estimate")),
            compression=CompressionCodec.NONE,
        ),
        execution_policy=ExecutionPolicySpec(),
    )

    orchestrator = Orchestrator.from_project_spec(
        project,
        registry=registry,
        dataset_provider=ArithmeticProvider(),
    )

    benchmark = _make_benchmark(
        "iter-demo",
        [ModelSpec(model_id="demo-model", provider="demo")],
    )

    # ── 1. Estimate before running ────────────────────────────────────────────
    # trial_count and trial_matrix surface the multiplicative expansion so you
    # know what you're committing to before execution.
    estimate = orchestrator.estimate(benchmark)
    print("=== Pre-run estimate ===")
    print(f"  trials:           {estimate.trial_count}")
    print(f"  trial matrix:     {estimate.trial_matrix}")
    print(f"  total work items: {estimate.total_work_items}")
    for note in estimate.notes:
        print(f"  note: {note}")
    print()

    # ── 2. run_benchmark_iter() ───────────────────────────────────────────────
    # Process each TrialRecord as it completes instead of waiting for the full
    # matrix.  Useful for logging progress, early stopping, or streaming results
    # to an external dashboard.
    print("=== Streaming results ===")
    for trial_record in orchestrator.run_benchmark_iter(benchmark):
        scores = [
            f"{e.scores[0].value:.1f}"
            for c in trial_record.candidates
            if (e := c.evaluation) and e.scores
        ]
        print(f"  trial {trial_record.trial_hash[:12]} → scores: {scores}")
    print()

    # ── 3. RunDiff.has_invalidated_resume_work ────────────────────────────────
    # When you add a second model, the diff shows new trial hashes were added
    # but none were removed — so existing completed work is safe.
    extended = _make_benchmark(
        "iter-demo",
        [
            ModelSpec(model_id="demo-model", provider="demo"),
            ModelSpec(model_id="demo-model-v2", provider="demo"),
        ],
    )
    diff = orchestrator.diff_specs(benchmark, extended)
    print("=== Spec diff ===")
    print(f"  added trials:   {len(diff.added_trial_hashes)}")
    print(f"  removed trials: {len(diff.removed_trial_hashes)}")
    print(f"  invalidates resume: {diff.has_invalidated_resume_work}")


if __name__ == "__main__":
    main()
