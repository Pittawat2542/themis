"""Run generation locally, score externally, and import benchmark results back."""

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
from themis.orchestration.run_manifest import EvaluationBundleItem, EvaluationWorkBundle
from themis.records import (
    CandidateRecord,
    EvaluationRecord,
    InferenceRecord,
    MetricScore,
    TrialRecord,
)
from themis.specs import DatasetSpec, GenerationSpec
from themis.specs.experiment import TrialSpec
from themis.types.enums import CompressionCodec, DatasetSource, PromptRole


class ArithmeticDatasetProvider:
    def scan(self, slice_spec, query):
        del slice_spec, query
        return [
            {"item_id": "item-1", "question": "1 + 1", "answer": "2"},
            {"item_id": "item-2", "question": "2 + 2", "answer": "4"},
        ]


class DemoEngine:
    def infer(self, trial, context, runtime):
        del runtime
        answer = context["answer"] if trial.model.model_id == "candidate" else "wrong"
        return InferenceResult(
            inference=InferenceRecord(
                spec_hash=f"inf_{trial.model.model_id}_{trial.item_id}",
                raw_text=answer,
            )
        )


class PlaceholderMetric:
    def score(self, trial, candidate, context):
        del trial, candidate, context
        return MetricScore(metric_id="external_exact_match", value=0.0)


def _build_external_records(bundle: EvaluationWorkBundle) -> list[TrialRecord]:
    by_trial: dict[str, list[EvaluationBundleItem]] = {}
    trial_specs: dict[str, TrialSpec] = {}
    for item in bundle.items:
        by_trial.setdefault(item.trial_hash, []).append(item)
        trial_specs[item.trial_hash] = item.trial_spec

    records: list[TrialRecord] = []
    for trial_hash, items in sorted(by_trial.items()):
        candidates: list[CandidateRecord] = []
        for item in items:
            expected = str(item.dataset_context["answer"])
            actual = (
                item.candidate.inference.raw_text if item.candidate.inference else ""
            )
            candidates.append(
                CandidateRecord(
                    spec_hash=item.candidate_id,
                    candidate_id=item.candidate_id,
                    sample_index=item.candidate_index,
                    evaluation=EvaluationRecord(
                        spec_hash=f"eval_{item.candidate_id}",
                        metric_scores=[
                            MetricScore(
                                metric_id="external_exact_match",
                                value=float(actual == expected),
                            )
                        ],
                    ),
                )
            )
        records.append(
            TrialRecord(
                spec_hash=trial_hash,
                trial_spec=trial_specs[trial_hash],
                candidates=candidates,
            )
        )
    return records


def main() -> None:
    registry = PluginRegistry()
    registry.register_inference_engine("demo", DemoEngine())
    registry.register_metric("external_exact_match", PlaceholderMetric())

    project = ProjectSpec(
        project_name="external-stage-benchmark",
        researcher_id="examples",
        global_seed=61,
        storage=StorageSpec(
            root_dir=str(
                Path(".cache/themis-examples/08-external-stage-handoff-benchmark-first")
            ),
            compression=CompressionCodec.NONE,
        ),
        execution_policy=ExecutionPolicySpec(),
    )
    benchmark = BenchmarkSpec(
        benchmark_id="external-eval",
        models=[
            ModelSpec(model_id="baseline", provider="demo"),
            ModelSpec(model_id="candidate", provider="demo"),
        ],
        slices=[
            SliceSpec(
                slice_id="qa",
                dataset=DatasetSpec(source=DatasetSource.MEMORY),
                generation=GenerationSpec(),
                prompt_variant_ids=["baseline"],
                scores=[ScoreSpec(name="external", metrics=["external_exact_match"])],
            )
        ],
        prompt_variants=[
            PromptVariantSpec(
                id="baseline",
                family="qa",
                messages=[
                    PromptMessage(
                        role=PromptRole.USER, content="Solve the arithmetic problem."
                    )
                ],
            )
        ],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec(max_tokens=32)]),
    )

    orchestrator = Orchestrator.from_project_spec(
        project,
        registry=registry,
        dataset_provider=ArithmeticDatasetProvider(),
    )
    generated = orchestrator.generate(benchmark)
    assert generated.trial_hashes
    bundle = orchestrator.export_evaluation_bundle(benchmark)
    result = orchestrator.import_evaluation_results(
        bundle, _build_external_records(bundle)
    )

    print(result.aggregate(group_by=["model_id", "slice_id", "metric_id"]))


if __name__ == "__main__":
    main()
