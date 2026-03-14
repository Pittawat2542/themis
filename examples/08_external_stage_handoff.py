"""Run generation locally, score externally, then continue with Themis analysis.

Requires the optional stats extra:

    uv add "themis-eval[stats]"
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from themis import (
    DatasetSpec,
    EvaluationSpec,
    ExecutionPolicySpec,
    ExperimentSpec,
    GenerationSpec,
    InferenceGridSpec,
    InferenceParamsSpec,
    ModelSpec,
    Orchestrator,
    PluginRegistry,
    ProjectSpec,
    PromptMessage,
    PromptTemplateSpec,
    StorageSpec,
    TaskSpec,
)
from themis.contracts.protocols import InferenceResult
from themis.records import (
    CandidateRecord,
    EvaluationRecord,
    InferenceRecord,
    MetricScore,
    TrialRecord,
)


class ArithmeticDatasetLoader:
    def load_task_items(self, task):
        del task
        return [
            {"item_id": "item-1", "question": "1 + 1", "answer": "2"},
            {"item_id": "item-2", "question": "2 + 2", "answer": "4"},
            {"item_id": "item-3", "question": "3 + 3", "answer": "6"},
            {"item_id": "item-4", "question": "4 + 4", "answer": "8"},
        ]


class ComparisonEngine:
    """Makes the candidate model stronger than the baseline."""

    def infer(self, trial, context, runtime):
        del runtime
        if trial.model.model_id == "baseline" and context["item_id"] in {
            "item-2",
            "item-4",
        }:
            answer = "wrong"
        else:
            answer = context["answer"]
        return InferenceResult(
            inference=InferenceRecord(
                spec_hash=f"inf_{trial.model.model_id}_{trial.item_id}",
                raw_text=answer,
            )
        )


class ExternalExactMatchMetric:
    """Placeholder metric used only for planning the evaluation overlay."""

    def score(self, trial, candidate, context):
        del trial, candidate, context
        return MetricScore(metric_id="external_exact_match", value=0.0)


def build_registry() -> PluginRegistry:
    registry = PluginRegistry()
    registry.register_inference_engine("demo", ComparisonEngine())
    registry.register_metric("external_exact_match", ExternalExactMatchMetric())
    return registry


def build_project() -> ProjectSpec:
    return ProjectSpec(
        project_name="external-stage-handoff",
        researcher_id="examples",
        global_seed=61,
        storage=StorageSpec(
            root_dir=str(Path(".cache/themis-examples/08-external-stage-handoff")),
            compression="none",
        ),
        execution_policy=ExecutionPolicySpec(),
    )


def build_experiment() -> ExperimentSpec:
    return ExperimentSpec(
        models=[
            ModelSpec(model_id="baseline", provider="demo"),
            ModelSpec(model_id="candidate", provider="demo"),
        ],
        tasks=[
            TaskSpec(
                task_id="external-eval-math",
                dataset=DatasetSpec(source="memory"),
                generation=GenerationSpec(),
                evaluations=[
                    EvaluationSpec(
                        name="external",
                        metrics=["external_exact_match"],
                    )
                ],
            )
        ],
        prompt_templates=[
            PromptTemplateSpec(
                id="baseline",
                messages=[
                    PromptMessage(role="user", content="Answer the arithmetic problem.")
                ],
            )
        ],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec(max_tokens=32)]),
    )


def build_external_evaluation_records(bundle) -> list[TrialRecord]:
    items_by_trial = defaultdict(list)
    trial_specs = {}
    for item in bundle.items:
        items_by_trial[item.trial_hash].append(item)
        trial_specs[item.trial_hash] = item.trial_spec

    records: list[TrialRecord] = []
    for trial_hash, items in sorted(items_by_trial.items()):
        candidates = []
        for item in items:
            expected = str(item.dataset_context["answer"])
            actual = (
                item.candidate.inference.raw_text if item.candidate.inference else ""
            )
            score = float(actual == expected)
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
                                value=score,
                                details={"expected": expected, "actual": actual},
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
    orchestrator = Orchestrator.from_project_spec(
        build_project(),
        registry=build_registry(),
        dataset_loader=ArithmeticDatasetLoader(),
    )
    experiment = build_experiment()

    generated = orchestrator.generate(experiment)
    print("Generation-only trial count:", len(generated.trial_hashes))

    evaluation_bundle = orchestrator.export_evaluation_bundle(experiment)
    print("Pending external evaluation items:", len(evaluation_bundle.items))

    external_records = build_external_evaluation_records(evaluation_bundle)
    result = orchestrator.import_evaluation_results(evaluation_bundle, external_records)
    evaluation_result = result.for_evaluation(result.evaluation_hashes[0])

    comparison = evaluation_result.compare(
        metric_id="external_exact_match",
        baseline_model_id="baseline",
        treatment_model_id="candidate",
        p_value_correction="holm",
    )
    leaderboard = evaluation_result.leaderboard(metric_id="external_exact_match")
    export_path = Path(
        ".cache/themis-examples/08-external-stage-handoff/external-result.json"
    )
    evaluation_result.export_json(str(export_path))

    row = comparison.rows[0]
    print("Comparison delta_mean:", round(row.delta_mean, 3))
    print("Leaderboard rows:", len(leaderboard))
    print("Exported overlay JSON:", export_path)


if __name__ == "__main__":
    main()
