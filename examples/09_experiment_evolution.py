"""Show how stored work is reused as an experiment grows over time."""

from __future__ import annotations

from pathlib import Path

from themis import (
    DatasetSpec,
    EvaluationSpec,
    ExecutionPolicySpec,
    ExperimentSpec,
    GenerationSpec,
    InferenceGridSpec,
    InferenceParamsSpec,
    ItemSamplingSpec,
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
from themis.records import InferenceRecord, MetricScore


class ArithmeticDatasetLoader:
    def load_task_items(self, task):
        del task
        return [
            {
                "item_id": "item-1",
                "question": "1 + 1",
                "answer": "2",
                "metadata": {"difficulty": "easy"},
            },
            {
                "item_id": "item-2",
                "question": "7 + 8",
                "answer": "15",
                "metadata": {"difficulty": "hard"},
            },
            {
                "item_id": "item-3",
                "question": "9 + 9",
                "answer": "18",
                "metadata": {"difficulty": "hard"},
            },
        ]


class DemoEngine:
    def __init__(self) -> None:
        self.calls = 0

    def infer(self, trial, context, runtime):
        del runtime
        self.calls += 1
        if trial.model.model_id == "baseline" and context["item_id"] == "item-3":
            answer = "wrong"
        else:
            answer = str(context["answer"])
        return InferenceResult(
            inference=InferenceRecord(
                spec_hash=(
                    f"inf_{trial.model.model_id}_{trial.prompt.id}_{trial.item_id}_"
                    f"{trial.params.spec_hash}"
                ),
                raw_text=answer,
            )
        )


class ExactMatchMetric:
    def score(self, trial, candidate, context):
        del trial
        actual = candidate.inference.raw_text if candidate.inference else ""
        return MetricScore(
            metric_id="exact_match",
            value=float(actual == str(context["answer"])),
        )


class AnswerLengthMetric:
    def score(self, trial, candidate, context):
        del trial, context
        actual = candidate.inference.raw_text if candidate.inference else ""
        return MetricScore(
            metric_id="answer_length",
            value=float(len(actual.strip())),
        )


def build_registry(engine: DemoEngine) -> PluginRegistry:
    registry = PluginRegistry()
    registry.register_inference_engine("demo", engine)
    registry.register_metric("exact_match", ExactMatchMetric())
    registry.register_metric("answer_length", AnswerLengthMetric())
    return registry


def build_project() -> ProjectSpec:
    return ProjectSpec(
        project_name="experiment-evolution",
        researcher_id="examples",
        global_seed=71,
        storage=StorageSpec(
            root_dir=str(Path(".cache/themis-examples/09-experiment-evolution")),
            compression="none",
        ),
        execution_policy=ExecutionPolicySpec(),
    )


def build_base_experiment() -> ExperimentSpec:
    return ExperimentSpec(
        models=[ModelSpec(model_id="baseline", provider="demo")],
        tasks=[
            TaskSpec(
                task_id="evolution-math",
                dataset=DatasetSpec(source="memory"),
                generation=GenerationSpec(),
                evaluations=[EvaluationSpec(name="default", metrics=["exact_match"])],
            )
        ],
        prompt_templates=[
            PromptTemplateSpec(
                id="zero-shot",
                messages=[
                    PromptMessage(role="user", content="Solve the arithmetic problem.")
                ],
            )
        ],
        inference_grid=InferenceGridSpec(
            params=[InferenceParamsSpec(max_tokens=32, temperature=0.0, seed=11)]
        ),
    )


def build_metric_expansion(base: ExperimentSpec) -> ExperimentSpec:
    task = base.tasks[0].model_copy(
        update={
            "evaluations": [
                EvaluationSpec(
                    name="default",
                    metrics=["exact_match", "answer_length"],
                )
            ]
        }
    )
    return base.model_copy(update={"tasks": [task]})


def build_matrix_expansion(base: ExperimentSpec) -> ExperimentSpec:
    return base.model_copy(
        update={
            "models": [
                *base.models,
                ModelSpec(model_id="candidate", provider="demo"),
            ],
            "prompt_templates": [
                *base.prompt_templates,
                PromptTemplateSpec(
                    id="few-shot",
                    messages=[
                        PromptMessage(role="user", content="Q: 2 + 2\nA: 4"),
                        PromptMessage(
                            role="user", content="Solve the next arithmetic problem."
                        ),
                    ],
                ),
            ],
            "inference_grid": InferenceGridSpec(
                params=[InferenceParamsSpec(max_tokens=32, seed=11)],
                overrides={"temperature": [0.0, 0.7]},
            ),
        }
    )


def build_hard_slice(base: ExperimentSpec) -> ExperimentSpec:
    return base.model_copy(
        update={
            "item_sampling": ItemSamplingSpec(
                metadata_filters={"difficulty": "hard"},
            )
        }
    )


def main() -> None:
    engine = DemoEngine()
    orchestrator = Orchestrator.from_project_spec(
        build_project(),
        registry=build_registry(engine),
        dataset_loader=ArithmeticDatasetLoader(),
    )

    base = build_base_experiment()
    base_result = orchestrator.run(base)
    base_calls = engine.calls
    print("Base run trials:", len(base_result.trial_hashes))

    metric_expanded = build_metric_expansion(base)
    metric_diff = orchestrator.diff_specs(base, metric_expanded)
    print("Metric expansion changed fields:", metric_diff.changed_experiment_fields)
    orchestrator.run(metric_expanded)
    metric_calls = engine.calls
    print("Metric expansion reused generation:", metric_calls == base_calls)

    matrix_expanded = build_matrix_expansion(metric_expanded)
    matrix_diff = orchestrator.diff_specs(metric_expanded, matrix_expanded)
    print("Matrix expansion added trials:", len(matrix_diff.added_trial_hashes))
    final_result = orchestrator.run(matrix_expanded)
    matrix_calls = engine.calls
    print("New generation calls after matrix expansion:", matrix_calls - metric_calls)
    print("Expanded run trials:", len(final_result.trial_hashes))

    hard_only = build_hard_slice(matrix_expanded)
    slice_diff = orchestrator.diff_specs(matrix_expanded, hard_only)
    print("Hard-slice changed fields:", slice_diff.changed_experiment_fields)
    print(
        "Hard-slice added/removed trials:",
        len(slice_diff.added_trial_hashes),
        len(slice_diff.removed_trial_hashes),
    )


if __name__ == "__main__":
    main()
