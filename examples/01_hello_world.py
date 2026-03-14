"""Minimal Themis workflow that mirrors the Quick Start guide."""

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
from themis.records import InferenceRecord, MetricScore


class ArithmeticDatasetLoader:
    """Provides two in-memory arithmetic items for the demo task."""

    def load_task_items(self, task):
        del task
        return [
            {"item_id": "item-1", "question": "2 + 2", "answer": "4"},
            {"item_id": "item-2", "question": "6 * 7", "answer": "42"},
        ]


class DemoEngine:
    """Small fake engine so the example runs without external providers."""

    def infer(self, trial, context, runtime):
        del runtime
        answer = "4" if context["question"] == "2 + 2" else "42"
        return InferenceResult(
            inference=InferenceRecord(
                spec_hash=f"inference_{trial.item_id}",
                raw_text=answer,
                latency_ms=2,
            )
        )


class ExactMatchMetric:
    """Scores the engine output against the dataset answer."""

    def score(self, trial, candidate, context):
        del trial
        actual = candidate.inference.raw_text if candidate.inference else ""
        expected = str(context["answer"])
        return MetricScore(
            metric_id="exact_match",
            value=float(actual.strip() == expected),
            details={"actual": actual, "expected": expected},
        )


def build_registry() -> PluginRegistry:
    registry = PluginRegistry()
    registry.register_inference_engine("demo", DemoEngine())
    registry.register_metric("exact_match", ExactMatchMetric())
    return registry


def build_project() -> ProjectSpec:
    return ProjectSpec(
        project_name="hello-world",
        researcher_id="examples",
        global_seed=7,
        storage=StorageSpec(
            root_dir=str(Path(".cache/themis-examples/01-hello-world")),
            compression="none",
        ),
        execution_policy=ExecutionPolicySpec(),
    )


def build_experiment() -> ExperimentSpec:
    return ExperimentSpec(
        models=[ModelSpec(model_id="demo-model", provider="demo")],
        tasks=[
            TaskSpec(
                task_id="arithmetic",
                dataset=DatasetSpec(source="memory"),
                generation=GenerationSpec(),
                evaluations=[EvaluationSpec(name="default", metrics=["exact_match"])],
            )
        ],
        prompt_templates=[
            PromptTemplateSpec(
                id="baseline",
                messages=[
                    PromptMessage(role="user", content="Solve the arithmetic problem.")
                ],
            )
        ],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec(max_tokens=32)]),
    )


def main() -> None:
    orchestrator = Orchestrator.from_project_spec(
        build_project(),
        registry=build_registry(),
        dataset_loader=ArithmeticDatasetLoader(),
    )
    result = orchestrator.run(build_experiment())

    print(
        "Stored SQLite database:",
        ".cache/themis-examples/01-hello-world/themis.sqlite3",
    )
    for trial in result.iter_trials():
        score = trial.candidates[0].evaluation.aggregate_scores["exact_match"]
        print(f"{trial.trial_spec.item_id}: exact_match={score:.1f}")


if __name__ == "__main__":
    main()
