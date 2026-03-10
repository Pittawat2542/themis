"""Show how repeated runs skip completed trials in the same storage root."""

from pathlib import Path

from themis import (
    DatasetSpec,
    ExecutionPolicySpec,
    ExperimentSpec,
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
from themis.records.evaluation import MetricScore
from themis.records.inference import InferenceRecord


class ResumeDatasetLoader:
    def load_task_items(self, task):
        del task
        return [
            {"item_id": "item-1", "question": "6 * 7", "answer": "42"},
            {"item_id": "item-2", "question": "8 * 8", "answer": "64"},
        ]


class CountingEngine:
    """Counts engine invocations so resume behavior is obvious."""

    def __init__(self) -> None:
        self.calls = 0

    def infer(self, trial, context, runtime):
        del trial, runtime
        self.calls += 1
        return InferenceResult(
            inference=InferenceRecord(
                spec_hash=f"inf_{context['item_id']}",
                raw_text=context["answer"],
            )
        )


class ExactMatchMetric:
    def score(self, trial, candidate, context):
        del trial
        actual = candidate.inference.raw_text if candidate.inference else ""
        return MetricScore(
            metric_id="exact_match", value=float(actual == context["answer"])
        )


def build_registry(engine: CountingEngine) -> PluginRegistry:
    registry = PluginRegistry()
    registry.register_inference_engine("demo", engine)
    registry.register_metric("exact_match", ExactMatchMetric())
    return registry


def build_project() -> ProjectSpec:
    return ProjectSpec(
        project_name="resume-demo",
        researcher_id="examples",
        global_seed=31,
        storage=StorageSpec(
            root_dir=str(Path(".cache/themis-examples/05-resume-run")),
            compression="none",
        ),
        execution_policy=ExecutionPolicySpec(),
    )


def build_experiment() -> ExperimentSpec:
    return ExperimentSpec(
        models=[ModelSpec(model_id="resume-model", provider="demo")],
        tasks=[
            TaskSpec(
                task_id="resume-math",
                dataset=DatasetSpec(source="memory"),
                default_metrics=["exact_match"],
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
    engine = CountingEngine()
    orchestrator = Orchestrator.from_project_spec(
        build_project(),
        registry=build_registry(engine),
        dataset_loader=ResumeDatasetLoader(),
    )
    experiment = build_experiment()

    first_result = orchestrator.run(experiment)
    first_calls = engine.calls
    second_result = orchestrator.run(experiment)
    second_calls = engine.calls

    print("First run trial hashes:", ", ".join(first_result.trial_hashes))
    print("Second run trial hashes:", ", ".join(second_result.trial_hashes))
    print("Engine calls after first run:", first_calls)
    print("Engine calls after second run:", second_calls)
    print("Second run reused cached trials:", second_calls == first_calls)


if __name__ == "__main__":
    main()
