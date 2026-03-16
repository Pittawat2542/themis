"""Load shared execution policy from a TOML project file."""

from pathlib import Path
import textwrap

from themis import (
    DatasetSpec,
    EvaluationSpec,
    ExperimentSpec,
    GenerationSpec,
    InferenceGridSpec,
    InferenceParamsSpec,
    ModelSpec,
    Orchestrator,
    PluginRegistry,
    PromptMessage,
    PromptTemplateSpec,
    TaskSpec,
)
from themis.contracts.protocols import InferenceResult
from themis.records import InferenceRecord, MetricScore
from themis.types.enums import PromptRole, DatasetSource


class GreetingDatasetLoader:
    def load_task_items(self, task):
        del task
        return [
            {"item_id": "item-1", "name": "Ada", "expected": "Hello, Ada!"},
            {"item_id": "item-2", "name": "Grace", "expected": "Hello, Grace!"},
        ]


class GreetingEngine:
    def infer(self, trial, context, runtime):
        del trial, runtime
        return InferenceResult(
            inference=InferenceRecord(
                spec_hash=f"inf_{context['item_id']}",
                raw_text=f"Hello, {context['name']}!",
            )
        )


class GreetingMetric:
    def score(self, trial, candidate, context):
        del trial
        actual = candidate.inference.raw_text if candidate.inference else ""
        return MetricScore(
            metric_id="matches_expected",
            value=float(actual == context["expected"]),
            details={"actual": actual, "expected": context["expected"]},
        )


def write_project_file(project_dir: Path) -> Path:
    project_dir.mkdir(parents=True, exist_ok=True)
    project_path = project_dir / "project.toml"
    project_path.write_text(
        textwrap.dedent(
            """
            project_name = "project-file-demo"
            researcher_id = "examples"
            global_seed = 11

            [storage]
            backend = "sqlite_blob"
            root_dir = ".cache/themis-examples/02-project-file"
            store_item_payloads = true
            compression = "none"

            [execution_policy]
            max_retries = 2
            retry_backoff_factor = 1.5
            circuit_breaker_threshold = 4
            """
        ).strip()
        + "\n"
    )
    return project_path


def build_registry() -> PluginRegistry:
    registry = PluginRegistry()
    registry.register_inference_engine("demo", GreetingEngine())
    registry.register_metric("matches_expected", GreetingMetric())
    return registry


def build_experiment() -> ExperimentSpec:
    return ExperimentSpec(
        models=[ModelSpec(model_id="greeting-model", provider="demo")],
        tasks=[
            TaskSpec(
                task_id="greetings",
                dataset=DatasetSpec(source=DatasetSource.MEMORY),
                generation=GenerationSpec(),
                evaluations=[
                    EvaluationSpec(name="default", metrics=["matches_expected"])
                ],
            )
        ],
        prompt_templates=[
            PromptTemplateSpec(
                id="hello",
                messages=[
                    PromptMessage(role=PromptRole.USER, content="Greet the user.")
                ],
            )
        ],
        inference_grid=InferenceGridSpec(params=[InferenceParamsSpec(max_tokens=32)]),
    )


def _format_display_path(path: Path) -> str:
    return path.as_posix()


def main() -> None:
    project_path = write_project_file(
        Path(".cache/themis-examples/02-project-file-config")
    )
    orchestrator = Orchestrator.from_project_file(
        str(project_path),
        registry=build_registry(),
        dataset_loader=GreetingDatasetLoader(),
    )
    result = orchestrator.run(build_experiment())

    print("Loaded project file:", _format_display_path(project_path))
    print("Trial hashes:", ", ".join(result.trial_hashes))


if __name__ == "__main__":
    main()
