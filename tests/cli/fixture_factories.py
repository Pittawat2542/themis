from __future__ import annotations

from themis import (
    ExecutionPolicySpec,
    InferenceGridSpec,
    InferenceParamsSpec,
    ModelSpec,
    ProjectSpec,
    SqliteBlobStorageSpec,
)
from themis.types.enums import DatasetSource
from themis.specs.experiment import ExperimentSpec, PromptTemplateSpec
from themis.specs.foundational import DatasetSpec, EvaluationSpec, TaskSpec


def build_config_bundle() -> dict[str, object]:
    return {
        "project": ProjectSpec(
            project_name="factory-report-demo",
            researcher_id="tests",
            global_seed=5,
            storage=SqliteBlobStorageSpec(root_dir=".cache/factory-report"),
            execution_policy=ExecutionPolicySpec(),
        ),
        "experiment": ExperimentSpec(
            models=[ModelSpec(model_id="demo-model", provider="demo")],
            tasks=[
                TaskSpec(
                    task_id="math",
                    dataset=DatasetSpec(source=DatasetSource.MEMORY),
                    evaluations=[
                        EvaluationSpec(name="default", metrics=["exact_match"])
                    ],
                )
            ],
            prompt_templates=[PromptTemplateSpec(id="baseline", messages=[])],
            inference_grid=InferenceGridSpec(params=[InferenceParamsSpec()]),
        ),
    }
