from __future__ import annotations

from themis import Experiment
from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.models import Case, Dataset
from themis.core.prompts import FewShotExample, PromptSpec


def test_prompt_specs_change_run_identity_for_generation_and_evaluation() -> None:
    base = Experiment(
        generation=GenerationConfig(
            generator="builtin/demo_generator",
            candidate_policy={"num_samples": 1},
            reducer="builtin/majority_vote",
            prompt_spec=PromptSpec(instructions="Answer directly."),
        ),
        evaluation=EvaluationConfig(
            metrics=["builtin/llm_rubric"],
            parsers=["builtin/json_identity"],
            judge_models=["builtin/demo_judge"],
            prompt_spec=PromptSpec(
                prefix="Grade carefully.",
                few_shot_examples=[
                    FewShotExample(input={"answer": "4"}, output="PASS")
                ],
            ),
        ),
        storage=StorageConfig(store="memory"),
        datasets=[
            Dataset(
                dataset_id="dataset-1",
                cases=[
                    Case(
                        case_id="case-1",
                        input={"question": "2+2"},
                        expected_output={"answer": "4"},
                    )
                ],
            )
        ],
        seeds=[7],
    )
    changed_generation = base.model_copy(
        update={
            "generation": base.generation.model_copy(
                update={"prompt_spec": PromptSpec(instructions="Think step by step.")}
            )
        }
    )
    changed_evaluation = base.model_copy(
        update={
            "evaluation": base.evaluation.model_copy(
                update={"prompt_spec": PromptSpec(prefix="Use a stricter rubric.")}
            )
        }
    )

    assert base.compile().run_id != changed_generation.compile().run_id
    assert base.compile().run_id != changed_evaluation.compile().run_id
