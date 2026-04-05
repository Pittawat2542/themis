from __future__ import annotations

import pytest

from themis.catalog import load
from themis.core.components import component_ref_from_value
from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.contexts import EvalScoreContext, SelectContext
from themis.core.experiment import Experiment
from themis.core.models import Case, Dataset, GenerationResult, ParsedOutput
from themis.core.prompts import FewShotExample, PromptSpec
from themis.core.results import RunStatus
from themis.core.stores import InMemoryRunStore
from themis.core.subjects import CandidateSetSubject
from themis.core.workflows import JudgeResponse


class ChoosingJudgeModel:
    version = "1.0"

    def __init__(self, component_id: str, response: str) -> None:
        self.component_id = component_id
        self.response = response

    def fingerprint(self) -> str:
        return f"{self.component_id}-fingerprint"

    async def judge(self, prompt: str, *, seed: int | None = None) -> JudgeResponse:
        del prompt, seed
        return JudgeResponse(
            judge_model_id=self.component_id,
            judge_model_version=self.version,
            judge_model_fingerprint=self.fingerprint(),
            raw_response=self.response,
        )


@pytest.mark.asyncio
async def test_catalog_builtin_best_of_n_uses_judge_models_to_pick_winner() -> None:
    selector = load("builtin/best_of_n")
    candidates = [
        GenerationResult(
            candidate_id="case-1-candidate-0", final_output={"answer": "4"}
        ),
        GenerationResult(
            candidate_id="case-1-candidate-1", final_output={"answer": "5"}
        ),
    ]

    selected = await selector.select(
        candidates,
        SelectContext(
            run_id="run-1",
            case_id="case-1",
            candidate_ids=[candidate.candidate_id for candidate in candidates],
            seed=7,
            judge_models=[ChoosingJudgeModel("judge/a", "B")],
        ),
    )

    assert [candidate.candidate_id for candidate in selected] == ["case-1-candidate-1"]
    assert selected[0].final_output == {"answer": "5"}


def test_catalog_builtin_judge_metrics_build_expected_workflows() -> None:
    llm_rubric = load("builtin/llm_rubric")
    panel = load("builtin/panel_of_judges")
    majority = load("builtin/majority_vote_judge")
    pairwise = load("builtin/pairwise_judge")
    candidate = GenerationResult(
        candidate_id="case-1-reduced", final_output={"answer": "4"}
    )
    pair_a = GenerationResult(
        candidate_id="case-1-candidate-0", final_output={"answer": "4"}
    )
    pair_b = GenerationResult(
        candidate_id="case-1-candidate-1", final_output={"answer": "5"}
    )
    ctx = EvalScoreContext(
        run_id="run-1",
        case=Case(
            case_id="case-1", input={"question": "2+2"}, expected_output={"answer": "4"}
        ),
        parsed_output=ParsedOutput(value={"answer": "4"}),
        judge_model_refs=[
            component_ref_from_value("builtin/demo_judge"),
            component_ref_from_value("builtin/demo_judge"),
        ],
        prompt_spec=PromptSpec(
            instructions="Use the few-shot examples before grading.",
            few_shot_examples=[
                FewShotExample(input={"answer": "3"}, output="FAIL"),
            ],
        ),
        eval_workflow_config={"rubric": "grade factual accuracy"},
    )

    llm_workflow = llm_rubric.build_workflow(
        CandidateSetSubject(candidates=[candidate]), ctx
    )
    panel_workflow = panel.build_workflow(
        CandidateSetSubject(candidates=[candidate]), ctx
    )
    majority_workflow = majority.build_workflow(
        CandidateSetSubject(candidates=[candidate]), ctx
    )
    pairwise_workflow = pairwise.build_workflow(
        CandidateSetSubject(candidates=[pair_a, pair_b]), ctx
    )

    assert len(llm_workflow.judge_calls()) == 2
    assert (
        "grade factual accuracy"
        in llm_workflow.render_prompt(
            llm_workflow.judge_calls()[0],
            CandidateSetSubject(candidates=[candidate]),
            ctx,
        ).content
    )
    rendered = llm_workflow.render_prompt(
        llm_workflow.judge_calls()[0],
        CandidateSetSubject(candidates=[candidate]),
        ctx,
    ).content
    assert "Use the few-shot examples before grading." in rendered
    assert "Example 1 input:" in rendered
    assert len(panel_workflow.judge_calls()) == 2
    assert len(majority_workflow.judge_calls()) == 2
    assert pairwise_workflow.judge_calls()[0].candidate_indices == [0, 1]


def test_catalog_builtin_judge_metrics_run_end_to_end_through_experiment() -> None:
    experiment = Experiment(
        generation=GenerationConfig(
            generator="builtin/demo_generator",
            candidate_policy={"num_samples": 2},
            selector="builtin/best_of_n",
            reducer="builtin/majority_vote",
        ),
        evaluation=EvaluationConfig(
            metrics=[
                "builtin/llm_rubric",
                "builtin/pairwise_judge",
                "builtin/panel_of_judges",
                "builtin/majority_vote_judge",
            ],
            parsers=["builtin/json_identity"],
            judge_models=["builtin/demo_judge", "builtin/demo_judge"],
            workflow_overrides={"rubric": "pass if the answer is correct"},
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
        seeds=[7, 11],
    )
    store = InMemoryRunStore()

    result = experiment.run(store=store)

    assert result.status is RunStatus.COMPLETED
    assert sorted(score.metric_id for score in result.cases[0].scores) == [
        "builtin/llm_rubric",
        "builtin/majority_vote_judge",
        "builtin/pairwise_judge",
        "builtin/panel_of_judges",
    ]
