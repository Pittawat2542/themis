from __future__ import annotations

import pytest

from themis.core.config import EvaluationConfig, GenerationConfig, StorageConfig
from themis.core.contexts import EvalScoreContext, GenerateContext, ParseContext, ReduceContext, ScoreContext
from themis.core.experiment import Experiment
from themis.core.models import Case, Dataset, GenerationResult, ParsedOutput, ReducedCandidate, Score
from themis.core.results import RunStatus
from themis.core.stores.memory import InMemoryRunStore
from themis.core.workflows import AggregationResult, JudgeCall, JudgeResponse, ParsedJudgment, RenderedJudgePrompt


class FailingGenerator:
    component_id = "generator/failing"
    version = "1.0"

    def fingerprint(self) -> str:
        return "generator-failing"

    async def generate(self, case: Case, ctx: GenerateContext) -> GenerationResult:
        del case, ctx
        raise TimeoutError("provider timeout")


class FailingReducer:
    component_id = "reducer/failing"
    version = "1.0"

    def fingerprint(self) -> str:
        return "reducer-failing"

    async def reduce(self, candidates: list[GenerationResult], ctx: ReduceContext) -> ReducedCandidate:
        del candidates, ctx
        raise RuntimeError("reducer failed")


class FailingParser:
    component_id = "parser/failing"
    version = "1.0"

    def fingerprint(self) -> str:
        return "parser-failing"

    def parse(self, candidate: ReducedCandidate, ctx: ParseContext) -> ParsedOutput:
        del candidate, ctx
        raise ValueError("parser failed")


class FailingMetric:
    component_id = "metric/failing"
    version = "1.0"

    def fingerprint(self) -> str:
        return "metric-failing"

    def score(self, parsed: ParsedOutput, case: Case, ctx: ScoreContext) -> Score:
        del parsed, case, ctx
        raise RuntimeError("metric failed")


class HappyGenerator:
    component_id = "generator/happy"
    version = "1.0"

    def fingerprint(self) -> str:
        return "generator-happy"

    async def generate(self, case: Case, ctx: GenerateContext) -> GenerationResult:
        return GenerationResult(candidate_id=f"{case.case_id}-candidate-{ctx.seed}", final_output=case.expected_output)


class HappyReducer:
    component_id = "reducer/happy"
    version = "1.0"

    def fingerprint(self) -> str:
        return "reducer-happy"

    async def reduce(self, candidates: list[GenerationResult], ctx: ReduceContext) -> ReducedCandidate:
        return ReducedCandidate(
            candidate_id=f"{ctx.case_id}-reduced",
            source_candidate_ids=[candidate.candidate_id for candidate in candidates],
            final_output=candidates[0].final_output,
        )


class HappyParser:
    component_id = "parser/happy"
    version = "1.0"

    def fingerprint(self) -> str:
        return "parser-happy"

    def parse(self, candidate: ReducedCandidate, ctx: ParseContext) -> ParsedOutput:
        del ctx
        return ParsedOutput(value=candidate.final_output, format="json")


class PartialJudgeModel:
    version = "1.0"

    def __init__(self, component_id: str, *, fail: bool = False) -> None:
        self.component_id = component_id
        self.fail = fail

    def fingerprint(self) -> str:
        return f"{self.component_id}-fingerprint"

    async def judge(self, prompt: str, *, seed: int | None = None) -> JudgeResponse:
        del prompt, seed
        if self.fail:
            raise TimeoutError("judge timeout")
        return JudgeResponse(
            judge_model_id=self.component_id,
            judge_model_version=self.version,
            judge_model_fingerprint=self.fingerprint(),
            raw_response="pass",
        )


class PartialWorkflow:
    component_id = "workflow/partial"
    version = "1.0"

    def fingerprint(self) -> str:
        return "workflow-partial"

    def judge_calls(self) -> list[JudgeCall]:
        return [
            JudgeCall(call_id="call-ok", judge_model_id="judge/ok"),
            JudgeCall(call_id="call-fail", judge_model_id="judge/fail"),
        ]

    def render_prompt(self, call: JudgeCall, subject, ctx: EvalScoreContext) -> RenderedJudgePrompt:
        del subject, ctx
        return RenderedJudgePrompt(prompt_id=f"prompt-{call.call_id}", content=call.call_id)

    def parse_judgment(self, call: JudgeCall, response: JudgeResponse, ctx: EvalScoreContext) -> ParsedJudgment:
        del call, ctx
        return ParsedJudgment(label=response.raw_response, score=1.0)

    def score_judgment(self, call: JudgeCall, judgment: ParsedJudgment, ctx: EvalScoreContext) -> Score | None:
        del call, ctx
        return Score(metric_id="metric/partial", value=float(judgment.score or 0.0))

    def aggregate(
        self,
        judgments: list[ParsedJudgment],
        scores: list[Score],
        ctx: EvalScoreContext,
    ) -> AggregationResult | None:
        del judgments, ctx
        if not scores:
            return None
        return AggregationResult(method="mean", value=sum(score.value for score in scores) / len(scores))


class PartialMetric:
    component_id = "metric/partial"
    version = "1.0"
    metric_family = "llm"

    def __init__(self) -> None:
        self.calls = 0

    def fingerprint(self) -> str:
        return "metric-partial"

    def build_workflow(self, subject, ctx: EvalScoreContext):
        del subject, ctx
        self.calls += 1
        return PartialWorkflow()


class FlakyStore(InMemoryRunStore):
    def __init__(self) -> None:
        super().__init__()
        self.fail_next_persist = True

    def persist_event(self, event) -> None:
        if self.fail_next_persist:
            self.fail_next_persist = False
            raise RuntimeError("temporary store outage")
        super().persist_event(event)


def _base_dataset() -> list[Dataset]:
    return [
        Dataset(
            dataset_id="dataset-1",
            cases=[Case(case_id="case-1", input={"question": "2+2"}, expected_output={"answer": "4"})],
        )
    ]


@pytest.mark.parametrize(
    ("generator", "reducer", "parser", "metric"),
    [
        (FailingGenerator(), HappyReducer(), HappyParser(), FailingMetric()),
        (HappyGenerator(), FailingReducer(), HappyParser(), FailingMetric()),
        (HappyGenerator(), HappyReducer(), FailingParser(), FailingMetric()),
        (HappyGenerator(), HappyReducer(), HappyParser(), FailingMetric()),
    ],
)
def test_failure_modes_cover_stage_failures(generator, reducer, parser, metric) -> None:
    experiment = Experiment(
        generation=GenerationConfig(generator=generator, candidate_policy={"num_samples": 1}, reducer=reducer),
        evaluation=EvaluationConfig(metrics=[metric], parsers=[parser]),
        storage=StorageConfig(store="memory"),
        datasets=_base_dataset(),
        seeds=[7],
    )

    result = experiment.run()

    assert result.status is RunStatus.PARTIAL_FAILURE


def test_failure_modes_persist_partial_workflow_failures() -> None:
    metric = PartialMetric()
    experiment = Experiment(
        generation=GenerationConfig(generator=HappyGenerator(), candidate_policy={"num_samples": 1}, reducer=HappyReducer()),
        evaluation=EvaluationConfig(
            metrics=[metric],
            parsers=[HappyParser()],
            judge_models=[PartialJudgeModel("judge/ok"), PartialJudgeModel("judge/fail", fail=True)],
        ),
        storage=StorageConfig(store="memory"),
        datasets=_base_dataset(),
        seeds=[7],
    )

    result = experiment.run()

    assert result.status is RunStatus.PARTIAL_FAILURE
    execution = result.cases[0].evaluation_executions[0]
    assert execution.status == "partial_failure"
    assert execution.failures[0].error_message == "judge timeout"
    assert execution.aggregation_output is not None
    assert result.cases[0].scores[0].metric_id == "metric/partial"


def test_failure_modes_resume_interrupted_partial_workflow_only_retries_judging() -> None:
    metric = PartialMetric()
    experiment = Experiment(
        generation=GenerationConfig(generator=HappyGenerator(), candidate_policy={"num_samples": 1}, reducer=HappyReducer()),
        evaluation=EvaluationConfig(
            metrics=[metric],
            parsers=[HappyParser()],
            judge_models=[PartialJudgeModel("judge/ok")],
        ),
        storage=StorageConfig(store="memory"),
        datasets=_base_dataset(),
        seeds=[7],
    )
    store = InMemoryRunStore()
    experiment.run(store=store)

    stored_run = store.resume(experiment.compile().run_id)
    assert stored_run is not None
    partial_execution = stored_run.execution_state.case_states["case-1"].evaluation_executions["metric/partial"]
    assert partial_execution is not None

    metric.calls = 0
    resumed = experiment.run(store=store)

    assert resumed.run_id == experiment.compile().run_id
    assert metric.calls == 0 or metric.calls == 1


def test_failure_modes_recover_from_store_write_retry() -> None:
    experiment = Experiment(
        generation=GenerationConfig(generator=HappyGenerator(), candidate_policy={"num_samples": 1}, reducer=HappyReducer()),
        evaluation=EvaluationConfig(metrics=[FailingMetric()], parsers=[HappyParser()]),
        storage=StorageConfig(store="memory"),
        datasets=_base_dataset(),
        seeds=[7],
    )
    store = FlakyStore()

    result = experiment.run(store=store)

    assert result.status in {RunStatus.COMPLETED, RunStatus.PARTIAL_FAILURE}
