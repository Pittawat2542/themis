from __future__ import annotations

from themis.core.contexts import (
    EvalScoreContext,
    GenerationContext,
    ParseContext,
    ReduceContext,
    ScoreContext,
)
from themis.core.events import RunEvent
from themis.core.models import (
    Case,
    ConversationTrace,
    ParsedOutput,
    ReducedCandidate,
    MetricResult,
    MetricInterpretation,
    ScoreError,
    Candidate,
    TraceStep,
    WorkflowTrace,
)
from themis.core.protocols import (
    CandidateReducer,
    EvaluationWorkflow,
    Generator,
    JudgeModel,
    EventSubscriber,
    Parser,
    PureMetric,
    WorkflowMetric,
    TracingProvider,
    WorkflowRunner,
)
from themis.core.snapshot import ComponentRef
from themis.core.subjects import (
    CandidateSetSubject,
    ConversationSubject,
    CandidateSubject,
    TraceSubject,
)
from themis.core.workflows import (
    AggregationResult,
    EvaluationExecution,
    JudgeCall,
    JudgeResponse,
    ParsedJudgment,
    RenderedJudgePrompt,
)


class DummyWorkflow:
    component_id = "workflow/demo"
    version = "1.0"

    def fingerprint(self) -> str:
        return "workflow-fingerprint"

    def judge_calls(self) -> list[JudgeCall]:
        return [JudgeCall(call_id="call-0", judge_model_id="builtin/demo_judge")]

    def render_prompt(
        self, call: JudgeCall, subject: CandidateSetSubject, ctx: EvalScoreContext
    ) -> RenderedJudgePrompt:
        del call, subject, ctx
        return RenderedJudgePrompt(prompt_id="prompt-0", content="demo")

    def parse_judgment(
        self, call: JudgeCall, response: JudgeResponse, ctx: EvalScoreContext
    ) -> ParsedJudgment:
        del call, ctx
        return ParsedJudgment(label=response.raw_response)

    def score_judgment(
        self, call: JudgeCall, judgment: ParsedJudgment, ctx: EvalScoreContext
    ) -> MetricResult | None:
        del call, ctx
        return MetricResult(metric_id="judge", value=float(judgment.label == "pass"))

    def aggregate(
        self,
        judgments: list[ParsedJudgment],
        scores: list[MetricResult],
        ctx: EvalScoreContext,
    ) -> AggregationResult | None:
        del judgments, ctx
        values = [score.value for score in scores if score.value is not None]
        return AggregationResult(
            method="mean", value=sum(values) / len(values) if values else 0.0
        )


class DummyGenerator:
    component_id = "builtin/demo_generator"
    version = "1.0"

    def fingerprint(self) -> str:
        return "generator-fingerprint"

    async def generate(self, case: Case, ctx: GenerationContext) -> Candidate:
        return Candidate(
            candidate_id=f"{case.case_id}-candidate", final_output={"seed": ctx.seed}
        )


class DummyParser:
    component_id = "builtin/json_identity"
    version = "1.0"

    def fingerprint(self) -> str:
        return "parser-fingerprint"

    def parse(self, candidate: ReducedCandidate, ctx: ParseContext) -> ParsedOutput:
        return ParsedOutput(
            value={"candidate": candidate.candidate_id, "run": ctx.run_id}
        )


class DummyReducer:
    component_id = "builtin/majority_vote"
    version = "1.0"

    def fingerprint(self) -> str:
        return "reducer-fingerprint"

    async def reduce(
        self,
        candidates: list[Candidate],
        ctx: ReduceContext,
    ) -> ReducedCandidate:
        return ReducedCandidate(
            candidate_id=f"{ctx.case_id}-reduced",
            source_candidate_ids=[candidate.candidate_id for candidate in candidates],
            final_output=candidates[0].final_output,
        )


class DummyPureMetric:
    interpretation = MetricInterpretation()
    component_id = "metric/exact_match"
    version = "1.0"

    def fingerprint(self) -> str:
        return "pure-metric-fingerprint"

    def score(
        self, parsed: ParsedOutput, case: Case, ctx: ScoreContext
    ) -> MetricResult | ScoreError:
        del ctx
        return MetricResult(
            metric_id="exact_match", value=float(parsed.value == case.expected_output)
        )


class DummyLLMMetric:
    interpretation = MetricInterpretation()
    component_id = "metric/llm_judge"
    version = "1.0"
    subject_kind = "candidate"

    def fingerprint(self) -> str:
        return "llm-metric-fingerprint"

    def build_workflow(
        self,
        subject: CandidateSetSubject,
        ctx: EvalScoreContext,
    ) -> DummyWorkflow:
        del subject, ctx
        return DummyWorkflow()


class DummySelectionMetric:
    interpretation = MetricInterpretation()
    component_id = "metric/select"
    version = "1.0"
    subject_kind = "candidates"

    def fingerprint(self) -> str:
        return "selection-metric-fingerprint"

    def build_workflow(
        self,
        subject: CandidateSetSubject,
        ctx: EvalScoreContext,
    ) -> DummyWorkflow:
        del subject, ctx
        return DummyWorkflow()


class DummyTraceMetric:
    interpretation = MetricInterpretation()
    component_id = "metric/trace"
    version = "1.0"
    subject_kind = "trace"

    def fingerprint(self) -> str:
        return "trace-metric-fingerprint"

    def build_workflow(
        self,
        subject: TraceSubject | ConversationSubject | CandidateSubject,
        ctx: EvalScoreContext,
    ) -> DummyWorkflow:
        del subject, ctx
        return DummyWorkflow()


class DummyJudgeModel:
    component_id = "builtin/demo_judge"
    version = "1.0"

    def fingerprint(self) -> str:
        return "judge-model-fingerprint"

    async def judge(self, prompt: str, *, seed: int | None = None) -> JudgeResponse:
        del seed
        return JudgeResponse(
            judge_model_id=self.component_id,
            judge_model_version=self.version,
            judge_model_fingerprint=self.fingerprint(),
            raw_response=prompt,
        )


class DummyWorkflowRunner:
    async def run_evaluation(
        self,
        workflow: EvaluationWorkflow,
        subject: CandidateSetSubject
        | TraceSubject
        | ConversationSubject
        | CandidateSubject,
        metric_id: str,
        ctx: EvalScoreContext,
    ) -> EvaluationExecution:
        del workflow, subject, metric_id, ctx
        return EvaluationExecution(
            execution_id="execution-1",
            subject_kind="candidate_set",
            trace=WorkflowTrace(trace_id="trace-1"),
        )


class DummySubscriber:
    def before_generate(self, case: Case, ctx: GenerationContext) -> None:
        del case, ctx

    def after_generate(self, result: Candidate, ctx: GenerationContext) -> None:
        del result, ctx

    def before_reduce(
        self, candidates: list[Candidate], ctx: ReduceContext
    ) -> None:
        del candidates, ctx

    def after_reduce(self, reduced: ReducedCandidate, ctx: ReduceContext) -> None:
        del reduced, ctx

    def before_parse(self, candidate: ReducedCandidate, ctx: ParseContext) -> None:
        del candidate, ctx

    def after_parse(self, parsed: ParsedOutput, ctx: ParseContext) -> None:
        del parsed, ctx

    def before_score(self, parsed: ParsedOutput, ctx: ScoreContext) -> None:
        del parsed, ctx

    def after_score(self, score: MetricResult | ScoreError, ctx: ScoreContext) -> None:
        del score, ctx

    def before_judge(
        self,
        subject: CandidateSetSubject
        | TraceSubject
        | ConversationSubject
        | CandidateSubject,
        ctx: EvalScoreContext,
    ) -> None:
        del subject, ctx

    def after_judge(
        self, execution: EvaluationExecution, ctx: EvalScoreContext
    ) -> None:
        del execution, ctx

    def on_event(self, event: RunEvent) -> None:
        del event


class DummyTracingProvider:
    def start_span(self, name: str, attributes: dict[str, object]) -> object:
        return {"name": name, "attributes": attributes}

    def end_span(self, span: object, status: str) -> None:
        del span, status


def _score_context() -> EvalScoreContext:
    case = Case(
        case_id="case-1", input={"question": "2+2"}, expected_output={"answer": "4"}
    )
    parsed = ParsedOutput(value={"answer": "4"})
    return EvalScoreContext(
        run_id="run-1",
        case=case,
        parsed_views={"default": parsed},
        dataset_metadata={"split": "test"},
        seed=7,
        judge_model_refs=[
            ComponentRef(
                component_id="builtin/demo_judge",
                version="1.0",
                fingerprint="judge-fingerprint",
            )
        ],
        judge_seed=11,
        eval_workflow_config={"rubric": "pass_fail"},
    )


def test_protocol_dummy_implementations_satisfy_runtime_protocols() -> None:
    assert isinstance(DummyGenerator(), Generator)
    assert isinstance(DummyParser(), Parser)
    assert isinstance(DummyReducer(), CandidateReducer)
    assert isinstance(DummyWorkflow(), EvaluationWorkflow)
    assert isinstance(DummyJudgeModel(), JudgeModel)
    assert isinstance(DummyPureMetric(), PureMetric)
    assert isinstance(DummyLLMMetric(), WorkflowMetric)
    assert isinstance(DummySelectionMetric(), WorkflowMetric)
    assert isinstance(DummyTraceMetric(), WorkflowMetric)
    assert isinstance(DummyWorkflowRunner(), WorkflowRunner)
    assert isinstance(DummySubscriber(), EventSubscriber)
    assert isinstance(DummyTracingProvider(), TracingProvider)


def test_dummy_component_fingerprints_are_deterministic() -> None:
    assert DummyGenerator().fingerprint() == DummyGenerator().fingerprint()
    assert DummyWorkflow().fingerprint() == DummyWorkflow().fingerprint()
    assert DummyTraceMetric().fingerprint() == DummyTraceMetric().fingerprint()


def test_workflow_and_execution_models_capture_judge_artifacts() -> None:
    execution = EvaluationExecution(
        execution_id="execution-1",
        subject_kind="candidate_set",
        rendered_prompts=[
            RenderedJudgePrompt(prompt_id="prompt-1", content="Grade this")
        ],
        judge_responses=[
            JudgeResponse(
                judge_model_id="judge-demo",
                judge_model_version="1.0",
                judge_model_fingerprint="judge-fingerprint",
                raw_response="pass",
                token_usage={"prompt_tokens": 3, "completion_tokens": 1},
                latency_ms=10.0,
            )
        ],
        parsed_judgments=[ParsedJudgment(label="pass", score=1.0)],
        metric_results=[MetricResult(metric_id="judge", value=1.0)],
        aggregation_output=AggregationResult(method="mean", value=1.0),
        trace=WorkflowTrace(
            trace_id="trace-1",
            steps=[
                TraceStep(
                    step_name="judge",
                    step_type="model_call",
                    input={"prompt": "Grade"},
                    output={"result": "pass"},
                )
            ],
        ),
    )

    assert execution.judge_responses[0].judge_model_id == "judge-demo"
    assert execution.aggregation_output is not None


def test_metric_protocols_accept_expected_subject_shapes() -> None:
    llm_metric = DummyLLMMetric()
    selection_metric = DummySelectionMetric()
    trace_metric = DummyTraceMetric()
    single = CandidateSetSubject(
        candidates=[Candidate(candidate_id="candidate-1", final_output="4")]
    )
    pair = CandidateSetSubject(
        candidates=[
            Candidate(candidate_id="candidate-1", final_output="4"),
            Candidate(candidate_id="candidate-2", final_output="4"),
        ]
    )
    trace = TraceSubject(
        trace=WorkflowTrace(
            trace_id="trace-1",
            steps=[TraceStep(step_name="judge", step_type="model_call")],
        )
    )
    conversation = ConversationSubject(
        conversation=ConversationTrace(
            trace_id="conversation-1",
            messages=[],
        )
    )
    candidate = CandidateSubject(
        candidate=Candidate(candidate_id="candidate-1", final_output="4")
    )
    ctx = _score_context()

    assert llm_metric.build_workflow(single, ctx).component_id == "workflow/demo"
    assert selection_metric.build_workflow(pair, ctx).component_id == "workflow/demo"
    assert trace_metric.build_workflow(trace, ctx).component_id == "workflow/demo"
    assert (
        trace_metric.build_workflow(conversation, ctx).component_id == "workflow/demo"
    )
    assert trace_metric.build_workflow(candidate, ctx).component_id == "workflow/demo"
