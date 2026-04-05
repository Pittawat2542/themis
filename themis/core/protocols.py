"""Runtime-checkable extension protocols for Themis."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from themis.core.contexts import (
    EvalScoreContext,
    GenerateContext,
    ParseContext,
    ReduceContext,
    ScoreContext,
    SelectContext,
)
from themis.core.events import RunEvent
from themis.core.models import (
    Case,
    GenerationResult,
    ParsedOutput,
    ReducedCandidate,
    Score,
    ScoreError,
)
from themis.core.subjects import CandidateSetSubject, ConversationSubject, TraceSubject
from themis.core.workflows import (
    AggregationResult,
    EvaluationExecution,
    JudgeCall,
    JudgeResponse,
    ParsedJudgment,
    RenderedJudgePrompt,
)


@runtime_checkable
class Generator(Protocol):
    """Protocol for generation components that produce candidate outputs."""

    component_id: str
    version: str

    def fingerprint(self) -> str: ...

    async def generate(self, case: Case, ctx: GenerateContext) -> GenerationResult: ...


@runtime_checkable
class Parser(Protocol):
    """Protocol for parsers that normalize reduced candidate outputs."""

    component_id: str
    version: str

    def fingerprint(self) -> str: ...

    def parse(self, candidate: ReducedCandidate, ctx: ParseContext) -> ParsedOutput: ...


@runtime_checkable
class CandidateReducer(Protocol):
    """Protocol for reducers that collapse multiple candidates into one."""

    component_id: str
    version: str

    def fingerprint(self) -> str: ...

    async def reduce(
        self,
        candidates: list[GenerationResult],
        ctx: ReduceContext,
    ) -> ReducedCandidate: ...


@runtime_checkable
class CandidateSelector(Protocol):
    """Protocol for selectors that choose candidates before reduction."""

    component_id: str
    version: str

    def fingerprint(self) -> str: ...

    async def select(
        self,
        candidates: list[GenerationResult],
        ctx: SelectContext,
    ) -> list[GenerationResult]: ...


@runtime_checkable
class EvaluationWorkflow(Protocol):
    """Protocol for workflow-backed metrics driven by judge model calls."""

    component_id: str
    version: str

    def fingerprint(self) -> str: ...

    def judge_calls(self) -> list[JudgeCall]: ...

    def render_prompt(
        self,
        call: JudgeCall,
        subject: CandidateSetSubject | TraceSubject | ConversationSubject,
        ctx: EvalScoreContext,
    ) -> RenderedJudgePrompt: ...

    def parse_judgment(
        self,
        call: JudgeCall,
        response: JudgeResponse,
        ctx: EvalScoreContext,
    ) -> ParsedJudgment: ...

    def score_judgment(
        self,
        call: JudgeCall,
        judgment: ParsedJudgment,
        ctx: EvalScoreContext,
    ) -> Score | None: ...

    def aggregate(
        self,
        judgments: list[ParsedJudgment],
        scores: list[Score],
        ctx: EvalScoreContext,
    ) -> AggregationResult | None: ...


@runtime_checkable
class JudgeModel(Protocol):
    """Protocol for judge models used inside evaluation workflows."""

    component_id: str
    version: str

    def fingerprint(self) -> str: ...

    async def judge(self, prompt: str, *, seed: int | None = None) -> JudgeResponse: ...


@runtime_checkable
class PureMetric(Protocol):
    """Protocol for deterministic metrics that score parsed outputs directly."""

    component_id: str
    version: str

    def fingerprint(self) -> str: ...

    def score(
        self, parsed: ParsedOutput, case: Case, ctx: ScoreContext
    ) -> Score | ScoreError: ...


@runtime_checkable
class LLMMetric(Protocol):
    """Protocol for metrics that judge a reduced candidate set with an LLM."""

    component_id: str
    version: str

    def fingerprint(self) -> str: ...

    def build_workflow(
        self,
        subject: CandidateSetSubject,
        ctx: EvalScoreContext,
    ) -> EvaluationWorkflow: ...


@runtime_checkable
class SelectionMetric(Protocol):
    """Protocol for metrics that judge multiple generated candidates."""

    component_id: str
    version: str

    def fingerprint(self) -> str: ...

    def build_workflow(
        self,
        subject: CandidateSetSubject,
        ctx: EvalScoreContext,
    ) -> EvaluationWorkflow: ...


@runtime_checkable
class TraceMetric(Protocol):
    """Protocol for metrics that score traces or conversations."""

    component_id: str
    version: str

    def fingerprint(self) -> str: ...

    def build_workflow(
        self,
        subject: TraceSubject | ConversationSubject,
        ctx: EvalScoreContext,
    ) -> EvaluationWorkflow: ...


@runtime_checkable
class WorkflowRunner(Protocol):
    """Protocol for executing evaluation workflows and returning traces."""

    async def run_evaluation(
        self,
        workflow: EvaluationWorkflow,
        subject: CandidateSetSubject | TraceSubject | ConversationSubject,
        metric_id: str,
        ctx: EvalScoreContext,
    ) -> EvaluationExecution: ...


@runtime_checkable
class BeforeGenerate(Protocol):
    """Hook invoked before a generator runs."""

    def before_generate(self, case: Case, ctx: GenerateContext) -> None: ...


@runtime_checkable
class AfterGenerate(Protocol):
    """Hook invoked after a generator returns a candidate."""

    def after_generate(
        self, result: GenerationResult, ctx: GenerateContext
    ) -> None: ...


@runtime_checkable
class BeforeReduce(Protocol):
    """Hook invoked before reduction starts."""

    def before_reduce(
        self, candidates: list[GenerationResult], ctx: ReduceContext
    ) -> None: ...


@runtime_checkable
class AfterReduce(Protocol):
    """Hook invoked after reduction produces a final candidate."""

    def after_reduce(self, reduced: ReducedCandidate, ctx: ReduceContext) -> None: ...


@runtime_checkable
class BeforeParse(Protocol):
    """Hook invoked before parsing a reduced candidate."""

    def before_parse(self, candidate: ReducedCandidate, ctx: ParseContext) -> None: ...


@runtime_checkable
class AfterParse(Protocol):
    """Hook invoked after parsing completes."""

    def after_parse(self, parsed: ParsedOutput, ctx: ParseContext) -> None: ...


@runtime_checkable
class BeforeScore(Protocol):
    """Hook invoked before a pure metric runs."""

    def before_score(self, parsed: ParsedOutput, ctx: ScoreContext) -> None: ...


@runtime_checkable
class AfterScore(Protocol):
    """Hook invoked after a pure metric emits a score or error."""

    def after_score(self, score: Score | ScoreError, ctx: ScoreContext) -> None: ...


@runtime_checkable
class BeforeJudge(Protocol):
    """Hook invoked before a workflow-backed metric begins judging."""

    def before_judge(
        self,
        subject: CandidateSetSubject | TraceSubject | ConversationSubject,
        ctx: EvalScoreContext,
    ) -> None: ...


@runtime_checkable
class AfterJudge(Protocol):
    """Hook invoked after a workflow-backed metric finishes."""

    def after_judge(
        self, execution: EvaluationExecution, ctx: EvalScoreContext
    ) -> None: ...


@runtime_checkable
class OnEvent(Protocol):
    """Hook invoked after an execution event is persisted."""

    def on_event(self, event: RunEvent) -> None: ...


@runtime_checkable
class LifecycleSubscriber(
    BeforeGenerate,
    AfterGenerate,
    BeforeReduce,
    AfterReduce,
    BeforeParse,
    AfterParse,
    BeforeScore,
    AfterScore,
    BeforeJudge,
    AfterJudge,
    OnEvent,
    Protocol,
):
    """Aggregate lifecycle subscriber protocol."""


@runtime_checkable
class TracingProvider(Protocol):
    """Protocol for span-based tracing integrations."""

    def start_span(self, name: str, attributes: dict[str, object]) -> object: ...

    def end_span(self, span: object, status: str) -> None: ...
