"""Runtime-checkable extension protocols for Themis."""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

from themis.core.contexts import (
    EvalScoreContext,
    GenerationContext,
    ParseContext,
    ReduceContext,
    ScoreContext,
    SelectContext,
)
from themis.core.events import RunEvent
from themis.core.models import (
    Case,
    Candidate,
    MetricResult,
    ParsedOutput,
    ReducedCandidate,
    ScoreError,
)
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

type MetricSubjectKind = Literal["candidate", "candidates", "trace"]


@runtime_checkable
class Generator(Protocol):
    """Protocol for candidate generation components."""

    component_id: str
    version: str

    def fingerprint(self) -> str: ...

    async def generate(self, case: Case, ctx: GenerationContext) -> Candidate: ...


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
        candidates: list[Candidate],
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
        candidates: list[Candidate],
        ctx: SelectContext,
    ) -> list[Candidate]: ...


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
        subject: CandidateSetSubject
        | TraceSubject
        | ConversationSubject
        | CandidateSubject,
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
    ) -> MetricResult | None: ...

    def aggregate(
        self,
        judgments: list[ParsedJudgment],
        scores: list[MetricResult],
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
    ) -> MetricResult | ScoreError: ...


@runtime_checkable
class WorkflowMetric(Protocol):
    """Protocol for judge-backed metrics over an explicitly declared subject."""

    component_id: str
    version: str
    subject_kind: MetricSubjectKind

    def fingerprint(self) -> str: ...

    def build_workflow(
        self,
        subject: Any,
        ctx: EvalScoreContext,
    ) -> EvaluationWorkflow: ...


@runtime_checkable
class WorkflowRunner(Protocol):
    """Protocol for executing evaluation workflows and returning traces."""

    async def run_evaluation(
        self,
        workflow: EvaluationWorkflow,
        subject: CandidateSetSubject
        | TraceSubject
        | ConversationSubject
        | CandidateSubject,
        metric_id: str,
        ctx: EvalScoreContext,
    ) -> EvaluationExecution: ...


@runtime_checkable
class EventSubscriber(Protocol):
    """Receives immutable events only after they have been persisted."""

    def on_event(self, event: RunEvent) -> None: ...


@runtime_checkable
class TracingProvider(Protocol):
    """Protocol for span-based tracing integrations."""

    def start_span(self, name: str, attributes: dict[str, object]) -> object: ...

    def end_span(self, span: object, status: str) -> None: ...
