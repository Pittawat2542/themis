"""Runtime-checkable extension protocols for Themis v4."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from themis.core.contexts import (
    EvalScoreContext,
    GenerateContext,
    ParseContext,
    ReduceContext,
    ScoreContext,
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
from themis.core.workflows import EvalStep, EvaluationExecution, JudgeResponse


@runtime_checkable
class Generator(Protocol):
    component_id: str
    version: str

    def fingerprint(self) -> str: ...

    async def generate(self, case: Case, ctx: GenerateContext) -> GenerationResult: ...


@runtime_checkable
class Parser(Protocol):
    component_id: str
    version: str

    def fingerprint(self) -> str: ...

    def parse(self, candidate: ReducedCandidate, ctx: ParseContext) -> ParsedOutput: ...


@runtime_checkable
class CandidateReducer(Protocol):
    component_id: str
    version: str

    def fingerprint(self) -> str: ...

    def reduce(
        self,
        candidates: list[GenerationResult],
        ctx: ReduceContext,
    ) -> ReducedCandidate: ...


@runtime_checkable
class EvaluationWorkflow(Protocol):
    component_id: str
    version: str

    def fingerprint(self) -> str: ...

    def steps(self) -> list[EvalStep]: ...


@runtime_checkable
class JudgeModel(Protocol):
    component_id: str
    version: str

    def fingerprint(self) -> str: ...

    async def judge(self, prompt: str, *, seed: int | None = None) -> JudgeResponse: ...


@runtime_checkable
class PureMetric(Protocol):
    component_id: str
    version: str

    def fingerprint(self) -> str: ...

    def score(self, parsed: ParsedOutput, case: Case, ctx: ScoreContext) -> Score | ScoreError: ...


@runtime_checkable
class LLMMetric(Protocol):
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
    async def run_evaluation(
        self,
        workflow: EvaluationWorkflow,
        subject: CandidateSetSubject | TraceSubject | ConversationSubject,
        metric_id: str,
        ctx: EvalScoreContext,
    ) -> EvaluationExecution: ...


@runtime_checkable
class BeforeGenerate(Protocol):
    def before_generate(self, case: Case, ctx: GenerateContext) -> None: ...


@runtime_checkable
class AfterGenerate(Protocol):
    def after_generate(self, result: GenerationResult, ctx: GenerateContext) -> None: ...


@runtime_checkable
class BeforeReduce(Protocol):
    def before_reduce(self, candidates: list[GenerationResult], ctx: ReduceContext) -> None: ...


@runtime_checkable
class AfterReduce(Protocol):
    def after_reduce(self, reduced: ReducedCandidate, ctx: ReduceContext) -> None: ...


@runtime_checkable
class BeforeParse(Protocol):
    def before_parse(self, candidate: ReducedCandidate, ctx: ParseContext) -> None: ...


@runtime_checkable
class AfterParse(Protocol):
    def after_parse(self, parsed: ParsedOutput, ctx: ParseContext) -> None: ...


@runtime_checkable
class BeforeScore(Protocol):
    def before_score(self, parsed: ParsedOutput, ctx: ScoreContext) -> None: ...


@runtime_checkable
class AfterScore(Protocol):
    def after_score(self, score: Score | ScoreError, ctx: ScoreContext) -> None: ...


@runtime_checkable
class BeforeJudge(Protocol):
    def before_judge(
        self,
        subject: CandidateSetSubject | TraceSubject | ConversationSubject,
        ctx: EvalScoreContext,
    ) -> None: ...


@runtime_checkable
class AfterJudge(Protocol):
    def after_judge(self, execution: EvaluationExecution, ctx: EvalScoreContext) -> None: ...


@runtime_checkable
class OnEvent(Protocol):
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
    def start_span(self, name: str, attributes: dict[str, object]) -> object: ...

    def end_span(self, span: object, status: str) -> None: ...
