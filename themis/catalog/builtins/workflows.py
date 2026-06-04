"""Builtin workflow-backed metrics."""

from __future__ import annotations

from collections import Counter

from themis.core.contexts import EvalScoreContext
from themis.core.models import MetricResult
from themis.core.prompts import render_prompt_spec
from themis.core.subjects import CandidateSetSubject, ConversationSubject, TraceSubject
from themis.core.workflows import (
    AggregationResult,
    JudgeCall,
    JudgeResponse,
    ParsedJudgment,
    RenderedJudgePrompt,
    build_prompt_template_context,
)


class LLMRubricMetric:
    component_id = "builtin/llm_rubric"
    version = "1.0"
    metric_family = "llm"

    def fingerprint(self) -> str:
        return "builtin-llm-rubric-fingerprint"

    def build_workflow(self, subject: CandidateSetSubject, ctx: EvalScoreContext):
        return _RubricWorkflow(
            metric_id=self.component_id,
            fingerprint_value=self.fingerprint(),
            rubric=_rubric_from_ctx(ctx),
            judge_model_ids=[ref.component_id for ref in ctx.judge_model_refs],
            aggregation="mean",
        )


class PanelOfJudgesMetric:
    component_id = "builtin/panel_of_judges"
    version = "1.0"
    metric_family = "llm"

    def fingerprint(self) -> str:
        return "builtin-panel-of-judges-fingerprint"

    def build_workflow(self, subject: CandidateSetSubject, ctx: EvalScoreContext):
        return _RubricWorkflow(
            metric_id=self.component_id,
            fingerprint_value=self.fingerprint(),
            rubric=_rubric_from_ctx(ctx),
            judge_model_ids=[ref.component_id for ref in ctx.judge_model_refs],
            aggregation="mean",
        )


class MajorityVoteJudgeMetric:
    component_id = "builtin/majority_vote_judge"
    version = "1.0"
    metric_family = "llm"

    def fingerprint(self) -> str:
        return "builtin-majority-vote-judge-fingerprint"

    def build_workflow(self, subject: CandidateSetSubject, ctx: EvalScoreContext):
        return _RubricWorkflow(
            metric_id=self.component_id,
            fingerprint_value=self.fingerprint(),
            rubric=_rubric_from_ctx(ctx),
            judge_model_ids=[ref.component_id for ref in ctx.judge_model_refs],
            aggregation="majority_vote",
        )


class PairwiseJudgeMetric:
    component_id = "builtin/pairwise_judge"
    version = "1.0"
    metric_family = "selection"

    def fingerprint(self) -> str:
        return "builtin-pairwise-judge-fingerprint"

    def build_workflow(self, subject: CandidateSetSubject, ctx: EvalScoreContext):
        return _PairwiseWorkflow(
            metric_id=self.component_id,
            fingerprint_value=self.fingerprint(),
            rubric=_rubric_from_ctx(ctx),
            judge_model_ids=[ref.component_id for ref in ctx.judge_model_refs],
        )


class RankingJudgeMetric:
    component_id = "builtin/ranking_judge"
    version = "1.0"
    metric_family = "selection"

    def fingerprint(self) -> str:
        return "builtin-ranking-judge-fingerprint"

    def build_workflow(self, subject: CandidateSetSubject, ctx: EvalScoreContext):
        return _RankingWorkflow(
            metric_id=self.component_id,
            fingerprint_value=self.fingerprint(),
            rubric=_rubric_from_ctx(ctx),
            judge_model_ids=[ref.component_id for ref in ctx.judge_model_refs],
            candidate_count=len(subject.candidates),
        )


class _RubricWorkflow:
    component_id = "workflow/builtin_rubric"
    version = "1.0"

    def __init__(
        self,
        *,
        metric_id: str,
        fingerprint_value: str,
        rubric: str,
        judge_model_ids: list[str],
        aggregation: str,
    ) -> None:
        self.metric_id = metric_id
        self._fingerprint_value = fingerprint_value
        self.rubric = rubric
        self.judge_model_ids = judge_model_ids
        self.aggregation = aggregation

    def fingerprint(self) -> str:
        return self._fingerprint_value

    def judge_calls(self) -> list[JudgeCall]:
        return [
            JudgeCall(
                call_id=f"judge-{index}",
                judge_model_id=judge_model_id,
                repeat_index=index,
            )
            for index, judge_model_id in enumerate(self.judge_model_ids)
        ]

    def render_prompt(
        self,
        call: JudgeCall,
        subject: CandidateSetSubject | TraceSubject | ConversationSubject,
        ctx: EvalScoreContext,
    ) -> RenderedJudgePrompt:
        template_ctx = build_prompt_template_context(subject, ctx, call)
        return RenderedJudgePrompt(
            prompt_id=f"{self.metric_id}:{call.call_id}",
            content=render_prompt_spec(
                ctx.prompt_spec,
                (
                    f"Rubric: {self.rubric}\n"
                    f"Case input: {template_ctx['case_input']}\n"
                    f"Candidate output: {template_ctx['candidate_output']}\n"
                    "Respond with PASS or FAIL."
                ),
            ),
        )

    def parse_judgment(
        self,
        call: JudgeCall,
        response: JudgeResponse,
        ctx: EvalScoreContext,
    ) -> ParsedJudgment:
        del call, ctx
        label = response.raw_response.strip().split()[0].lower()
        score = _pass_fail_score(label)
        return ParsedJudgment(label=label, score=score, rationale=response.raw_response)

    def score_judgment(
        self,
        call: JudgeCall,
        judgment: ParsedJudgment,
        ctx: EvalScoreContext,
    ) -> MetricResult | None:
        del call, ctx
        return MetricResult(
            metric_id=self.metric_id,
            value=float(judgment.score or 0.0),
            metadata={"label": judgment.label},
        )

    def aggregate(
        self,
        judgments: list[ParsedJudgment],
        metric_results: list[MetricResult],
        ctx: EvalScoreContext,
    ) -> AggregationResult | None:
        del ctx
        if not metric_results:
            return AggregationResult(method=self.aggregation, value=0.0)
        if self.aggregation == "majority_vote":
            labels = [judgment.label for judgment in judgments]
            winner = Counter(labels).most_common(1)[0][0]
            return AggregationResult(
                method="majority_vote",
                value=_pass_fail_score(winner),
                metadata={"winner": winner},
            )
        return AggregationResult(
            method="mean",
            value=sum((result.value or 0.0) for result in metric_results)
            / len(metric_results),
            metadata={"count": len(metric_results)},
        )


class _PairwiseWorkflow:
    component_id = "workflow/builtin_pairwise"
    version = "1.0"

    def __init__(
        self,
        *,
        metric_id: str,
        fingerprint_value: str,
        rubric: str,
        judge_model_ids: list[str],
    ) -> None:
        self.metric_id = metric_id
        self._fingerprint_value = fingerprint_value
        self.rubric = rubric
        self.judge_model_ids = judge_model_ids

    def fingerprint(self) -> str:
        return self._fingerprint_value

    def judge_calls(self) -> list[JudgeCall]:
        return [
            JudgeCall(
                call_id=f"pairwise-{index}",
                judge_model_id=judge_model_id,
                dimension_id="winner",
                repeat_index=index,
                candidate_indices=[0, 1],
            )
            for index, judge_model_id in enumerate(self.judge_model_ids)
        ]

    def render_prompt(
        self,
        call: JudgeCall,
        subject: CandidateSetSubject | TraceSubject | ConversationSubject,
        ctx: EvalScoreContext,
    ) -> RenderedJudgePrompt:
        template_ctx = build_prompt_template_context(subject, ctx, call)
        return RenderedJudgePrompt(
            prompt_id=f"{self.metric_id}:{call.call_id}",
            content=render_prompt_spec(
                ctx.prompt_spec,
                (
                    f"Rubric: {self.rubric}\n"
                    f"Candidate A: {template_ctx['candidate_a_output']}\n"
                    f"Candidate B: {template_ctx['candidate_b_output']}\n"
                    "Respond with A, B, or TIE."
                ),
            ),
        )

    def parse_judgment(
        self,
        call: JudgeCall,
        response: JudgeResponse,
        ctx: EvalScoreContext,
    ) -> ParsedJudgment:
        del call, ctx
        label = response.raw_response.strip().split()[0].lower()
        return ParsedJudgment(
            label=label, score=_pairwise_score(label), rationale=response.raw_response
        )

    def score_judgment(
        self,
        call: JudgeCall,
        judgment: ParsedJudgment,
        ctx: EvalScoreContext,
    ) -> MetricResult | None:
        del call, ctx
        return MetricResult(
            metric_id=self.metric_id,
            result_type="preference",
            value=float(judgment.score or 0.0),
            labels={"winner": judgment.label},
            confidence=1.0,
            metadata={"label": judgment.label},
        )

    def aggregate(
        self,
        judgments: list[ParsedJudgment],
        metric_results: list[MetricResult],
        ctx: EvalScoreContext,
    ) -> AggregationResult | None:
        del ctx
        if not metric_results:
            return AggregationResult(method="mean", value=0.0)
        labels = [judgment.label for judgment in judgments]
        winner = Counter(labels).most_common(1)[0][0]
        confidence = labels.count(winner) / len(labels) if labels else None
        return AggregationResult(
            method="majority_vote",
            value=sum((result.value or 0.0) for result in metric_results)
            / len(metric_results),
            metric_result=MetricResult(
                metric_id=self.metric_id,
                result_type="preference",
                value=sum((result.value or 0.0) for result in metric_results)
                / len(metric_results),
                labels={"winner": winner},
                confidence=confidence,
                metadata={"votes": dict(Counter(labels))},
            ),
            metadata={"winner": winner},
        )


class _RankingWorkflow:
    component_id = "workflow/builtin_ranking"
    version = "1.0"

    def __init__(
        self,
        *,
        metric_id: str,
        fingerprint_value: str,
        rubric: str,
        judge_model_ids: list[str],
        candidate_count: int,
    ) -> None:
        self.metric_id = metric_id
        self._fingerprint_value = fingerprint_value
        self.rubric = rubric
        self.judge_model_ids = judge_model_ids
        self.candidate_count = candidate_count

    def fingerprint(self) -> str:
        return self._fingerprint_value

    def judge_calls(self) -> list[JudgeCall]:
        candidate_indices = list(range(self.candidate_count))
        return [
            JudgeCall(
                call_id=f"ranking-{index}",
                judge_model_id=judge_model_id,
                dimension_id="ranking",
                repeat_index=index,
                candidate_indices=candidate_indices,
            )
            for index, judge_model_id in enumerate(self.judge_model_ids)
        ]

    def render_prompt(
        self,
        call: JudgeCall,
        subject: CandidateSetSubject | TraceSubject | ConversationSubject,
        ctx: EvalScoreContext,
    ) -> RenderedJudgePrompt:
        del call
        candidates = []
        if isinstance(subject, CandidateSetSubject):
            candidates = [
                f"{index}: {candidate.final_output}"
                for index, candidate in enumerate(subject.candidates)
            ]
        return RenderedJudgePrompt(
            prompt_id=f"{self.metric_id}:ranking",
            content=render_prompt_spec(
                ctx.prompt_spec,
                (
                    f"Rubric: {self.rubric}\n"
                    "Rank candidates from best to worst by numeric index.\n"
                    + "\n".join(candidates)
                ),
            ),
        )

    def parse_judgment(
        self,
        call: JudgeCall,
        response: JudgeResponse,
        ctx: EvalScoreContext,
    ) -> ParsedJudgment:
        del call, ctx
        ranking = _parse_ranking(response.raw_response, self.candidate_count)
        winner = str(ranking[0]) if ranking else ""
        return ParsedJudgment(
            label=winner,
            score=1.0 if ranking else 0.0,
            rationale=response.raw_response,
            details={"ranking": [str(candidate_id) for candidate_id in ranking]},
        )

    def score_judgment(
        self,
        call: JudgeCall,
        judgment: ParsedJudgment,
        ctx: EvalScoreContext,
    ) -> MetricResult | None:
        del call, ctx
        ranking = judgment.details.get("ranking", [])
        labels = {
            "winner": str(ranking[0])
            for ranking in [ranking]
            if isinstance(ranking, list) and ranking
        }
        return MetricResult(
            metric_id=self.metric_id,
            result_type="ranking",
            value=float(judgment.score or 0.0),
            labels=labels,
            confidence=1.0,
            metadata={
                "ranking": [str(candidate_id) for candidate_id in ranking]
                if isinstance(ranking, list)
                else []
            },
        )

    def aggregate(
        self,
        judgments: list[ParsedJudgment],
        metric_results: list[MetricResult],
        ctx: EvalScoreContext,
    ) -> AggregationResult | None:
        del ctx
        if not metric_results:
            return AggregationResult(method="ranking_vote", value=0.0)
        winners = [
            result.labels["winner"]
            for result in metric_results
            if "winner" in result.labels
        ]
        winner = Counter(winners).most_common(1)[0][0] if winners else ""
        confidence = winners.count(winner) / len(winners) if winners else None
        ranking_votes = [
            result.metadata.get("ranking", []) for result in metric_results
        ]
        aggregate_value = sum((result.value or 0.0) for result in metric_results) / len(
            metric_results
        )
        return AggregationResult(
            method="ranking_vote",
            value=aggregate_value,
            metric_result=MetricResult(
                metric_id=self.metric_id,
                result_type="ranking",
                value=aggregate_value,
                labels={"winner": winner} if winner else {},
                confidence=confidence,
                metadata={
                    "ranking_votes": ranking_votes,
                    "winner_votes": dict(Counter(winners)),
                },
            ),
            metadata={"winner": winner},
        )


def _rubric_from_ctx(ctx: EvalScoreContext) -> str:
    rubric = ctx.eval_workflow_config.get("rubric")
    if isinstance(rubric, str) and rubric:
        return rubric
    return "Judge whether the candidate output correctly answers the case."


def _pass_fail_score(label: str) -> float:
    if label in {"pass", "yes", "correct", "true", "1"}:
        return 1.0
    if label in {"fail", "no", "incorrect", "false", "0"}:
        return 0.0
    try:
        return float(label)
    except ValueError:
        return 0.0


def _pairwise_score(label: str) -> float:
    if label == "a":
        return 1.0
    if label == "b":
        return 0.0
    if label == "tie":
        return 0.5
    return 0.0


def _parse_ranking(raw_response: str, candidate_count: int) -> list[int]:
    values: list[int] = []
    for token in raw_response.replace(",", " ").replace(">", " ").split():
        normalized = token.strip().strip("[]()#:")
        if not normalized.isdigit():
            continue
        value = int(normalized)
        if 0 <= value < candidate_count and value not in values:
            values.append(value)
    values.extend(index for index in range(candidate_count) if index not in values)
    return values
