"""FrontierScience-specific judge metric."""

from __future__ import annotations

import json

from themis import PromptMessage
from themis.extractors.builtin import extract_embedded_json_payload
from themis.records import MetricScore
from themis.types.enums import PromptRole

from ...runtime.common import _coerce_json_dict, _run_text_judge

_FRONTIERSCIENCE_GRADER_TEMPLATE = """You are grading a model response against the benchmark rubric.

Problem:
{problem}

Candidate response:
{predicted_answer}

Reference grading rubric:
{rubric}

Return a JSON object with:
- "score": a number between 0 and 1
- "explanation": a short explanation
""".strip()


class FrontierScienceJudgeMetric:
    def __init__(self, judge_model_id: str, judge_provider: str) -> None:
        self.judge_model_id = judge_model_id
        self.judge_provider = judge_provider

    def score(self, trial, candidate, context):
        judge_service = context["judge_service"]
        predicted_answer = (
            candidate.inference.raw_text if candidate.inference is not None else ""
        )
        problem = str(context.get("problem", ""))
        rubric = str(context.get("answer", context.get("expected_response", "")))
        demo_score = float(predicted_answer.strip() == rubric.strip() and bool(rubric))
        judge_raw = _run_text_judge(
            judge_service=judge_service,
            metric_id="frontierscience_score",
            trial=trial,
            candidate=candidate,
            context=context,
            judge_model_id=self.judge_model_id,
            judge_provider=self.judge_provider,
            messages=[
                PromptMessage(
                    role=PromptRole.USER,
                    content=_FRONTIERSCIENCE_GRADER_TEMPLATE.format(
                        problem=problem,
                        predicted_answer=predicted_answer,
                        rubric=rubric,
                    ),
                )
            ],
            demo_expected_response=json.dumps(
                {"score": demo_score, "explanation": "demo rubric grading"}
            ),
        )
        parsed = _coerce_json_dict(extract_embedded_json_payload(judge_raw))
        score = parsed.get("score", 0.0)
        try:
            numeric_score = max(0.0, min(1.0, float(score)))
        except (TypeError, ValueError):
            numeric_score = 0.0
        return MetricScore(
            metric_id="frontierscience_score",
            value=numeric_score,
            details={
                "subject": context.get("subject"),
                "task_group_id": context.get("task_group_id"),
                "judge_response": parsed,
            },
        )
