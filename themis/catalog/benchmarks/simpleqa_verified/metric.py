"""SimpleQA-specific judge metric."""

from __future__ import annotations

from themis import PromptMessage
from themis.records import MetricScore
from themis.types.enums import PromptRole

from ...runtime._coercion import _parse_simpleqa_grade, _simpleqa_demo_grade
from ...runtime._judges import _run_text_judge

_SIMPLEQA_GRADER_TEMPLATE = """Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].

Question: {question}
Gold target: {target}
Predicted answer: {predicted_answer}

Grade the predicted answer as one of:
A: CORRECT
B: INCORRECT
C: NOT_ATTEMPTED

Just return "A", "B", or "C".
""".strip()


class SimpleQAVerifiedJudgeMetric:
    def __init__(self, judge_model_id: str, judge_provider: str) -> None:
        self.judge_model_id = judge_model_id
        self.judge_provider = judge_provider

    def score(self, trial, candidate, context):
        judge_service = context["judge_service"]
        predicted_answer = (
            candidate.inference.raw_text if candidate.inference is not None else ""
        )
        question = str(context.get("problem", ""))
        target = str(context.get("answer", ""))
        demo_response = _simpleqa_demo_grade(question, target, predicted_answer)
        judge_raw = _run_text_judge(
            judge_service=judge_service,
            metric_id="simpleqa_verified_score",
            trial=trial,
            candidate=candidate,
            context=context,
            judge_model_id=self.judge_model_id,
            judge_provider=self.judge_provider,
            messages=[
                PromptMessage(
                    role=PromptRole.USER,
                    content=_SIMPLEQA_GRADER_TEMPLATE.format(
                        question=question,
                        target=target,
                        predicted_answer=predicted_answer,
                    ),
                )
            ],
            demo_expected_response=demo_response,
        )
        grade = _parse_simpleqa_grade(judge_raw)
        attempted = grade in {"CORRECT", "INCORRECT"}
        return MetricScore(
            metric_id="simpleqa_verified_score",
            value=float(grade == "CORRECT"),
            details={
                "grade": grade,
                "attempted": attempted,
                "topic": context.get("topic"),
                "answer_type": context.get("answer_type"),
            },
        )
