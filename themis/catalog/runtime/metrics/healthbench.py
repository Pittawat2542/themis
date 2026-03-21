"""HealthBench-specific judge metric."""

from __future__ import annotations

import json

from themis import PromptMessage
from themis.extractors.builtin import extract_embedded_json_payload
from themis.records import MetricScore
from themis.types.enums import PromptRole

from ...datasets import _prompt_messages_from_context
from ..common import _coerce_json_dict, _run_text_judge

_HEALTHBENCH_GRADER_TEMPLATE = """Your job is to look at a conversation and a rubric item, and score the last turn in the conversation on how well it follows the rubric item.

Return a JSON object with:
- "explanation": short explanation
- "criteria_met": boolean

Conversation:
{conversation}

Rubric item:
[{points}] {criterion}
""".strip()


class HealthBenchRubricMetric:
    def __init__(self, judge_model_id: str, judge_provider: str) -> None:
        self.judge_model_id = judge_model_id
        self.judge_provider = judge_provider

    def score(self, trial, candidate, context):
        judge_service = context["judge_service"]
        response_text = (
            candidate.inference.raw_text if candidate.inference is not None else ""
        )
        prompt_messages = _prompt_messages_from_context(context)
        convo_lines = [
            f"{message['role']}: {message['content']}" for message in prompt_messages
        ]
        convo_lines.append(f"assistant: {response_text}")
        conversation = "\n\n".join(convo_lines)
        rubric_rows = [
            item
            for item in context.get("rubrics", [])
            if isinstance(item, dict)
            and isinstance(item.get("criterion"), str)
            and isinstance(item.get("points"), int)
        ]
        grading_results: list[dict[str, object]] = []
        for rubric in rubric_rows:
            criterion = str(rubric["criterion"])
            points = int(rubric["points"])
            demo_expected = json.dumps(
                {
                    "explanation": "demo rubric evaluation",
                    "criteria_met": points > 0,
                }
            )
            judge_raw = _run_text_judge(
                judge_service=judge_service,
                metric_id="healthbench_score",
                trial=trial,
                candidate=candidate,
                context=context,
                judge_model_id=self.judge_model_id,
                judge_provider=self.judge_provider,
                messages=[
                    PromptMessage(
                        role=PromptRole.USER,
                        content=_HEALTHBENCH_GRADER_TEMPLATE.format(
                            conversation=conversation,
                            criterion=criterion,
                            points=points,
                        ),
                    )
                ],
                demo_expected_response=demo_expected,
            )
            parsed = _coerce_json_dict(extract_embedded_json_payload(judge_raw))
            grading_results.append(
                {
                    "criterion": criterion,
                    "points": points,
                    "criteria_met": bool(parsed.get("criteria_met", False)),
                    "explanation": parsed.get("explanation", ""),
                    "tags": list(rubric.get("tags", [])),
                }
            )
        total_possible = sum(
            int(row["points"]) for row in grading_results if int(row["points"]) > 0
        )
        achieved = sum(
            int(row["points"])
            for row in grading_results
            if int(row["points"]) > 0 and bool(row["criteria_met"])
        )
        overall = achieved / total_possible if total_possible > 0 else 0.0
        return MetricScore(
            metric_id="healthbench_score",
            value=float(overall),
            details={
                "example_tags": list(context.get("example_tags", [])),
                "rubric_grades": grading_results,
            },
        )
