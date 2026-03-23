"""HLE-specific judge metric."""

from __future__ import annotations

import json

from themis import PromptMessage
from themis.extractors.builtin import _normalize_text, extract_embedded_json_payload
from themis.records import MetricScore
from themis.types.enums import PromptRole

from ...runtime._coercion import (
    _coerce_json_dict,
    _extract_hle_answer,
    _extract_hle_confidence,
)
from ...runtime._judges import _run_text_judge

_HLE_JUDGE_TEMPLATE = """Judge whether the following [response] to [question] is correct based on [correct_answer].

[question]: {question}
[response]: {response}
[correct_answer]: {correct_answer}

Return JSON with:
- "extracted_final_answer": final answer extracted from the response, or "None"
- "reasoning": short explanation
- "correct": "yes" or "no"
- "confidence": integer from 0 to 100
""".strip()


class HLEJudgeMetric:
    def __init__(self, judge_model_id: str, judge_provider: str) -> None:
        self.judge_model_id = judge_model_id
        self.judge_provider = judge_provider

    def score(self, trial, candidate, context):
        judge_service = context["judge_service"]
        response_text = (
            candidate.inference.raw_text if candidate.inference is not None else ""
        )
        question = str(context.get("question", ""))
        correct_answer = str(context.get("answer", ""))
        answer_value = _extract_hle_answer(response_text)
        confidence = _extract_hle_confidence(response_text)
        demo_expected = json.dumps(
            {
                "extracted_final_answer": answer_value or "None",
                "reasoning": "demo HLE judge output",
                "correct": "yes"
                if _normalize_text(answer_value or "")
                == _normalize_text(correct_answer)
                else "no",
                "confidence": confidence,
            }
        )
        judge_raw = _run_text_judge(
            judge_service=judge_service,
            metric_id="hle_accuracy",
            trial=trial,
            candidate=candidate,
            context=context,
            judge_model_id=self.judge_model_id,
            judge_provider=self.judge_provider,
            messages=[
                PromptMessage(
                    role=PromptRole.USER,
                    content=_HLE_JUDGE_TEMPLATE.format(
                        question=question,
                        response=response_text,
                        correct_answer=correct_answer,
                    ),
                )
            ],
            demo_expected_response=demo_expected,
        )
        parsed = _coerce_json_dict(extract_embedded_json_payload(judge_raw))
        correct = str(parsed.get("correct", "no")).lower() == "yes"
        parsed_confidence = int(parsed.get("confidence", confidence or 100))
        return MetricScore(
            metric_id="hle_accuracy",
            value=float(correct),
            details={
                "correct": correct,
                "confidence": parsed_confidence,
                "extracted_final_answer": parsed.get("extracted_final_answer"),
            },
        )
