"""LPFQA-specific judge metric."""

from __future__ import annotations

import json

from themis.extractors.builtin import _normalize_text, extract_embedded_json_payload
from themis.records import MetricScore

from ...runtime.common import (
    _coerce_json_dict,
    _extract_lpfqa_reference_answer,
    _prompt_messages_with_optional_system,
    _run_text_judge,
)


class LPFQAJudgeMetric:
    def __init__(self, judge_model_id: str, judge_provider: str) -> None:
        self.judge_model_id = judge_model_id
        self.judge_provider = judge_provider

    def score(self, trial, candidate, context):
        judge_service = context["judge_service"]
        response_text = (
            candidate.inference.raw_text if candidate.inference is not None else ""
        )
        response_reference = str(context.get("response_reference", ""))
        judge_prompt_template = str(
            context.get("judge_prompt_template", "{response_reference}\n{response}")
        )
        judge_system_prompt = str(context.get("judge_system_prompt", ""))
        prompt = judge_prompt_template.format(
            response_reference=response_reference,
            response=response_text,
        )
        demo_score = int(
            _normalize_text(response_text)
            == _normalize_text(_extract_lpfqa_reference_answer(response_reference))
        )
        judge_raw = _run_text_judge(
            judge_service=judge_service,
            metric_id="lpfqa_score",
            trial=trial,
            candidate=candidate,
            context=context,
            judge_model_id=self.judge_model_id,
            judge_provider=self.judge_provider,
            messages=_prompt_messages_with_optional_system(judge_system_prompt, prompt),
            demo_expected_response=json.dumps({"answer_score": demo_score}),
        )
        parsed = _coerce_json_dict(extract_embedded_json_payload(judge_raw))
        score = int(parsed.get("answer_score", 0))
        return MetricScore(
            metric_id="lpfqa_score",
            value=float(score),
            details={"answer_score": score, "judge_response": parsed},
        )
