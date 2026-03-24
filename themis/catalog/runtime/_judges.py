"""Judge helper functions for catalog runtime."""

from __future__ import annotations

from themis import PromptMessage
from themis.errors import MetricError
from themis.extractors.builtin import extract_embedded_json_payload
from themis.types.enums import ErrorCode, PromptRole
from themis.types.json_types import JSONDict
from themis.types.json_validation import validate_json_dict

from ._coercion import _coerce_json_dict
from ._provider import _build_judge_spec


def _run_text_judge(
    *,
    judge_service,
    metric_id: str,
    trial,
    candidate,
    context,
    judge_model_id: str,
    judge_provider: str,
    messages: list[PromptMessage],
    demo_expected_response: str,
) -> str:
    runtime_context = context.get("runtime_context")
    dataset_context = dict(context)
    if judge_provider == "demo":
        dataset_context["judge_expected_response"] = demo_expected_response
    prompt = trial.prompt.model_copy(
        update={"messages": messages, "follow_up_turns": []}
    )
    record = judge_service.judge(
        metric_id,
        candidate,
        _build_judge_spec(model_id=judge_model_id, provider=judge_provider),
        prompt,
        {
            "runtime_context": runtime_context,
            "dataset_context": dataset_context,
            "task_spec": trial.task,
        },
    )
    return record.raw_text or ""


def _prompt_messages_with_optional_system(
    system_prompt: str,
    user_prompt: str,
) -> list[PromptMessage]:
    messages: list[PromptMessage] = []
    if system_prompt:
        messages.append(PromptMessage(role=PromptRole.SYSTEM, content=system_prompt))
    messages.append(PromptMessage(role=PromptRole.USER, content=user_prompt))
    return messages


def _run_json_judge(
    *,
    judge_service,
    metric_id: str,
    trial,
    candidate,
    context,
    judge_model_id: str,
    judge_provider: str,
    messages: list[PromptMessage],
    demo_expected_response: str,
) -> JSONDict:
    judge_raw = _run_text_judge(
        judge_service=judge_service,
        metric_id=metric_id,
        trial=trial,
        candidate=candidate,
        context=context,
        judge_model_id=judge_model_id,
        judge_provider=judge_provider,
        messages=messages,
        demo_expected_response=demo_expected_response,
    )
    parsed = _coerce_json_dict(extract_embedded_json_payload(judge_raw))
    if not parsed:
        raise MetricError(
            code=ErrorCode.METRIC_COMPUTATION,
            message=f"{metric_id} judge returned no JSON payload.",
            details={"judge_raw": judge_raw},
        )
    return validate_json_dict(parsed, label=f"{metric_id} judge payload")
