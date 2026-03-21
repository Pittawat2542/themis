"""Starter engines, metrics, and judge helpers."""

from __future__ import annotations

import json
from time import perf_counter
import re
from typing import cast, Literal

from themis import InferenceParamsSpec, ModelSpec, PluginRegistry, PromptMessage
from themis._optional import import_optional
from themis.contracts.protocols import InferenceResult
from themis.errors import InferenceError, RetryableProviderError
from themis.extractors.builtin import _normalize_text, extract_embedded_json_payload
from themis.records import InferenceRecord, MetricScore, TokenUsage
from themis.registry import EngineCapabilities
from themis.specs.foundational import JudgeInferenceSpec
from themis.types.enums import ErrorCode, PromptRole, ResponseFormat
from themis.types.json_types import JSONDict

from .datasets import _prompt_messages_from_context

_SIMPLEQA_GRADE_PATTERN = re.compile(
    r"\b(CORRECT|INCORRECT|NOT[_ ]ATTEMPTED|A|B|C)\b", flags=re.IGNORECASE
)
_LPFQA_REFERENCE_PATTERN = re.compile(
    r"<参考答案>[：:]\s*(?P<answer>.*?)\s*(?:<评估要点>|$)",
    flags=re.DOTALL,
)
_HLE_ANSWER_PATTERN = re.compile(r"(?im)^answer:\s*(?P<value>.+)$")
_HLE_CONFIDENCE_PATTERN = re.compile(r"(?im)^confidence:\s*(?P<value>\d+)")

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

_HEALTHBENCH_GRADER_TEMPLATE = """Your job is to look at a conversation and a rubric item, and score the last turn in the conversation on how well it follows the rubric item.

Return a JSON object with:
- "explanation": short explanation
- "criteria_met": boolean

Conversation:
{conversation}

Rubric item:
[{points}] {criterion}
""".strip()

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


def build_starter_registry(providers: str | list[str]) -> PluginRegistry:
    """Build a registry containing the starter metrics and requested engines."""

    registry = PluginRegistry()
    register_starter_metrics(registry)
    resolved_providers = [providers] if isinstance(providers, str) else list(providers)
    for provider in sorted(
        {_normalize_provider_name(provider) for provider in resolved_providers}
    ):
        register_starter_engine(registry, provider)
    return registry


def register_starter_metrics(registry: PluginRegistry) -> None:
    """Register the curated starter metric set on an existing registry."""

    registry.register_metric("exact_match", ExactMatchMetric())
    registry.register_metric("normalized_exact_match", NormalizedExactMatchMetric())
    registry.register_metric("choice_accuracy", ChoiceAccuracyMetric())
    registry.register_metric("numeric_exact_match", NumericExactMatchMetric())


def register_starter_engine(registry: PluginRegistry, provider: str) -> None:
    """Register one starter inference engine on an existing registry."""

    normalized_provider = _normalize_provider_name(provider)
    if normalized_provider == "demo":
        registry.register_inference_engine(
            "demo",
            DemoEngine(),
            capabilities=EngineCapabilities(supports_seed=True),
        )
        return
    if normalized_provider == "openai":
        registry.register_inference_engine(
            "openai",
            OpenAIChatEngine(),
            capabilities=EngineCapabilities(supports_seed=True),
        )
        return
    if normalized_provider == "openai_compatible":
        registry.register_inference_engine(
            "openai_compatible",
            OpenAICompatibleChatEngine(),
            capabilities=EngineCapabilities(supports_seed=True),
        )
        return
    raise ValueError(f"Unsupported quick-start provider '{provider}'.")


class DemoEngine:
    """Offline engine that echoes the expected answer for smoke tests."""

    def infer(self, trial, context, runtime):
        del trial, runtime
        raw_text = _expected_demo_response(context)
        return InferenceResult(
            inference=InferenceRecord(
                spec_hash=f"inference_{_context_item_id(context)}",
                raw_text=raw_text,
            )
        )


class OpenAIChatEngine:
    """Minimal OpenAI chat-completions adapter for starter workflows."""

    def infer(self, trial, context, runtime):
        return _run_openai_chat_inference(
            trial,
            context,
            runtime,
            base_url=None,
            provider_label="OpenAI",
            missing_extra="providers-openai",
        )


class OpenAICompatibleChatEngine:
    """Minimal OpenAI-compatible chat-completions adapter for starter workflows."""

    def infer(self, trial, context, runtime):
        extras = dict(trial.model.extras)
        base_url = str(extras.get("base_url", "http://127.0.0.1:8000/v1")).rstrip("/")
        return _run_openai_chat_inference(
            trial,
            context,
            runtime,
            base_url=base_url,
            provider_label="OpenAI-compatible endpoint",
            missing_extra="providers-openai",
        )


class ExactMatchMetric:
    def score(self, trial, candidate, context):
        del trial
        actual = candidate.inference.raw_text if candidate.inference is not None else ""
        return MetricScore(
            metric_id="exact_match",
            value=float(actual == _expected_text(context)),
        )


class NormalizedExactMatchMetric:
    def score(self, trial, candidate, context):
        del trial
        extraction = candidate.best_extraction()
        parsed = (
            extraction.parsed_answer
            if extraction is not None and extraction.success
            else ""
        )
        return MetricScore(
            metric_id="normalized_exact_match",
            value=float(str(parsed) == _normalize_text(_expected_text(context))),
        )


class ChoiceAccuracyMetric:
    def score(self, trial, candidate, context):
        del trial
        extraction = candidate.best_extraction()
        parsed = (
            extraction.parsed_answer
            if extraction is not None and extraction.success
            else ""
        )
        expected = _expected_text(context).strip().upper()
        return MetricScore(
            metric_id="choice_accuracy",
            value=float(str(parsed).strip().upper() == expected),
        )


class NumericExactMatchMetric:
    def score(self, trial, candidate, context):
        del trial
        extraction = candidate.best_extraction()
        parsed = (
            extraction.parsed_answer
            if extraction is not None and extraction.success
            else None
        )
        actual = _coerce_float(parsed)
        expected = _coerce_float(_expected_text(context))
        return MetricScore(
            metric_id="numeric_exact_match",
            value=float(
                actual is not None and expected is not None and actual == expected
            ),
        )


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


def _run_openai_chat_inference(
    trial,
    context,
    runtime,
    *,
    base_url: str | None,
    provider_label: str,
    missing_extra: str,
) -> InferenceResult:
    if trial.params.response_format not in (None, ResponseFormat.TEXT):
        raise InferenceError(
            code=ErrorCode.PLUGIN_INCOMPATIBLE,
            message=f"{provider_label} starter engine currently supports text responses only.",
            details={"response_format": str(trial.params.response_format)},
        )
    openai = import_optional("openai", extra=missing_extra)
    extras = dict(trial.model.extras)
    timeout_seconds = float(extras.get("timeout_seconds", 60.0))
    client_kwargs: dict[str, object] = {"timeout": timeout_seconds}
    if base_url is not None:
        client_kwargs["base_url"] = base_url
    api_key = (
        _runtime_secret(runtime, "OPENAI_API_KEY")
        or _runtime_secret(runtime, "OPENAI_COMPAT_API_KEY")
        or extras.get("api_key")
        or "dummy"
    )
    client_kwargs["api_key"] = str(api_key)
    client = openai.OpenAI(**client_kwargs)
    messages = _resolved_messages(trial, context)
    request_kwargs: dict[str, object] = {
        "model": trial.model.model_id,
        "messages": messages,
        "temperature": trial.params.temperature,
        "max_tokens": trial.params.max_tokens,
    }
    if trial.params.top_p is not None:
        request_kwargs["top_p"] = trial.params.top_p
    if trial.params.stop_sequences:
        request_kwargs["stop"] = trial.params.stop_sequences
    if trial.params.seed is not None:
        request_kwargs["seed"] = trial.params.seed & 0xFFFFFFFF
    if trial.params.logprobs is not None:
        request_kwargs["logprobs"] = True
        request_kwargs["top_logprobs"] = trial.params.logprobs
    if trial.params.top_k is not None:
        extra_body = cast(
            dict[str, object], request_kwargs.setdefault("extra_body", {})
        )
        extra_body["top_k"] = trial.params.top_k
    for key, value in trial.params.extras.items():
        extra_body = cast(
            dict[str, object], request_kwargs.setdefault("extra_body", {})
        )
        extra_body[key] = value

    start = perf_counter()
    try:
        response = client.chat.completions.create(**request_kwargs)
    except openai.AuthenticationError as exc:
        raise InferenceError(
            code=ErrorCode.PROVIDER_AUTH,
            message=f"{provider_label} rejected authentication.",
            details={"body": getattr(exc, "body", None)},
        ) from exc
    except openai.RateLimitError as exc:
        raise RetryableProviderError(
            code=ErrorCode.PROVIDER_RATE_LIMIT,
            message=f"{provider_label} rate limited the request.",
            details={"body": getattr(exc, "body", None)},
        ) from exc
    except openai.APIConnectionError as exc:
        raise RetryableProviderError(
            code=ErrorCode.PROVIDER_UNAVAILABLE,
            message=f"Could not reach {provider_label}: {exc}",
        ) from exc
    except openai.APIStatusError as exc:
        error_cls = RetryableProviderError if exc.status_code >= 500 else InferenceError
        raise error_cls(
            code=ErrorCode.PROVIDER_UNAVAILABLE,
            message=f"{provider_label} returned HTTP {exc.status_code}.",
            details={"body": getattr(exc, "body", None)},
        ) from exc

    choice = response.choices[0] if getattr(response, "choices", None) else None
    if choice is None or getattr(choice, "message", None) is None:
        raise InferenceError(
            code=ErrorCode.PARSE_ERROR,
            message=f"{provider_label} returned no message choices.",
            details={"provider_request_id": getattr(response, "id", None)},
        )
    latency_ms = (perf_counter() - start) * 1000
    usage = getattr(response, "usage", None)
    return InferenceResult(
        inference=InferenceRecord(
            spec_hash=f"inference_{trial.item_id}",
            raw_text=_coerce_message_text(choice.message.content),
            finish_reason=getattr(choice, "finish_reason", None),
            latency_ms=latency_ms,
            provider_request_id=getattr(response, "id", None),
            token_usage=TokenUsage(
                prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
            ),
        )
    )


def _resolved_messages(trial, context: object) -> list[dict[str, object]]:
    context_messages = _prompt_messages_from_context(context)
    if context_messages:
        return [dict(message) for message in context_messages]
    return [message.model_dump(mode="json") for message in trial.prompt.messages]


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


def _build_judge_spec(*, model_id: str, provider: str):
    return JudgeInferenceSpec(
        model=ModelSpec(
            model_id=model_id,
            provider=provider,
            extras=_provider_model_extras(provider),
        ),
        params=InferenceParamsSpec(max_tokens=8192, temperature=0.0),
    )


def _normalize_provider_name(provider: str) -> str:
    return provider.replace("-", "_")


def _provider_model_extras(provider: str) -> JSONDict:
    normalized = _normalize_provider_name(provider)
    if normalized == "openai_compatible":
        return {
            "base_url": "http://127.0.0.1:8000/v1",
            "timeout_seconds": 60.0,
        }
    return {}


def _context_item_id(context: object) -> str:
    item_id = getattr(context, "item_id", None)
    if item_id is not None:
        return str(item_id)
    if hasattr(context, "get"):
        resolved = context.get("item_id")  # type: ignore[attr-defined]
        if resolved is not None:
            return str(resolved)
    return "item"


def _expected_text(context: object) -> str:
    if hasattr(context, "get"):
        for key in ("expected", "answer", "answer_letter"):
            resolved = context.get(key)  # type: ignore[attr-defined]
            if resolved is not None:
                return _coerce_text(resolved)
    return ""


def _expected_demo_response(context: object) -> str:
    if hasattr(context, "get"):
        for key in ("judge_expected_response", "expected_response"):
            resolved = context.get(key)  # type: ignore[attr-defined]
            if isinstance(resolved, str):
                return resolved
    return _expected_text(context)


def _coerce_float(value: object) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (str, int, float)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, sort_keys=True)


def _coerce_message_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
        return "".join(text_parts)
    return ""


def _runtime_secret(runtime, key: str) -> str | None:
    secrets = getattr(runtime, "secrets", {}) or {}
    value = secrets.get(key)
    if value is None:
        return None
    if hasattr(value, "get_secret_value"):
        return value.get_secret_value()
    return str(value)


def _extract_lpfqa_reference_answer(text: str) -> str:
    match = _LPFQA_REFERENCE_PATTERN.search(text)
    if match is None:
        return text
    return match.group("answer").strip()


def _extract_hle_answer(text: str) -> str | None:
    match = _HLE_ANSWER_PATTERN.search(text)
    if match is None:
        return None
    return match.group("value").strip()


def _extract_hle_confidence(text: str) -> int:
    match = _HLE_CONFIDENCE_PATTERN.search(text)
    if match is None:
        return 100
    return int(match.group("value"))


def _coerce_json_dict(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return {str(key): value[key] for key in value}
    return {}


def _prompt_messages_with_optional_system(
    system_prompt: str,
    user_prompt: str,
) -> list[PromptMessage]:
    messages: list[PromptMessage] = []
    if system_prompt:
        messages.append(PromptMessage(role=PromptRole.SYSTEM, content=system_prompt))
    messages.append(PromptMessage(role=PromptRole.USER, content=user_prompt))
    return messages


def _parse_simpleqa_grade(
    text: str,
) -> Literal["CORRECT", "INCORRECT", "NOT_ATTEMPTED"]:
    match = _SIMPLEQA_GRADE_PATTERN.search(text)
    if match is None:
        return "NOT_ATTEMPTED"
    token = match.group(1).upper().replace(" ", "_")
    return {
        "A": "CORRECT",
        "B": "INCORRECT",
        "C": "NOT_ATTEMPTED",
        "NOT_ATTEMPTED": "NOT_ATTEMPTED",
        "CORRECT": "CORRECT",
        "INCORRECT": "INCORRECT",
    }.get(token, "NOT_ATTEMPTED")


def _simpleqa_demo_grade(question: str, target: str, predicted_answer: str) -> str:
    del question
    normalized_target = _normalize_text(target)
    normalized_predicted = _normalize_text(predicted_answer)
    if not normalized_predicted:
        return "C"
    if normalized_predicted == normalized_target:
        return "A"
    return "B"
