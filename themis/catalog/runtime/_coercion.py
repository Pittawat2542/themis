"""Generic coercion and benchmark parsing helpers for catalog runtime."""

from __future__ import annotations

import json
import re
from typing import Literal

from themis.extractors.builtin import _normalize_text

_SIMPLEQA_GRADE_PATTERN = re.compile(
    r"\b(CORRECT|INCORRECT|NOT[_ ]ATTEMPTED|A|B|C)\b", flags=re.IGNORECASE
)
_LPFQA_REFERENCE_PATTERN = re.compile(
    r"<参考答案>[：:]\s*(?P<answer>.*?)\s*(?:<评估要点>|$)",
    flags=re.DOTALL,
)
_HLE_ANSWER_PATTERN = re.compile(r"(?im)^answer:\s*(?P<value>.+)$")
_HLE_CONFIDENCE_PATTERN = re.compile(r"(?im)^confidence:\s*(?P<value>\d+)")


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


def _coerce_usage_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, str)):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return None


def _parse_simpleqa_grade(
    text: str,
) -> Literal["CORRECT", "INCORRECT", "NOT_ATTEMPTED"]:
    match = _SIMPLEQA_GRADE_PATTERN.search(text)
    if match is None:
        return "NOT_ATTEMPTED"
    token = match.group(1).upper().replace(" ", "_")
    if token in {"A", "CORRECT"}:
        return "CORRECT"
    if token in {"B", "INCORRECT"}:
        return "INCORRECT"
    if token in {"C", "NOT_ATTEMPTED"}:
        return "NOT_ATTEMPTED"
    return "NOT_ATTEMPTED"


def _simpleqa_demo_grade(question: str, target: str, predicted_answer: str) -> str:
    del question
    normalized_target = _normalize_text(target)
    normalized_predicted = _normalize_text(predicted_answer)
    if not normalized_predicted:
        return "C"
    if normalized_predicted == normalized_target:
        return "A"
    return "B"
