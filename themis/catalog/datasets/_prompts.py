"""Prompt-message helpers for catalog datasets."""

from __future__ import annotations

from collections.abc import Mapping

from ._types import CatalogPromptMessage, CatalogRow

_HLE_RESPONSE_TEMPLATE = """Explanation: {explanation}
Answer: {answer}
Confidence: {confidence}%"""


def _prompt_messages_from_payload(payload: CatalogRow) -> list[CatalogPromptMessage]:
    prompt = payload.get("prompt")
    if not isinstance(prompt, list):
        return []
    messages: list[CatalogPromptMessage] = []
    for entry in prompt:
        if not isinstance(entry, dict):
            continue
        role = entry.get("role")
        content = entry.get("content")
        if isinstance(role, str) and _is_prompt_content(content):
            messages.append({"role": role, "content": content})
    return messages


def _prompt_messages_from_context(context: object) -> list[CatalogPromptMessage]:
    if isinstance(context, Mapping):
        payload = context.get("prompt_messages")
        if isinstance(payload, list):
            messages: list[CatalogPromptMessage] = []
            for entry in payload:
                if (
                    isinstance(entry, dict)
                    and isinstance(entry.get("role"), str)
                    and _is_prompt_content(entry.get("content"))
                ):
                    messages.append(
                        {
                            "role": str(entry["role"]),
                            "content": entry["content"],
                        }
                    )
            return messages
    return []


def _is_prompt_content(value: object) -> bool:
    if isinstance(value, str):
        return True
    if not isinstance(value, list):
        return False
    for part in value:
        if not isinstance(part, dict):
            return False
        part_type = part.get("type")
        if part_type == "text" and isinstance(part.get("text"), str):
            continue
        if part_type == "image_url" and isinstance(part.get("image_url"), str):
            continue
        return False
    return True
