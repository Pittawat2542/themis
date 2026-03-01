"""Common stop conditions for conversations."""

from __future__ import annotations

from collections.abc import Callable

from themis.core.conversation.context import ConversationContext


def stop_on_keyword(keyword: str) -> Callable[[ConversationContext], bool]:
    """Create stop condition that triggers when keyword appears."""

    def condition(context: ConversationContext) -> bool:
        if not context.messages:
            return False
        last_msg = context.messages[-1]
        if last_msg.role == "assistant":
            return keyword.lower() in last_msg.content.lower()
        return False

    return condition


def stop_on_pattern(
    pattern: str,
) -> Callable[[ConversationContext], bool]:
    """Create stop condition that triggers when regex pattern matches."""
    import re

    compiled = re.compile(pattern, re.IGNORECASE)

    def condition(context: ConversationContext) -> bool:
        if not context.messages:
            return False
        last_msg = context.messages[-1]
        if last_msg.role == "assistant":
            return compiled.search(last_msg.content) is not None
        return False

    return condition


def stop_on_empty_response() -> Callable[[ConversationContext], bool]:
    """Create stop condition that triggers on empty assistant response."""

    def condition(context: ConversationContext) -> bool:
        if not context.messages:
            return False
        last_msg = context.messages[-1]
        if last_msg.role == "assistant":
            return not last_msg.content.strip()
        return False

    return condition
