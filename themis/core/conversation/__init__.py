"""Conversation primitives for multi-turn interactions.

This module provides abstractions for managing multi-turn conversations,
enabling research on dialogue systems, debugging interactions, and
agentic workflows.
"""

from themis.core.conversation.messages import Message, MessageRole
from themis.core.conversation.context import ConversationContext
from themis.core.conversation.tasks import ConversationTask
from themis.core.conversation.records import ConversationTurn, ConversationRecord
from themis.core.conversation.stop_conditions import (
    stop_on_keyword,
    stop_on_pattern,
    stop_on_empty_response,
)

__all__ = [
    "MessageRole",
    "Message",
    "ConversationContext",
    "ConversationTask",
    "ConversationTurn",
    "ConversationRecord",
    "stop_on_keyword",
    "stop_on_pattern",
    "stop_on_empty_response",
]
