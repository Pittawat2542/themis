"""Conversation primitives for multi-turn interactions.

This module provides abstractions for managing multi-turn conversations,
enabling research on dialogue systems, debugging interactions, and
agentic workflows.

Examples:
    # Create a conversation
    context = ConversationContext()
    context.add_message("user", "What is 2+2?")
    context.add_message("assistant", "2+2 equals 4.")
    context.add_message("user", "What about 3+3?")

    # Convert to prompt
    prompt = context.to_prompt()

    # Get conversation history
    history = context.get_history(max_turns=2)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from themis.core import entities as core_entities
from themis.generation import templates

MessageRole = Literal["system", "user", "assistant", "tool"]


@dataclass
class Message:
    """Single message in a conversation.

    Attributes:
        role: Message role (system/user/assistant/tool)
        content: Message text content
        metadata: Additional metadata (tool calls, timestamps, etc.)
        timestamp: Unix timestamp when message was created
    """

    role: MessageRole
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "role": self.role,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


@dataclass
class ConversationContext:
    """Maintains conversation state across turns.

    This class manages the conversation history and provides utilities
    for rendering conversations as prompts.

    Examples:
        context = ConversationContext()
        context.add_message("system", "You are a helpful assistant.")
        context.add_message("user", "Hello!")
        context.add_message("assistant", "Hi! How can I help you?")

        # Get history
        messages = context.get_history()

        # Render to prompt
        prompt = context.to_prompt()
    """

    messages: list[Message] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: MessageRole, content: str, **metadata: Any) -> None:
        """Add a message to the conversation.

        Args:
            role: Message role (system/user/assistant/tool)
            content: Message text content
            **metadata: Additional metadata to attach to message
        """
        self.messages.append(Message(role=role, content=content, metadata=metadata))

    def get_history(self, max_turns: int | None = None) -> list[Message]:
        """Get conversation history.

        Args:
            max_turns: Maximum number of messages to return (from end)

        Returns:
            List of messages (most recent if limited)
        """
        if max_turns is None:
            return list(self.messages)
        return self.messages[-max_turns:]

    def get_messages_by_role(self, role: MessageRole) -> list[Message]:
        """Get all messages with a specific role.

        Args:
            role: Role to filter by

        Returns:
            List of messages with matching role
        """
        return [msg for msg in self.messages if msg.role == role]

    def to_prompt(self, template: templates.PromptTemplate | None = None) -> str:
        """Render conversation to prompt string.

        Args:
            template: Optional template for custom formatting

        Returns:
            Formatted prompt string
        """
        if template is not None:
            return template.render(messages=self.messages)

        # Default format: role-prefixed messages
        lines = []
        for msg in self.messages:
            lines.append(f"{msg.role}: {msg.content}")

        return "\n\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert conversation to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "messages": [msg.to_dict() for msg in self.messages],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConversationContext:
        """Create conversation from dictionary.

        Args:
            data: Dictionary with messages and metadata

        Returns:
            ConversationContext instance
        """
        context = cls(metadata=data.get("metadata", {}))
        for msg_data in data.get("messages", []):
            context.messages.append(
                Message(
                    role=msg_data["role"],
                    content=msg_data["content"],
                    metadata=msg_data.get("metadata", {}),
                    timestamp=msg_data.get("timestamp", time.time()),
                )
            )
        return context

    def __len__(self) -> int:
        """Return number of messages in conversation."""
        return len(self.messages)


@dataclass
class ConversationTask:
    """Task for multi-turn conversation execution.

    This extends the basic GenerationTask concept to support
    multi-turn conversations with configurable stopping conditions.

    Attributes:
        context: Conversation context with message history
        model: Model to use for generation
        sampling: Sampling configuration
        metadata: Additional metadata
        reference: Optional reference for evaluation
        max_turns: Maximum number of conversation turns
        stop_condition: Optional function to determine when to stop
    """

    context: ConversationContext
    model: core_entities.ModelSpec
    sampling: core_entities.SamplingConfig
    metadata: dict[str, Any] = field(default_factory=dict)
    reference: core_entities.Reference | None = None
    max_turns: int = 10
    stop_condition: Callable[[ConversationContext], bool] | None = None

    def should_stop(self) -> bool:
        """Check if conversation should stop.

        Returns:
            True if stop condition is met or max turns reached
        """
        if len(self.context) >= self.max_turns:
            return True

        if self.stop_condition is not None:
            return self.stop_condition(self.context)

        return False


@dataclass
class ConversationTurn:
    """Single turn in a conversation.

    Attributes:
        turn_number: Turn index (0-based)
        user_message: User message for this turn (if any)
        generation_record: Generation result for this turn
        context_snapshot: Conversation context at this turn
    """

    turn_number: int
    user_message: Message | None
    generation_record: core_entities.GenerationRecord
    context_snapshot: ConversationContext


@dataclass
class ConversationRecord:
    """Complete record of a multi-turn conversation.

    This is the result of running a ConversationTask through
    a ConversationRunner.

    Attributes:
        task: Original conversation task
        context: Final conversation context
        turns: List of turns executed
        metadata: Additional metadata (e.g., total turns, stop reason)
    """

    task: ConversationTask
    context: ConversationContext
    turns: list[ConversationTurn] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_final_output(self) -> core_entities.ModelOutput | None:
        """Get the final model output.

        Returns:
            Last turn's output, or None if no turns
        """
        if not self.turns:
            return None
        return self.turns[-1].generation_record.output

    def get_all_outputs(self) -> list[core_entities.ModelOutput | None]:
        """Get all model outputs from all turns.

        Returns:
            List of outputs (may contain None for failed turns)
        """
        return [turn.generation_record.output for turn in self.turns]

    def total_turns(self) -> int:
        """Get total number of turns executed.

        Returns:
            Number of turns
        """
        return len(self.turns)


# Common stop conditions


def stop_on_keyword(keyword: str) -> Callable[[ConversationContext], bool]:
    """Create stop condition that triggers when keyword appears.

    Args:
        keyword: Keyword to look for in assistant messages

    Returns:
        Stop condition function
    """

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
    """Create stop condition that triggers when regex pattern matches.

    Args:
        pattern: Regex pattern to match

    Returns:
        Stop condition function
    """
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
    """Create stop condition that triggers on empty assistant response.

    Returns:
        Stop condition function
    """

    def condition(context: ConversationContext) -> bool:
        if not context.messages:
            return False
        last_msg = context.messages[-1]
        if last_msg.role == "assistant":
            return not last_msg.content.strip()
        return False

    return condition


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
