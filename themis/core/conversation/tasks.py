"""Conversation task primitives."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from themis.core import entities as core_entities
from themis.core.conversation.context import ConversationContext


@dataclass
class ConversationTask:
    """Task for multi-turn conversation execution.

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
        """Check if conversation should stop."""
        if len(self.context) >= self.max_turns:
            return True

        if self.stop_condition is not None:
            return self.stop_condition(self.context)

        return False
