"""Conversation turn and record primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from themis.core import entities as core_entities
from themis.core.conversation.messages import Message
from themis.core.conversation.context import ConversationContext
from themis.core.conversation.tasks import ConversationTask


@dataclass
class ConversationTurn:
    """Single turn in a conversation."""

    turn_number: int
    user_message: Message | None
    generation_record: core_entities.GenerationRecord
    context_snapshot: ConversationContext


@dataclass
class ConversationRecord:
    """Complete record of a multi-turn conversation."""

    task: ConversationTask
    context: ConversationContext
    turns: list[ConversationTurn] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_final_output(self) -> core_entities.ModelOutput | None:
        """Get the final model output."""
        if not self.turns:
            return None
        return self.turns[-1].generation_record.output

    def get_all_outputs(self) -> list[core_entities.ModelOutput | None]:
        """Get all model outputs from all turns."""
        return [turn.generation_record.output for turn in self.turns]

    def total_turns(self) -> int:
        """Get total number of turns executed."""
        return len(self.turns)
