"""Dynamic turn strategy."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from themis.core import conversation as conv
from themis.core import entities as core_entities


@dataclass
class DynamicTurnStrategy:
    """Generate next message based on conversation state."""

    planner: Callable[
        [conv.ConversationContext, core_entities.GenerationRecord], str | None
    ]

    def next_turn(
        self,
        context: conv.ConversationContext,
        last_record: core_entities.GenerationRecord,
    ) -> str | None:
        """Generate next message using planner function."""
        return self.planner(context, last_record)
