"""Base interfaces for turn strategies."""

from __future__ import annotations

from typing import Protocol

from themis.core import conversation as conv
from themis.core import entities as core_entities


class TurnStrategy(Protocol):
    """Strategy for determining the next turn in a conversation.

    A turn strategy decides what the user's next message should be
    based on the current conversation state and the last model response.
    """

    def next_turn(
        self,
        context: conv.ConversationContext,
        last_record: core_entities.GenerationRecord,
    ) -> str | None:
        """Determine the next user message."""
        ...
