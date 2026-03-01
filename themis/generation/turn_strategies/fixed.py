"""Fixed sequence turn strategy."""

from __future__ import annotations

from dataclasses import dataclass

from themis.core import conversation as conv
from themis.core import entities as core_entities


@dataclass
class FixedSequenceTurnStrategy:
    """Pre-determined sequence of user messages."""

    messages: list[str]
    _index: int = 0

    def next_turn(
        self,
        context: conv.ConversationContext,
        last_record: core_entities.GenerationRecord,
    ) -> str | None:
        """Return next message from sequence."""
        if self._index >= len(self.messages):
            return None

        message = self.messages[self._index]
        self._index += 1
        return message

    def reset(self) -> None:
        """Reset strategy to beginning of sequence."""
        self._index = 0
