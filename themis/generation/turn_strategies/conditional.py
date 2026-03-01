"""Conditional turn strategy."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from themis.core import conversation as conv
from themis.core import entities as core_entities


@dataclass
class ConditionalTurnStrategy:
    """Choose next message based on conditions."""

    conditions: list[
        tuple[
            Callable[[conv.ConversationContext, core_entities.GenerationRecord], bool],
            str | None,
        ]
    ]
    default: str | None = None

    def next_turn(
        self,
        context: conv.ConversationContext,
        last_record: core_entities.GenerationRecord,
    ) -> str | None:
        """Evaluate conditions and return matching message."""
        for condition, message in self.conditions:
            try:
                if condition(context, last_record):
                    return message
            except Exception:
                # Skip conditions that fail
                continue

        return self.default
