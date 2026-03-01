"""Chained turn strategy."""

from __future__ import annotations

from dataclasses import dataclass

from themis.core import conversation as conv
from themis.core import entities as core_entities
from themis.generation.turn_strategies.base import TurnStrategy


@dataclass
class ChainedTurnStrategy:
    """Chain multiple strategies together."""

    strategies: list[TurnStrategy]

    def next_turn(
        self,
        context: conv.ConversationContext,
        last_record: core_entities.GenerationRecord,
    ) -> str | None:
        """Try each strategy until one returns a message."""
        for strategy in self.strategies:
            message = strategy.next_turn(context, last_record)
            if message is not None:
                return message

        return None
