"""Repeat turn strategy."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from themis.core import conversation as conv
from themis.core import entities as core_entities


@dataclass
class RepeatUntilSuccessTurnStrategy:
    """Repeat the same question until getting a successful response."""

    question: str
    success_checker: Callable[[str], bool]
    max_attempts: int = 5
    _attempts: int = 0

    def next_turn(
        self,
        context: conv.ConversationContext,
        last_record: core_entities.GenerationRecord,
    ) -> str | None:
        """Repeat question until success or max attempts."""
        # Check if this is first turn
        if self._attempts == 0:
            self._attempts += 1
            return self.question

        # Check if last response was successful
        if last_record.output:
            if self.success_checker(last_record.output.text):
                return None  # Success, stop

        # Check if we've exhausted attempts
        if self._attempts >= self.max_attempts:
            return None  # Give up

        self._attempts += 1
        return self.question

    def reset(self) -> None:
        """Reset attempt counter."""
        self._attempts = 0
