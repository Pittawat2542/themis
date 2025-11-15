"""Turn strategies for multi-turn conversations.

This module provides strategies for determining the next turn in a conversation.
Strategies can be fixed (predefined sequences), dynamic (generated based on context),
or interactive.

Examples:
    # Fixed sequence
    strategy = FixedSequenceTurnStrategy([
        "What is 2+2?",
        "What about 3+3?",
        "And 5+5?"
    ])

    # Dynamic strategy
    def planner(context):
        if len(context) < 2:
            return "Can you explain more?"
        return None  # Stop

    strategy = DynamicTurnStrategy(planner)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

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
        """Determine the next user message.

        Args:
            context: Current conversation context
            last_record: Last generation record

        Returns:
            Next user message, or None to end conversation
        """
        ...


@dataclass
class FixedSequenceTurnStrategy:
    """Pre-determined sequence of user messages.

    This strategy iterates through a fixed list of user messages,
    useful for scripted conversations or testing.

    Examples:
        strategy = FixedSequenceTurnStrategy([
            "Hello!",
            "How are you?",
            "Goodbye!"
        ])
    """

    messages: list[str]
    _index: int = 0

    def next_turn(
        self,
        context: conv.ConversationContext,
        last_record: core_entities.GenerationRecord,
    ) -> str | None:
        """Return next message from sequence.

        Args:
            context: Current conversation context
            last_record: Last generation record

        Returns:
            Next message or None if sequence exhausted
        """
        if self._index >= len(self.messages):
            return None

        message = self.messages[self._index]
        self._index += 1
        return message

    def reset(self) -> None:
        """Reset strategy to beginning of sequence."""
        self._index = 0


@dataclass
class DynamicTurnStrategy:
    """Generate next message based on conversation state.

    This strategy uses a function to dynamically determine the next
    user message based on the conversation context.

    Examples:
        def planner(context, record):
            outputs = [msg.content for msg in context.get_messages_by_role("assistant")]
            if "error" in outputs[-1].lower():
                return "Can you try again?"
            elif len(context) >= 10:
                return None  # Stop after 10 messages
            else:
                return "Please continue."

        strategy = DynamicTurnStrategy(planner)
    """

    planner: Callable[
        [conv.ConversationContext, core_entities.GenerationRecord], str | None
    ]

    def next_turn(
        self,
        context: conv.ConversationContext,
        last_record: core_entities.GenerationRecord,
    ) -> str | None:
        """Generate next message using planner function.

        Args:
            context: Current conversation context
            last_record: Last generation record

        Returns:
            Next message or None to stop
        """
        return self.planner(context, last_record)


@dataclass
class RepeatUntilSuccessTurnStrategy:
    """Repeat the same question until getting a successful response.

    This strategy is useful for testing robustness or debugging.

    Examples:
        strategy = RepeatUntilSuccessTurnStrategy(
            question="What is 2+2?",
            success_checker=lambda output: "4" in output,
            max_attempts=5
        )
    """

    question: str
    success_checker: Callable[[str], bool]
    max_attempts: int = 5
    _attempts: int = 0

    def next_turn(
        self,
        context: conv.ConversationContext,
        last_record: core_entities.GenerationRecord,
    ) -> str | None:
        """Repeat question until success or max attempts.

        Args:
            context: Current conversation context
            last_record: Last generation record

        Returns:
            Question or None if success/max attempts reached
        """
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


@dataclass
class ConditionalTurnStrategy:
    """Choose next message based on conditions.

    This strategy evaluates conditions and returns different messages
    based on which condition matches.

    Examples:
        strategy = ConditionalTurnStrategy(
            conditions=[
                (lambda ctx, rec: "error" in rec.output.text.lower(), "Please try again."),
                (lambda ctx, rec: len(ctx) >= 5, None),  # Stop after 5 turns
            ],
            default="Continue."
        )
    """

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
        """Evaluate conditions and return matching message.

        Args:
            context: Current conversation context
            last_record: Last generation record

        Returns:
            Message from first matching condition, or default
        """
        for condition, message in self.conditions:
            try:
                if condition(context, last_record):
                    return message
            except Exception:
                # Skip conditions that fail
                continue

        return self.default


@dataclass
class ChainedTurnStrategy:
    """Chain multiple strategies together.

    This strategy tries strategies in sequence until one returns
    a non-None message.

    Examples:
        strategy = ChainedTurnStrategy([
            FixedSequenceTurnStrategy(["Hello", "How are you?"]),
            DynamicTurnStrategy(lambda ctx, rec: "Goodbye" if len(ctx) > 5 else None)
        ])
    """

    strategies: list[TurnStrategy]

    def next_turn(
        self,
        context: conv.ConversationContext,
        last_record: core_entities.GenerationRecord,
    ) -> str | None:
        """Try each strategy until one returns a message.

        Args:
            context: Current conversation context
            last_record: Last generation record

        Returns:
            First non-None message, or None if all return None
        """
        for strategy in self.strategies:
            message = strategy.next_turn(context, last_record)
            if message is not None:
                return message

        return None


# Helper functions for creating common strategies


def create_qa_strategy(questions: list[str]) -> FixedSequenceTurnStrategy:
    """Create a simple Q&A strategy from a list of questions.

    Args:
        questions: List of questions to ask

    Returns:
        FixedSequenceTurnStrategy with questions
    """
    return FixedSequenceTurnStrategy(messages=questions)


def create_max_turns_strategy(
    max_turns: int, message: str = "Continue."
) -> DynamicTurnStrategy:
    """Create strategy that stops after max turns.

    Args:
        max_turns: Maximum number of turns
        message: Message to send each turn

    Returns:
        DynamicTurnStrategy that stops after max_turns
    """

    def planner(
        context: conv.ConversationContext, record: core_entities.GenerationRecord
    ) -> str | None:
        if len(context) >= max_turns:
            return None
        return message

    return DynamicTurnStrategy(planner=planner)


def create_keyword_stop_strategy(
    keywords: list[str], message: str = "Continue."
) -> DynamicTurnStrategy:
    """Create strategy that stops when any keyword appears in response.

    Args:
        keywords: List of keywords to trigger stop
        message: Message to send each turn

    Returns:
        DynamicTurnStrategy that stops on keywords
    """

    def planner(
        context: conv.ConversationContext, record: core_entities.GenerationRecord
    ) -> str | None:
        if record.output:
            text_lower = record.output.text.lower()
            if any(kw.lower() in text_lower for kw in keywords):
                return None
        return message

    return DynamicTurnStrategy(planner=planner)


# Prompt perturbation and seed helpers for robustness sweeps

import random


def set_sampling_seed(task_metadata: dict[str, object], seed: int) -> dict[str, object]:
    """Attach a deterministic seed to task metadata for providers that support it.

    This does not enforce provider behavior but offers a convention: 'sampling_seed'.
    """
    md = dict(task_metadata)
    md["sampling_seed"] = int(seed)
    return md


def perturb_prompt(text: str, *, seed: int | None = None, max_changes: int = 2) -> str:
    """Apply small, semantics-preserving perturbations to a prompt.

    Changes include optional punctuation tweaks and inserting polite filler words.
    """
    rng = random.Random(seed)
    t = text
    changes = 0
    # Optional punctuation swap
    if "?" in t and changes < max_changes and rng.random() < 0.5:
        t = t.replace("?", "??", 1)
        changes += 1
    # Optional polite filler insertion
    fillers = ["please", "kindly", "if possible"]
    if changes < max_changes and rng.random() < 0.5:
        words = t.split()
        if words:
            idx = rng.randint(0, len(words) - 1)
            words.insert(idx, rng.choice(fillers))
            t = " ".join(words)
            changes += 1
    return t


def create_prompt_variants(base_text: str, *, count: int, seed: int) -> list[str]:
    """Create multiple perturbed variants of a base prompt with deterministic seeding."""
    rng = random.Random(seed)
    return [
        perturb_prompt(base_text, seed=rng.randint(0, 1_000_000))
        for _ in range(max(1, count))
    ]
