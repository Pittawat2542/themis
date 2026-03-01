"""Helper functions for turn strategies."""

from __future__ import annotations

import random

from themis.core import conversation as conv
from themis.core import entities as core_entities
from themis.generation.turn_strategies.fixed import FixedSequenceTurnStrategy
from themis.generation.turn_strategies.dynamic import DynamicTurnStrategy


def create_qa_strategy(questions: list[str]) -> FixedSequenceTurnStrategy:
    """Create a simple Q&A strategy from a list of questions."""
    return FixedSequenceTurnStrategy(messages=questions)


def create_max_turns_strategy(
    max_turns: int, message: str = "Continue."
) -> DynamicTurnStrategy:
    """Create strategy that stops after max turns."""

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
    """Create strategy that stops when any keyword appears in response."""

    def planner(
        context: conv.ConversationContext, record: core_entities.GenerationRecord
    ) -> str | None:
        if record.output:
            text_lower = record.output.text.lower()
            if any(kw.lower() in text_lower for kw in keywords):
                return None
        return message

    return DynamicTurnStrategy(planner=planner)


def set_sampling_seed(task_metadata: dict[str, object], seed: int) -> dict[str, object]:
    """Attach a deterministic seed to task metadata for providers that support it."""
    md = dict(task_metadata)
    md["sampling_seed"] = int(seed)
    return md


def perturb_prompt(text: str, *, seed: int | None = None, max_changes: int = 2) -> str:
    """Apply small, semantics-preserving perturbations to a prompt."""
    rng = random.Random(seed)
    t = text
    changes = 0
    if "?" in t and changes < max_changes and rng.random() < 0.5:
        t = t.replace("?", "??", 1)
        changes += 1
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
