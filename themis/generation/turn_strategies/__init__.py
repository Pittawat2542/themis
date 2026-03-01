"""Turn strategies for multi-turn conversations."""

from themis.generation.turn_strategies.base import TurnStrategy
from themis.generation.turn_strategies.fixed import FixedSequenceTurnStrategy
from themis.generation.turn_strategies.dynamic import DynamicTurnStrategy
from themis.generation.turn_strategies.repeat import RepeatUntilSuccessTurnStrategy
from themis.generation.turn_strategies.conditional import ConditionalTurnStrategy
from themis.generation.turn_strategies.chained import ChainedTurnStrategy
from themis.generation.turn_strategies.helpers import (
    create_qa_strategy,
    create_max_turns_strategy,
    create_keyword_stop_strategy,
    set_sampling_seed,
    perturb_prompt,
    create_prompt_variants,
)

__all__ = [
    "TurnStrategy",
    "FixedSequenceTurnStrategy",
    "DynamicTurnStrategy",
    "RepeatUntilSuccessTurnStrategy",
    "ConditionalTurnStrategy",
    "ChainedTurnStrategy",
    "create_qa_strategy",
    "create_max_turns_strategy",
    "create_keyword_stop_strategy",
    "set_sampling_seed",
    "perturb_prompt",
    "create_prompt_variants",
]
