"""Conversation runner for multi-turn interactions.

This module provides a runner that executes multi-turn conversations
using turn strategies to determine the flow of the conversation.

Examples:
    from themis.generation import conversation_runner, turn_strategies
    from themis.core import conversation, entities

    # Create provider and strategy
    provider = FakeProvider()
    strategy = turn_strategies.FixedSequenceTurnStrategy([
        "What is 2+2?",
        "What about 3+3?"
    ])

    # Create runner
    runner = conversation_runner.ConversationRunner(
        provider=provider,
        turn_strategy=strategy,
        max_turns=5
    )

    # Create conversation task
    context = conversation.ConversationContext()
    context.add_message("system", "You are a math tutor.")

    task = conversation.ConversationTask(
        context=context,
        model=model_spec,
        sampling=sampling_config
    )

    # Run conversation
    record = runner.run_conversation(task)
    print(f"Conversation had {record.total_turns()} turns")
"""

from __future__ import annotations

import logging
from typing import Any

from themis.core import conversation as conv
from themis.core import entities as core_entities
from themis.generation import turn_strategies
from themis.interfaces import ModelProvider
from themis.utils import tracing

logger = logging.getLogger(__name__)


class ConversationRunner:
    """Runner for executing multi-turn conversations.

    This runner manages the conversation loop, generating responses
    and determining next turns using a TurnStrategy.

    Attributes:
        provider: Model provider for generation
        turn_strategy: Strategy for determining next turns
        max_turns: Maximum number of conversation turns
        prompt_template: Optional template for formatting context
    """

    def __init__(
        self,
        *,
        provider: ModelProvider,
        turn_strategy: turn_strategies.TurnStrategy,
        max_turns: int = 10,
        prompt_template: Any | None = None,
    ):
        """Initialize conversation runner.

        Args:
            provider: Model provider for generation
            turn_strategy: Strategy for determining next turns
            max_turns: Maximum number of conversation turns
            prompt_template: Optional template for formatting context
        """
        self._provider = provider
        self._turn_strategy = turn_strategy
        self._max_turns = max_turns
        self._prompt_template = prompt_template

    def run_conversation(self, task: conv.ConversationTask) -> conv.ConversationRecord:
        """Execute a multi-turn conversation.

        Args:
            task: Conversation task to execute

        Returns:
            ConversationRecord with full conversation history
        """
        with tracing.span(
            "run_conversation",
            model=task.model.identifier,
            max_turns=task.max_turns,
        ):
            turns: list[conv.ConversationTurn] = []
            context = task.context
            max_turns = min(task.max_turns, self._max_turns)

            for turn_num in range(max_turns):
                with tracing.span("conversation_turn", turn=turn_num):
                    logger.debug(
                        "Starting conversation turn %d/%d", turn_num + 1, max_turns
                    )

                    # Generate response for current context
                    with tracing.span("generate_response"):
                        prompt_text = context.to_prompt(self._prompt_template)
                        generation_task = self._create_generation_task(
                            task, prompt_text, turn_num
                        )
                        record = self._provider.generate(generation_task)

                    # Add assistant response to context
                    if record.output:
                        context.add_message("assistant", record.output.text)
                    else:
                        # Generation failed
                        logger.warning(
                            "Generation failed at turn %d: %s",
                            turn_num,
                            record.error.message if record.error else "unknown error",
                        )

                    # Create turn record (no user message yet)
                    turn = conv.ConversationTurn(
                        turn_number=turn_num,
                        user_message=None,
                        generation_record=record,
                        context_snapshot=self._snapshot_context(context),
                    )
                    turns.append(turn)

                    # Check stop conditions
                    if task.should_stop():
                        logger.debug("Task stop condition met at turn %d", turn_num)
                        break

                    # Determine next turn
                    with tracing.span("plan_next_turn"):
                        next_message = self._turn_strategy.next_turn(context, record)

                    if next_message is None:
                        logger.debug(
                            "Turn strategy ended conversation at turn %d", turn_num
                        )
                        break

                    # Add user message for next turn
                    user_msg = conv.Message(role="user", content=next_message)
                    context.add_message("user", next_message)
                    turn.user_message = user_msg

                    logger.debug(
                        "Planned next turn: %s",
                        next_message[:50] + ("..." if len(next_message) > 50 else ""),
                    )

            # Create conversation record
            record = conv.ConversationRecord(
                task=task,
                context=context,
                turns=turns,
                metadata={
                    "total_turns": len(turns),
                    "max_turns_reached": len(turns) >= max_turns,
                    "stop_condition_met": task.should_stop(),
                },
            )

            logger.info(
                "Conversation completed: %d turns, stop_reason=%s",
                len(turns),
                "max_turns" if record.metadata["max_turns_reached"] else "strategy",
            )

            return record

    def _create_generation_task(
        self, conv_task: conv.ConversationTask, prompt_text: str, turn_num: int
    ) -> core_entities.GenerationTask:
        """Create a generation task from conversation state.

        Args:
            conv_task: Conversation task
            prompt_text: Rendered prompt text
            turn_num: Current turn number

        Returns:
            GenerationTask for this turn
        """
        from themis.core.entities import PromptRender, PromptSpec

        prompt_render = PromptRender(
            spec=PromptSpec(
                name=f"conversation_turn_{turn_num}",
                template="",
                metadata={"turn": turn_num},
            ),
            text=prompt_text,
            context={"turn": turn_num},
            metadata={"turn": turn_num},
        )

        metadata = dict(conv_task.metadata)
        metadata["turn"] = turn_num
        metadata["conversation"] = True

        return core_entities.GenerationTask(
            prompt=prompt_render,
            model=conv_task.model,
            sampling=conv_task.sampling,
            metadata=metadata,
            reference=conv_task.reference,
        )

    def _snapshot_context(
        self, context: conv.ConversationContext
    ) -> conv.ConversationContext:
        """Create a snapshot of conversation context.

        Args:
            context: Context to snapshot

        Returns:
            Copy of context
        """
        return conv.ConversationContext.from_dict(context.to_dict())


__all__ = ["ConversationRunner"]
