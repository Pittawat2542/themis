"""Agentic runner with tool use support.

This module provides a runner that supports agentic workflows where
models can call tools/functions to augment their capabilities.

Examples:
    from themis.generation import agentic_runner
    from themis.core import tools, entities

    # Create registry with tools
    registry = tools.ToolRegistry()
    registry.register(tools.create_calculator_tool())

    # Create runner
    runner = agentic_runner.AgenticRunner(
        provider=provider,
        tool_registry=registry,
        max_iterations=10
    )

    # Create task
    task = entities.GenerationTask(...)

    # Run with tool use
    record = runner.run_agentic(task)
    print(f"Used {len(record.iterations)} iterations")
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from themis.core import conversation as conv
from themis.core import entities as core_entities
from themis.core import tools as tool_primitives
from themis.interfaces import ModelProvider
from themis.utils import tracing

logger = logging.getLogger(__name__)


@dataclass
class AgenticIteration:
    """Single iteration in an agentic workflow.

    Attributes:
        iteration_number: Iteration index (0-based)
        generation_record: Model generation for this iteration
        tool_calls: Tool calls extracted from generation
        tool_results: Results from executing tools
        context_snapshot: Conversation context at this iteration
    """

    iteration_number: int
    generation_record: core_entities.GenerationRecord
    tool_calls: list[tool_primitives.ToolCall] = field(default_factory=list)
    tool_results: list[tool_primitives.ToolResult] = field(default_factory=list)
    context_snapshot: conv.ConversationContext | None = None


@dataclass
class AgenticRecord:
    """Complete record of an agentic workflow execution.

    Attributes:
        task: Original generation task
        final_output: Final model output
        iterations: List of iterations executed
        context: Conversation context (if used)
        metadata: Additional metadata
    """

    task: core_entities.GenerationTask
    final_output: core_entities.ModelOutput | None
    iterations: list[AgenticIteration] = field(default_factory=list)
    context: conv.ConversationContext | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def total_iterations(self) -> int:
        """Get total number of iterations.

        Returns:
            Number of iterations
        """
        return len(self.iterations)

    def total_tool_calls(self) -> int:
        """Get total number of tool calls across all iterations.

        Returns:
            Total tool calls
        """
        return sum(len(it.tool_calls) for it in self.iterations)

    def successful_tool_calls(self) -> int:
        """Get number of successful tool calls.

        Returns:
            Number of successful tool executions
        """
        count = 0
        for iteration in self.iterations:
            count += sum(1 for result in iteration.tool_results if result.is_success())
        return count


class AgenticRunner:
    """Runner supporting tool use and agentic workflows.

    This runner executes an agentic loop where the model can make tool calls,
    receive results, and continue processing until completion or max iterations.

    Attributes:
        provider: Model provider for generation
        tool_registry: Registry of available tools
        max_iterations: Maximum number of iterations
        tool_call_parser: Function to parse tool calls from model output
    """

    def __init__(
        self,
        *,
        provider: ModelProvider,
        tool_registry: tool_primitives.ToolRegistry,
        max_iterations: int = 10,
        tool_call_parser: Callable[[str], list[tool_primitives.ToolCall]] | None = None,
    ):
        """Initialize agentic runner.

        Args:
            provider: Model provider for generation
            tool_registry: Registry of available tools
            max_iterations: Maximum number of iterations
            tool_call_parser: Optional custom parser for tool calls
        """
        self._provider = provider
        self._tools = tool_registry
        self._max_iterations = max_iterations
        self._tool_call_parser = tool_call_parser or self._default_tool_call_parser

    def run_agentic(self, task: core_entities.GenerationTask) -> AgenticRecord:
        """Run agentic loop with tool use.

        Args:
            task: Generation task to execute

        Returns:
            AgenticRecord with full iteration history
        """
        task_id = task.metadata.get("dataset_id", "unknown")

        with tracing.span(
            "run_agentic",
            task_id=task_id,
            model=task.model.identifier,
            max_iterations=self._max_iterations,
        ):
            # Initialize conversation context
            context = conv.ConversationContext()
            context.add_message("user", task.prompt.text)

            # Add system message with tool descriptions
            tool_descriptions = self._format_tool_descriptions()
            if tool_descriptions:
                context.add_message("system", tool_descriptions)

            iterations: list[AgenticIteration] = []

            for i in range(self._max_iterations):
                with tracing.span("agentic_iteration", iteration=i):
                    logger.debug(
                        "Starting agentic iteration %d/%d", i + 1, self._max_iterations
                    )

                    # Generate with current context
                    with tracing.span("generate"):
                        prompt_text = context.to_prompt()
                        gen_task = self._update_task_prompt(task, prompt_text, i)
                        record = self._provider.generate(gen_task)

                    # Parse tool calls from output
                    with tracing.span("parse_tool_calls"):
                        tool_calls = self._parse_tool_calls(record)

                    # If no tool calls, we're done
                    if not tool_calls:
                        logger.debug("No tool calls found, ending agentic loop")
                        iteration = AgenticIteration(
                            iteration_number=i,
                            generation_record=record,
                            tool_calls=[],
                            tool_results=[],
                            context_snapshot=self._snapshot_context(context),
                        )
                        iterations.append(iteration)

                        # Add final assistant message
                        if record.output:
                            context.add_message("assistant", record.output.text)

                        break

                    # Execute tool calls
                    with tracing.span("execute_tools", num_tools=len(tool_calls)):
                        tool_results = self._execute_tools(tool_calls)

                    # Create iteration record
                    iteration = AgenticIteration(
                        iteration_number=i,
                        generation_record=record,
                        tool_calls=tool_calls,
                        tool_results=tool_results,
                        context_snapshot=self._snapshot_context(context),
                    )
                    iterations.append(iteration)

                    # Add assistant response to context
                    if record.output:
                        context.add_message("assistant", record.output.text)

                    # Add tool results to context
                    for result in tool_results:
                        result_text = self._format_tool_result(result)
                        context.add_message("tool", result_text)

                    logger.debug(
                        "Iteration %d: %d tool calls, %d successful",
                        i,
                        len(tool_calls),
                        sum(1 for r in tool_results if r.is_success()),
                    )

            # Determine final output
            final_output = None
            if iterations:
                final_output = iterations[-1].generation_record.output

            # Create agentic record
            record = AgenticRecord(
                task=task,
                final_output=final_output,
                iterations=iterations,
                context=context,
                metadata={
                    "total_iterations": len(iterations),
                    "total_tool_calls": sum(len(it.tool_calls) for it in iterations),
                    "max_iterations_reached": len(iterations) >= self._max_iterations,
                },
            )

            logger.info(
                "Agentic execution completed: %d iterations, %d tool calls",
                len(iterations),
                record.total_tool_calls(),
            )

            return record

    def _format_tool_descriptions(self) -> str:
        """Format tool descriptions for system message.

        Returns:
            Formatted tool descriptions
        """
        tools = self._tools.list_tools()
        if not tools:
            return ""

        lines = ["Available tools:"]
        for tool in tools:
            lines.append(f"\n- {tool.name}: {tool.description}")
            lines.append(f"  Parameters: {json.dumps(tool.parameters, indent=2)}")

        lines.append(
            '\nTo use a tool, output: TOOL_CALL: {"name": "tool_name", "arguments": {...}}'
        )

        return "\n".join(lines)

    def _parse_tool_calls(
        self, record: core_entities.GenerationRecord
    ) -> list[tool_primitives.ToolCall]:
        """Parse tool calls from generation record.

        Args:
            record: Generation record

        Returns:
            List of parsed tool calls
        """
        if not record.output:
            return []

        return self._tool_call_parser(record.output.text)

    def _default_tool_call_parser(self, text: str) -> list[tool_primitives.ToolCall]:
        """Default parser for tool calls.

        Looks for lines like: TOOL_CALL: {"name": "...", "arguments": {...}}

        Args:
            text: Model output text

        Returns:
            List of parsed tool calls
        """
        calls = []

        # Look for TOOL_CALL: {...} pattern
        pattern = r"TOOL_CALL:\s*(\{.*?\})"
        matches = re.finditer(pattern, text, re.DOTALL)

        for match in matches:
            try:
                data = json.loads(match.group(1))
                if "name" in data and "arguments" in data:
                    call = tool_primitives.ToolCall(
                        tool_name=data["name"],
                        arguments=data["arguments"],
                    )
                    calls.append(call)
            except json.JSONDecodeError:
                logger.warning("Failed to parse tool call JSON: %s", match.group(1))
                continue

        return calls

    def _execute_tools(
        self, calls: list[tool_primitives.ToolCall]
    ) -> list[tool_primitives.ToolResult]:
        """Execute tool calls.

        Args:
            calls: Tool calls to execute

        Returns:
            List of tool results
        """
        results = []
        for call in calls:
            with tracing.span("execute_tool", tool_name=call.tool_name):
                result = self._tools.execute(call)
                results.append(result)

        return results

    def _format_tool_result(self, result: tool_primitives.ToolResult) -> str:
        """Format tool result for context.

        Args:
            result: Tool result

        Returns:
            Formatted string
        """
        if result.is_success():
            return f"Tool {result.call.tool_name} result: {result.result}"
        else:
            return f"Tool {result.call.tool_name} error: {result.error}"

    def _update_task_prompt(
        self,
        task: core_entities.GenerationTask,
        prompt_text: str,
        iteration: int,
    ) -> core_entities.GenerationTask:
        """Update task with new prompt text.

        Args:
            task: Original task
            prompt_text: New prompt text
            iteration: Iteration number

        Returns:
            Updated task
        """
        from themis.core.entities import PromptRender, PromptSpec

        new_prompt = PromptRender(
            spec=PromptSpec(
                name=f"agentic_iter_{iteration}",
                template="",
                metadata={"iteration": iteration},
            ),
            text=prompt_text,
            context={"iteration": iteration},
            metadata={"iteration": iteration},
        )

        metadata = dict(task.metadata)
        metadata["iteration"] = iteration
        metadata["agentic"] = True

        return core_entities.GenerationTask(
            prompt=new_prompt,
            model=task.model,
            sampling=task.sampling,
            metadata=metadata,
            reference=task.reference,
        )

    def _snapshot_context(
        self, context: conv.ConversationContext
    ) -> conv.ConversationContext:
        """Create snapshot of context.

        Args:
            context: Context to snapshot

        Returns:
            Copy of context
        """
        return conv.ConversationContext.from_dict(context.to_dict())


__all__ = ["AgenticIteration", "AgenticRecord", "AgenticRunner"]
