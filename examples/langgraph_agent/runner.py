"""Themis runner that wraps a LangGraph agent.

This module provides a custom GenerationRunner that executes LangGraph agents,
allowing agentic workflows to be systematically evaluated using Themis.
"""
from __future__ import annotations

import logging
from typing import Any

from themis.core import entities as core_entities
from themis.generation.runner import GenerationRunner

from .agent import create_agent, AgentState, LANGGRAPH_AVAILABLE

logger = logging.getLogger(__name__)


class LangGraphAgentRunner(GenerationRunner):
    """A custom runner that executes a LangGraph agent.
    
    This runner integrates agentic workflows (multi-step reasoning) with Themis's
    evaluation framework by wrapping a LangGraph state machine.
    
    Args:
        provider: The LLM provider (not used by the agent, but required by base class).
        **kwargs: Additional arguments passed to GenerationRunner.
    
    Raises:
        ImportError: If langgraph is not installed.
    
    Example:
        >>> from themis.generation.clients import FakeMathModelClient
        >>> provider = FakeMathModelClient()
        >>> runner = LangGraphAgentRunner(provider=provider)
        >>> # Use runner in experiment...
    
    Note:
        The agent in this example doesn't actually use the provider - it has
        placeholder logic. In production, you'd pass the provider to agent nodes.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        if not LANGGRAPH_AVAILABLE:
            raise ImportError(
                "langgraph is required for this example. "
                "Install it with: pip install langgraph"
            )
        
        try:
            self.agent = create_agent()
            logger.info("Initialized LangGraph agent")
        except Exception as e:
            logger.error(f"Failed to create agent: {e}")
            raise
    
    def _extract_prompt_text(self, task: core_entities.GenerationTask) -> str:
        """Extract text from task prompt.
        
        Args:
            task: The generation task.
        
        Returns:
            The prompt text as a string.
        """
        if hasattr(task.prompt, 'text'):
            return task.prompt.text
        elif hasattr(task.prompt, 'prompt_text'):
            return task.prompt.prompt_text
        else:
            return str(task.prompt)
        
    def _execute_task(
        self,
        task: core_entities.GenerationTask
    ) -> core_entities.GenerationRecord:
        """Execute the task using the LangGraph agent.
        
        Args:
            task: The generation task containing the problem.
        
        Returns:
            A generation record with the agent's response.
        
        Note:
            This method:
            1. Extracts the problem from the task
            2. Initializes agent state
            3. Runs the LangGraph agent
            4. Formats the output for Themis
        """
        # Extract problem text from prompt
        problem = self._extract_prompt_text(task)
        logger.debug(f"Running agent on problem: {problem[:100]}...")
        
        # Initialize agent state
        initial_state: AgentState = {
            "messages": [],
            "problem": problem,
            "plan": "",
            "answer": "",
            "step_count": 0
        }
        
        # Run the agent
        try:
            final_state = self.agent.invoke(initial_state)
            logger.debug(f"Agent completed in {final_state.get('step_count', 0)} steps")
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            # Return error record
            return core_entities.GenerationRecord(
                task=task,
                output=None,
                error=core_entities.ModelError(
                    message=f"Agent execution failed: {e}",
                    kind="agent_error"
                ),
                metrics={"agent_steps": 0}
            )
        
        # Extract answer from final state
        answer = final_state.get("answer", "")
        plan = final_state.get("plan", "")
        step_count = final_state.get("step_count", 0)
        messages = final_state.get("messages", [])
        
        # Format the response (JSON format for consistency)
        response_text = f'{{"answer": "{answer}"}}'
        
        # Create output with full agent state in raw field
        output = core_entities.ModelOutput(
            text=response_text,
            raw=final_state,
            usage=None
        )
        
        # Create record with agent metadata
        record = core_entities.GenerationRecord(
            task=task,
            output=output,
            error=None,
            metrics={
                "agent_steps": step_count,
                "agent_plan_length": len(plan),
                "agent_message_count": len(messages)
            }
        )
        
        return record


__all__ = ["LangGraphAgentRunner"]
