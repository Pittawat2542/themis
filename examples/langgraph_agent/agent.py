"""Simple LangGraph agent for demonstrating integration with Themis.

This module demonstrates how to create a simple agentic workflow using LangGraph
that can be integrated with Themis for systematic evaluation.
"""
from __future__ import annotations

import logging
from typing import TypedDict, Annotated

try:
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
   
    # Define minimal stubs for type checking when langgraph not installed
    class StateGraph:  # type: ignore
        pass
    
    END = "END"
    
    def add_messages(x, y):  # type: ignore
        return x + y

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State for the agent.
    
    Attributes:
        messages: List of messages in the conversation.
        problem: The problem to solve.
        plan: The plan for solving the problem.
        answer: The final answer.
        step_count: Number of steps taken by the agent.
    """
    messages: Annotated[list, add_messages]
    problem: str
    plan: str
    answer: str
    step_count: int


def plan_step(state: AgentState) -> AgentState:
    """Create a plan for solving the problem.
    
    Args:
        state: Current agent state.
    
    Returns:
        Updated state with the plan.
    
    Note:
        In a real agent, this would call an LLM to generate a plan.
        This is a simplified version for demonstration.
    """
    problem = state["problem"]
    
    # Simple planning logic (in production, call LLM here)
    plan = (
        f"To solve '{problem}', I will:\n"
        f"1. Analyze the problem\n"
        f"2. Apply relevant mathematical concepts\n"
        f"3. Calculate the answer"
    )
    
    logger.debug(f"Generated plan: {plan}")
    
    return {
        **state,
        "plan": plan,
        "step_count": state.get("step_count", 0) + 1,
        "messages": [{"role": "assistant", "content": f"Plan: {plan}"}]
    }


def solve_step(state: AgentState) -> AgentState:
    """Execute the plan and solve the problem.
    
    Args:
        state: Current agent state with plan.
    
    Returns:
        Updated state with the answer.
    
    Note:
        In a real agent, this would call an LLM to execute the plan.
    """
    problem = state["problem"]
    plan = state.get("plan", "")
    
    # Simple heuristic solving (in production, call LLM here)
    answer = "42"  # Placeholder answer
    reasoning = (
        f"Following the plan:\n{plan}\n\n"
        f"I determined the answer is {answer}"
    )
    
    logger.debug(f"Generated answer: {answer}")
    
    return {
        **state,
        "answer": answer,
        "step_count": state.get("step_count", 0) + 1,
        "messages": [{"role": "assistant", "content": reasoning}]
    }


def create_agent() -> StateGraph:
    """Create a simple LangGraph agent for math problem solving.
    
    Returns:
        Compiled LangGraph workflow.
    
    Raises:
        ImportError: If langgraph is not installed.
    
    Example:
        >>> agent = create_agent()
        >>> result = agent.invoke({
        ...     "problem": "What is 2+2?",
        ...     "messages": [],
        ...     "plan": "",
        ...     "answer": "",
        ...     "step_count": 0
        ... })
        >>> print(result["answer"])
        42
    """
    if not LANGGRAPH_AVAILABLE:
        raise ImportError(
            "langgraph is required for this example. "
            "Install it with: pip install langgraph"
        )
    
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("plan", plan_step)
    workflow.add_node("solve", solve_step)
    
    # Add edges
    workflow.set_entry_point("plan")
    workflow.add_edge("plan", "solve")
    workflow.add_edge("solve", END)
    
    logger.info("Created LangGraph agent with plan -> solve workflow")
    
    return workflow.compile()


__all__ = ["create_agent", "AgentState", "LANGGRAPH_AVAILABLE"]
