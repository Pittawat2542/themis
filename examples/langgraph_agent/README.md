# LangGraph Agent Example

This example demonstrates how to integrate a LangGraph agent with Themis for systematic evaluation of agentic workflows.

## Prerequisites

- Python 3.9+
- Themis installed
- **LangGraph**: `pip install langgraph`

## Features Demonstrated

- **LangGraph Integration**: Wrapping a LangGraph state machine in a Themis runner
- **Agentic Workflows**: Multi-step reasoning with planning and execution phases
- **Custom Runner**: Implementing `_execute_task` to run the agent
- **Metadata Tracking**: Capturing agent steps, plans, and intermediate states
- **Error Handling**: Graceful handling of agent failures

## Installation

Install the required dependency:

```bash
pip install langgraph
```

Or add to your `requirements.txt`:
```
langgraph>=0.1.0
```

## How it Works

### Architecture

```
User Problem
    ↓
LangGraphAgentRunner
    ↓
Create Initial State
    ↓
LangGraph Agent
  ├─> Plan Step (generate plan)
  └─> Solve Step (execute plan)
    ↓
Extract Answer
    ↓
Return Response
```

### Components

1. **Agent** (`agent.py`): LangGraph state machine
   - **Plan node**: Analyzes problem and creates a plan
   - **Solve node**: Executes the plan and produces an answer
   - **State**: Tracks messages, plan, answer, and step count

2. **Runner** (`runner.py`): Themis adapter
   - Converts Themis tasks to agent state
   - Invokes the LangGraph agent
   - Extracts results back to Themis format
   - Handles errors gracefully

3. **Experiment** (`experiment.py`): Wires everything together

## Running the Example

```bash
# Run with default settings
uv run python -m examples.langgraph_agent.cli run

# Note: Requires langgraph to be installed
pip install langgraph
uv run python -m examples.langgraph_agent.cli run
```

## Expected Output

```
Running LangGraph agent experiment with 2 samples...

Experiment completed!

--- Example Agent Execution ---
Problem: What is 5 + 3?
Answer: {"answer": "42"}
Agent Plan: To solve 'What is 5 + 3?', I will:
1. Analyze the problem
2. Apply relevant mathematical concepts
3. Calculate the answer
Agent Steps: 2
```

## Key Code Examples

### Defining the Agent State

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    problem: str
    plan: str
    answer: str
    step_count: int
```

### Creating the Agent

```python
from langgraph.graph import StateGraph, END

def create_agent():
    workflow = StateGraph(AgentState)
    
    # Add nodes (agent steps)
    workflow.add_node("plan", plan_step)
    workflow.add_node("solve", solve_step)
    
    # Define flow
    workflow.set_entry_point("plan")
    workflow.add_edge("plan", "solve")
    workflow.add_edge("solve", END)
    
    return workflow.compile()
```

### Wrapping in Themis Runner

```python
class LangGraphAgentRunner(GenerationRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent = create_agent()
    
    def _execute_task(self, task):
        # Initialize state
        initial_state = {
            "problem": task.prompt.text,
            "messages": [],
            "plan": "",
            "answer": "",
            "step_count": 0
        }
        
        # Run agent
        final_state = self.agent.invoke(initial_state)
        
        # Convert to Themis format
        return GenerationRecord(...)
```

## Extending This Example

### Add Real LLM Calls

Replace placeholder logic with actual LLM calls:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")

def plan_step(state: AgentState) -> AgentState:
    problem = state["problem"]
    
    # Call LLM to generate plan
    plan_prompt = f"Create a step-by-step plan to solve: {problem}"
    plan = llm.invoke(plan_prompt).content
    
    return {
        **state,
        "plan": plan,
        "step_count": state["step_count"] + 1,
        "messages": [{"role": "assistant", "content": plan}]
    }
```

### Implement Different Agentic Patterns

**ReAct Pattern:**
```python
workflow.add_node("think", think_step)
workflow.add_node("act", act_step)
workflow.add_node("observe", observe_step)

workflow.set_entry_point("think")
workflow.add_conditional_edges("think", should_act_or_finish)
workflow.add_edge("act", "observe")
workflow.add_edge("observe", "think")
```

**Plan-Execute Pattern:**
```python
workflow.add_node("plan", plan_step)
workflow.add_node("execute", execute_step)
workflow.add_node("replan", replan_step)

workflow.add_conditional_edges("execute", check_if_done)
```

### Add Tool Use

```python
from langchain.tools import Tool

tools = [
    Tool(name="calculator", func=lambda x: eval(x)),
    Tool(name="search", func=search_web)
]

def tool_step(state):
    # Agent decides which tool to use
    # Execute tool
    # Update state with results
    pass
```

## Troubleshooting

**Issue**: `ImportError: No module named 'langgraph'`
- Solution: Install langgraph: `pip install langgraph`

**Issue**: Agent gets stuck in a loop
- Solution: Add a `max_steps` limit and conditional exit logic

**Issue**: Want to pass LLM provider to agent
- Solution: Modify agent creation to accept provider:
  ```python
  def create_agent(provider):
      def plan_step(state):
          # Use provider here
          pass
      ...
  ```

## Performance Considerations

- **Agent Steps**: More steps = more LLM calls = higher cost
- **State Size**: Keep state minimal, only essential information
- **Caching**: LangGraph supports checkpointing for long-running agents

## Related Examples

- **[advanced](../advanced/)**: More custom runner patterns including `AgenticRunner`
- **[rag_pipeline](../rag_pipeline/)**: Another custom runner example
- **[projects](../projects/)**: Organizing multiple agent experiments

## Further Reading

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Agentic Patterns](https://www.anthropic.com/research/building-effective-agents)
- Themis [README](../../README.md) for core concepts
