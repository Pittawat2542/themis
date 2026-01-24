# LangGraph Agent Example

This example (`examples/langgraph_agent`) shows how to evaluate complex agentic workflows built with [LangGraph](https://langchain-ai.github.io/langgraph/).

## Key Integration
- **State Machine**: The agent is defined as a graph of steps.
- **Themis Runner**: Wraps the LangGraph execution to allow Themis to treat it as a standard "model".

## Prerequisites
Requires `langgraph` installed:
```bash
uv pip install langgraph
```

## Running the Example

```bash
uv run python -m examples.langgraph_agent.cli run
```
