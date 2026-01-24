# Examples Overview

Themis comes with a rich set of examples demonstrating how to use the framework for various use cases, from simple evaluations to complex agentic workflows.

Each example is a self-contained directory in the `examples/` folder of the repository, including a runnable Python module and configuration files.

## List of Examples

### Basics
- **[01_getting_started](getting_started.md)**: Your first Themis experiment. Learn the CLI and basic configuration.
- **[02_config_file](config_file.md)**: How to use JSON configuration files for reproducible experiments.

### Intermediate
- **[03_prompt_engineering](prompt_engineering.md)**: Systematically test and compare different prompt templates.
- **[04_projects](projects.md)**: Organize multiple related experiments (e.g., zero-shot vs few-shot) into a Project.
- **[litellm_example](litellm_example.md)**: Using 100+ LLM providers (OpenAI, Anthropic, Local models via vLLM/Ollama).

### Advanced
- **[05_advanced](advanced.md)**: Customization deep dive: custom runners, metrics, and pipelines.
- **[judge_evaluation](judge_evaluation.md)**: LLM-as-a-Judge evaluation patterns.
- **[finetuning_data](finetuning_data.md)**: Generating and filtering synthetic training data.
- **[rag_pipeline](rag_pipeline.md)**: Building and evaluating RAG systems with Themis.
- **[langgraph_agent](langgraph_agent.md)**: Evaluating complex agents built with LangGraph.

## Running Examples

All examples can be run using `uv run` or standard `python` commands. We recommend `uv` for easy dependency management.

**Standard Pattern:**
```bash
# General CLI runner for examples
uv run python -m examples.<example_name>.cli run
```

**With Configuration:**
```bash
uv run python -m examples.<example_name>.cli run --config-path examples/<example_name>/config.sample.json
```
