# Prompt Engineering Example

This example (`examples/prompt_engineering`) focuses on comparing different prompting strategies (Zero-shot, Few-shot, Chain-of-Thought) to optimize model performance.

## Key Features
- **Prompt Templates**: Define reusable templates with variables.
- **Systematic Comparison**: Compare multiple variants in a single run.
- **Metadata Tagging**: Track which strategy generated which result.

## Configuration

Define your prompt variants in the config file:

```json
"prompt_variants": [
  {
    "name": "zero-shot",
    "template": "Solve: {problem}",
    "description": "Direct asking"
  },
  {
    "name": "chain-of-thought",
    "template": "Solve: {problem}\nThink step-by-step:",
    "description": "Reasoning based"
  }
]
```

## Running the Example

```bash
uv run python -m examples.prompt_engineering.cli run \
  --config-path examples/prompt_engineering/config.sample.json
```

## Analysis
This example includes a script to generate a comparison report of how each prompt strategy performed.

```bash
uv run python -m examples.prompt_engineering.cli analyze \
  --run-id <your-run-id>
```
