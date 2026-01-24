# Advanced Customization Example

This example (`examples/advanced`) showcases Themis's extensibility. It covers custom runners, evaluation pipelines, and metrics.

## Key Features
- **Custom Runners**: Override how tasks are executed (e.g., for multi-step agents).
- **Custom Metrics**: Implement domain-specific scoring logic.
- **Subject-Aware Evaluation**: Break down scores by dataset fields (e.g., subject, difficulty).

## Config with Custom Features
The advanced config demonstrates enabling specific custom behaviors:

```json
{
  "enable_subject_breakdown": true, // Activates custom pipeline
  "prompt_style": "cot"
}
```

## Custom Runner Implementation
See `examples/advanced/runner.py` for how to implement a runner that modifies execution flow.

## Running the Example

```bash
uv run python -m examples.advanced.cli run \
  --config-path examples/advanced/config.sample.json
```
