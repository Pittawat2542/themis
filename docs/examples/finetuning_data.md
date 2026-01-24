# Finetuning Data Generation

This example (`examples/finetuning_data`) shows how to use Themis to generate, filter, and export synthetic data for fine-tuning.

## Workflow
1. **Generate**: Run a teacher model on a set of prompts.
2. **Filter**: Keep only high-quality responses (e.g., those that pass specific checks).
3. **Export**: Save to JSONL format compatible with training frameworks.

## Running the Example

```bash
# Generate data
uv run python -m examples.finetuning_data.cli run

# Check output
head -n 5 finetuning_data.jsonl
```
