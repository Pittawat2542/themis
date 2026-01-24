# Judge Evaluation Example

This example (`examples/judge_evaluation`) demonstrates the "LLM-as-a-Judge" pattern, where a strong model evaluates the outputs of another model.

## Key Features
- **Rubric-based Evaluation**: define specific criteria (reasoning, tone, correctness).
- **Reference-less Evaluation**: Evaluate quality without a ground truth answer.

## Configuration
The judge criteria are defined in code or config:

```python
rubric = {
    "reasoning": "Does the response provide a logical step-by-step argument?",
    "clarity": "Is the response easy to understand?"
}
```

## Running the Example

```bash
uv run python -m examples.judge_evaluation.cli
```
