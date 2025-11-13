# Judge Evaluation Example

This example demonstrates how to use Themis's judge-based evaluation metrics to evaluate model outputs using LLM-as-a-judge patterns.

## Features Demonstrated

- **RubricJudgeMetric**: Score outputs against specific rubric criteria with detailed scoring per criterion
- **ConsistencyMetric**: Measure agreement across multiple judge evaluations
- **JudgeEvaluationStrategy**: Aggregate scores from multiple judges and calculate inter-judge agreement

## Running the Example

### Using the CLI

```bash
uv run python -m examples.judge_evaluation.cli
```

### Programmatically

```python
from examples.judge_evaluation.experiment import run_experiment, summarize_report

# Run with default config
report = run_experiment()

# Display summary
print(summarize_report(report))
```

## Project Structure

```
judge_evaluation/
├── __init__.py          # Package metadata
├── cli.py               # CLI entry point using cyclopts
├── config.py            # Experiment configuration (judge models, rubrics)
├── datasets.py          # Demo dataset with candidate solutions
├── experiment.py        # Main experiment logic
└── README.md            # This file
```

## Key Components

### Rubric Configuration

The rubric defines criteria for evaluating solutions:

```python
from themis.evaluation.metrics import RubricJudgeMetric

rubric = {
    "correctness": "Answer matches ground truth exactly",
    "reasoning": "Provides clear step-by-step reasoning",
    "clarity": "Solution is well-explained and easy to follow"
}

judge_metric = RubricJudgeMetric(
    judge_model=judge_model_spec,
    judge_provider=provider,
    rubric=rubric,
)
```

### Judge Evaluation Strategy

Aggregates scores from multiple judges and calculates agreement:

```python
from themis.evaluation.strategies import JudgeEvaluationStrategy

builder = ExperimentBuilder(
    extractor=IdentityExtractor(),
    metrics=[judge_metric],
    evaluation_strategy_resolver=lambda record: JudgeEvaluationStrategy(),
)
```

### Dataset

The demo dataset contains math problems with candidate solutions:

```python
{
    "unique_id": "sample_001",
    "question": "What is 2+2?",
    "answer": "4",
    "candidate_solution": "The answer is 4.",
    "subject": "arithmetic"
}
```

## Output

The experiment produces:

- **Metric scores**: RubricJudge scores aggregated across all samples
- **Judge agreement**: Inter-judge agreement when using multiple judges
- **Per-criterion scores**: Detailed breakdown by rubric criteria
- **Evaluation report**: Complete report with all metrics and metadata

Example output:
```
Running judge evaluation experiment...
Evaluated 3 samples | ExactMatch: 0.000 | RubricJudge: 0.750 | No failures
```

## Extending the Example

### Add Custom Rubric Criteria

Modify `config.py` to add your own rubric:

```python
rubric = {
    "accuracy": "Mathematical correctness",
    "efficiency": "Solution uses optimal approach",
    "explanation": "Clear explanation of methodology"
}
```

### Use Real Judge Models

Replace `FakeMathModelClient` with actual LLM providers:

```python
from themis.generation import clients

judge_provider = clients.OpenAIModelClient(api_key="...")
judge_model = ModelSpec(identifier="gpt-4", provider="openai")
```

### Add More Metrics

Include additional metrics alongside judge evaluation:

```python
from themis.evaluation import metrics

builder = ExperimentBuilder(
    extractor=extractor,
    metrics=[
        metrics.ExactMatch(),
        metrics.RubricJudgeMetric(...),
        metrics.ConsistencyMetric(...),
        metrics.LengthDifferenceTolerance(max_diff=10),
    ],
    ...
)
```

## Related Documentation

- [RubricJudgeMetric](../../themis/evaluation/metrics/rubric_judge_metric.py)
- [PairwiseJudgeMetric](../../themis/evaluation/metrics/pairwise_judge_metric.py)
- [ConsistencyMetric](../../themis/evaluation/metrics/consistency_metric.py)
- [JudgeEvaluationStrategy](../../themis/evaluation/strategies/judge_evaluation_strategy.py)
