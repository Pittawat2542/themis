# Getting Started with Themis

This is the simplest introduction to Themis. You'll learn how to:

1. Define a prompt template
2. Configure models and sampling parameters
3. Load or create a dataset
4. Run an experiment
5. View results

## What This Example Does

This example evaluates a math-solving LLM using a simple prompt template. The model receives a math problem and must respond with a JSON object containing an `answer` field. We extract the answer and compare it against the reference using exact match.

## Prerequisites

```bash
# Install Themis
uv pip install -e .

# Or if you're developing
uv pip install -e '.[dev]'
```

## Quick Start (Programmatic)

The simplest way to use Themis is programmatically:

```python
from themis.generation import templates
from themis.experiment import builder as experiment_builder
from themis.evaluation import extractors, metrics
from themis.core import entities as core_entities

# 1. Define your prompt template
template = templates.PromptTemplate(
    name="math-zero-shot",
    template="""
    You are an expert mathematician. Solve the problem below and respond with a JSON object
    containing `answer` and `reasoning` keys only.

    Problem:
    {problem}
    """.strip(),
)

# 2. Configure sampling parameters
sampling = experiment_builder.SamplingConfig(
    name="zero-shot",
    temperature=0.0,
    top_p=0.95,
    max_tokens=512,
)

# 3. Configure your model
model_spec = core_entities.ModelSpec(
    identifier="fake-math-llm",
    provider="fake",
)
model_binding = experiment_builder.ModelBinding(
    spec=model_spec,
    provider_name="fake",
    provider_options={},
)

# 4. Create experiment definition
definition = experiment_builder.ExperimentDefinition(
    templates=[template],
    sampling_parameters=[sampling],
    model_bindings=[model_binding],
    dataset_id_field="unique_id",
    reference_field="answer",
    metadata_fields=("subject", "level"),
    context_builder=lambda row: {"problem": row["problem"]},
)

# 5. Build the experiment with evaluation
builder = experiment_builder.ExperimentBuilder(
    extractor=extractors.JsonFieldExtractor(field_path="answer"),
    metrics=[metrics.ExactMatch(case_sensitive=False, strip_whitespace=True)],
)

built = builder.build(definition, storage_dir=".cache/getting-started")

# 6. Create a simple dataset
dataset = [
    {
        "unique_id": "demo-1",
        "problem": "What is 2 + 2?",
        "answer": "4",
        "subject": "arithmetic",
        "level": "basic",
    },
    {
        "unique_id": "demo-2",
        "problem": "What is 10 - 3?",
        "answer": "7",
        "subject": "arithmetic",
        "level": "basic",
    },
]

# 7. Run the experiment
report = built.orchestrator.run(
    dataset=dataset,
    run_id="demo-run",
    resume=True,
)

# 8. View results
print(f"Total samples: {report.metadata['total_samples']}")
print(f"Successful: {report.metadata['successful_generations']}")
exact_match = report.evaluation_report.metrics.get("ExactMatch")
if exact_match:
    print(f"Exact Match: {exact_match.mean:.2%}")
```

## Quick Start (CLI)

For a more convenient workflow, use the CLI:

```bash
# Run with default configuration (uses fake model)
uv run python -m examples.getting_started.cli run

# Dry run to preview the configuration
uv run python -m examples.getting_started.cli run --dry-run

# Run with custom storage and run ID
uv run python -m examples.getting_started.cli run \
  --storage-dir .cache/my-first-run \
  --run-id my-first-run

# Export results to various formats
uv run python -m experiments.01_getting_started.cli run \
  --csv-output results.csv \
  --html-output results.html \
  --json-output results.json
```

## Using a Configuration File

Instead of command-line arguments, you can use a JSON configuration file:

```bash
# Use the provided sample config
uv run python -m examples.getting_started.cli run \
  --config-path examples/getting_started/config.sample.json

# Create your own config by copying the sample
cp examples/getting_started/config.sample.json my_config.json
# Edit my_config.json with your settings
uv run python -m examples.getting_started.cli run --config-path my_config.json
```

## Configuration File Structure

The `config.sample.json` file shows the basic structure:

```json
{
  "run_id": "math500-demo",
  "storage_dir": ".cache/math500-demo",
  "resume": true,
  "models": [
    {
      "name": "fake-math-llm",
      "provider": "fake",
      "description": "Heuristic math model"
    }
  ],
  "samplings": [
    {
      "name": "zero-shot",
      "temperature": 0.0,
      "top_p": 0.95,
      "max_tokens": 512
    }
  ],
  "datasets": [
    {
      "name": "demo",
      "kind": "demo",
      "limit": 2
    }
  ]
}
```

**Key fields:**
- `run_id`: Unique identifier for this experiment run
- `storage_dir`: Where to cache results (enables resumability)
- `resume`: If true, skips already-completed samples
- `models`: List of models to evaluate
- `samplings`: Different sampling configurations to try
- `datasets`: Which datasets to use

## Understanding the Output

When you run an experiment, you'll see:

1. **Progress bar**: Shows generation progress
2. **Summary line**: Key metrics like exact match accuracy
3. **Exported files**: CSV, HTML, or JSON reports if requested

Example output:
```
Generating: 100%|██████████| 4/4 [00:01<00:00,  2.50it/s]
Evaluated 4 samples | Successful generations: 4/4 | Exact match: 0.750 (4 evaluated) | No failures
```

## Next Steps

Now that you understand the basics:

- **02_config_file**: Learn advanced configuration techniques
- **03_prompt_engineering**: Master systematic prompt comparison
- **04_projects**: Organize multiple experiments in a project
- **05_advanced**: Customize generation loops and evaluation pipelines

## File Structure

```
01_getting_started/
├── README.md          # This file
├── cli.py             # CLI entry point
├── config.py          # Configuration models
├── config.sample.json # Sample configuration
├── datasets.py        # Dataset loaders
└── experiment.py      # Experiment implementation
```

## Common Issues

**Q: The fake model always returns wrong answers!**
A: The fake model is just for demonstration. See `03_prompt_engineering` for using real models or configuring your own endpoints.

**Q: Where are my results stored?**
A: Check the `storage_dir` from your config (default: `.cache/math500-demo`). Results are cached there for resumability.

**Q: How do I use my own dataset?**
A: See `datasets.py` for examples. You can load from JSON, CSV, or define inline in code.

**Q: Can I evaluate multiple models at once?**
A: Yes! Add more entries to the `models` array in your config file.