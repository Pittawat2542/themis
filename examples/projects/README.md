# Organizing Projects with Multiple Experiments

This example shows you how to organize multiple related experiments into a cohesive project. This is essential for:

- Research projects with multiple hypotheses to test
- Comparing different approaches on the same benchmark
- Running different experiments with shared configurations
- Organizing a systematic evaluation campaign
- Maintaining reproducible research workflows

## What You'll Learn

1. How to create a Project to group experiments
2. How to define multiple experiments with different configurations
3. How to share common setup across experiments
4. How to run experiments selectively
5. How to organize and track results across experiments

## Project Structure Overview

A **Project** is a container for multiple **Experiments**. Each experiment can have:

- Different prompt templates
- Different models to evaluate
- Different datasets or subsets
- Different evaluation metrics
- Shared or independent configurations

## Prerequisites

```bash
uv pip install -e .
```

## Understanding the Project Model

The Themis project structure has three levels:

```
Project
├── Experiment 1 (e.g., "zero-shot-math")
│   ├── Prompt templates
│   ├── Models to test
│   ├── Sampling strategies
│   └── Datasets
├── Experiment 2 (e.g., "few-shot-math")
│   ├── Different prompts
│   ├── Same or different models
│   └── Same or different datasets
└── Experiment 3 (e.g., "chain-of-thought")
    └── ...
```

## Quick Start: Simple Project

Here's the simplest project with two experiments:

```python
from themis.project.definitions import Project, ProjectExperiment
from themis.experiment.builder import ExperimentDefinition, SamplingConfig, ModelBinding
from themis.generation import templates
from themis.core import entities as core_entities

# Define shared model
model_spec = core_entities.ModelSpec(
    identifier="fake-math-llm",
    provider="fake",
)
model_binding = ModelBinding(
    spec=model_spec,
    provider_name="fake",
    provider_options={},
)

# Experiment 1: Zero-shot prompting
zero_shot_template = templates.PromptTemplate(
    name="math-zero-shot",
    template="Solve this problem: {problem}",
)

zero_shot_definition = ExperimentDefinition(
    templates=[zero_shot_template],
    sampling_parameters=[SamplingConfig(name="greedy", temperature=0.0, max_tokens=256)],
    model_bindings=[model_binding],
    dataset_id_field="unique_id",
    reference_field="answer",
    context_builder=lambda row: {"problem": row["problem"]},
)

# Experiment 2: Few-shot prompting
few_shot_template = templates.PromptTemplate(
    name="math-few-shot",
    template="""
    Examples:
    Problem: What is 2+2? Answer: 4
    Problem: What is 5*3? Answer: 15
    
    Now solve:
    Problem: {problem}
    Answer:
    """.strip(),
)

few_shot_definition = ExperimentDefinition(
    templates=[few_shot_template],
    sampling_parameters=[SamplingConfig(name="greedy", temperature=0.0, max_tokens=256)],
    model_bindings=[model_binding],
    dataset_id_field="unique_id",
    reference_field="answer",
    context_builder=lambda row: {"problem": row["problem"]},
)

# Create the project
project = Project(
    project_id="math-prompting-study",
    name="Math Prompting Strategies",
    description="Comparing zero-shot vs few-shot prompting on math problems",
    metadata={"version": "1.0", "date": "2024-01"},
    tags=("math", "prompting", "research"),
)

# Add experiments to the project
project.add_experiment(ProjectExperiment(
    name="zero-shot",
    definition=zero_shot_definition,
    description="Baseline zero-shot prompting",
    tags=("baseline",),
))

project.add_experiment(ProjectExperiment(
    name="few-shot",
    definition=few_shot_definition,
    description="Few-shot prompting with 2 examples",
    tags=("few-shot",),
))

# List all experiments
print("Experiments in project:", project.list_experiment_names())
# Output: ('zero-shot', 'few-shot')

# Get a specific experiment
zero_shot_exp = project.get_experiment("zero-shot")
print(f"Experiment: {zero_shot_exp.name}")
print(f"Description: {zero_shot_exp.description}")
```

## Running Project Experiments

Once you've defined a project, run each experiment independently:

```python
from themis.experiment.builder import ExperimentBuilder
from themis.evaluation import extractors, metrics

# Prepare dataset
dataset = [
    {"unique_id": "1", "problem": "What is 2+2?", "answer": "4"},
    {"unique_id": "2", "problem": "What is 10-3?", "answer": "7"},
]

# Build evaluation pipeline (shared across experiments)
builder = ExperimentBuilder(
    extractor=extractors.JsonFieldExtractor(field_path="answer"),
    metrics=[metrics.ExactMatch(case_sensitive=False, strip_whitespace=True)],
)

# Run experiment 1
zero_shot_exp = project.get_experiment("zero-shot")
zero_shot_built = builder.build(
    zero_shot_exp.definition,
    storage_dir=f".cache/{project.project_id}/zero-shot",
)
zero_shot_report = zero_shot_built.orchestrator.run(
    dataset=dataset,
    run_id=f"{project.project_id}-zero-shot",
    resume=True,
)

# Run experiment 2
few_shot_exp = project.get_experiment("few-shot")
few_shot_built = builder.build(
    few_shot_exp.definition,
    storage_dir=f".cache/{project.project_id}/few-shot",
)
few_shot_report = few_shot_built.orchestrator.run(
    dataset=dataset,
    run_id=f"{project.project_id}-few-shot",
    resume=True,
)

# Compare results
print(f"Zero-shot accuracy: {zero_shot_report.evaluation_report.metrics['ExactMatch'].mean:.2%}")
print(f"Few-shot accuracy: {few_shot_report.evaluation_report.metrics['ExactMatch'].mean:.2%}")
```

## Real-World Example: Multi-Model, Multi-Dataset Project

Here's a more complex example that evaluates multiple models on multiple datasets:

```python
from themis.project.definitions import Project, ProjectExperiment
from themis.experiment.builder import ExperimentDefinition, SamplingConfig, ModelBinding
from themis.generation import templates
from themis.core import entities as core_entities

# Define models
models = [
    ModelBinding(
        spec=core_entities.ModelSpec(identifier="gpt-4o-mini", provider="openai-compatible"),
        provider_name="openai-compatible",
        provider_options={
            "base_url": "https://api.openai.com/v1",
            "api_key": "your-key",
            "model_mapping": {"gpt-4o-mini": "gpt-4o-mini"},
        },
    ),
    ModelBinding(
        spec=core_entities.ModelSpec(identifier="local-llm", provider="openai-compatible"),
        provider_name="openai-compatible",
        provider_options={
            "base_url": "http://localhost:1234/v1",
            "api_key": "not-needed",
            "model_mapping": {"local-llm": "qwen2.5-7b-instruct"},
        },
    ),
]

# Define sampling strategies
samplings = [
    SamplingConfig(name="greedy", temperature=0.0, max_tokens=512),
    SamplingConfig(name="balanced", temperature=0.7, max_tokens=512),
]

# Shared template
math_template = templates.PromptTemplate(
    name="math-solver",
    template="Solve: {problem}\nAnswer:",
)

# Create project
project = Project(
    project_id="math-benchmark-2024",
    name="Math Benchmark Study 2024",
    description="Comprehensive evaluation of math reasoning across models and datasets",
    metadata={
        "author": "Research Team",
        "institution": "University",
        "version": "2.0",
    },
    tags=("math", "benchmark", "2024"),
)

# Experiment 1: MATH-500 evaluation
project.create_experiment(
    name="math500-full",
    definition=ExperimentDefinition(
        templates=[math_template],
        sampling_parameters=samplings,
        model_bindings=models,
        dataset_id_field="unique_id",
        reference_field="answer",
        context_builder=lambda row: {"problem": row["problem"]},
    ),
    description="Full MATH-500 dataset evaluation (500 samples)",
    metadata={"dataset": "math500", "size": 500},
    tags=("math500", "full"),
)

# Experiment 2: Quick validation subset
project.create_experiment(
    name="math500-quick",
    definition=ExperimentDefinition(
        templates=[math_template],
        sampling_parameters=[samplings[0]],  # Only greedy
        model_bindings=models,
        dataset_id_field="unique_id",
        reference_field="answer",
        context_builder=lambda row: {"problem": row["problem"]},
    ),
    description="Quick validation on 50 samples",
    metadata={"dataset": "math500", "size": 50},
    tags=("math500", "quick", "validation"),
)

# Experiment 3: Subject-specific (algebra only)
project.create_experiment(
    name="math500-algebra",
    definition=ExperimentDefinition(
        templates=[math_template],
        sampling_parameters=samplings,
        model_bindings=models,
        dataset_id_field="unique_id",
        reference_field="answer",
        metadata_fields=("subject",),
        context_builder=lambda row: {"problem": row["problem"]},
    ),
    description="Algebra problems only",
    metadata={"dataset": "math500", "subject": "algebra"},
    tags=("math500", "algebra", "subject-specific"),
)

# List all experiments
for exp_name in project.list_experiment_names():
    exp = project.get_experiment(exp_name)
    print(f"- {exp_name}: {exp.description}")
```

## CLI Integration

Create a CLI that lets you run experiments by name:

```python
# experiments/04_projects/cli.py
from pathlib import Path
from typing import Annotated
from cyclopts import App, Parameter
from themis.experiment.builder import ExperimentBuilder
from themis.evaluation import extractors, metrics
from .project_setup import create_project, load_dataset

app = App(help="Run project experiments")

@app.command()
def run(
    *,
    experiment: Annotated[str, Parameter(help="Experiment name to run")],
    storage_dir: Annotated[Path, Parameter(help="Storage directory")] = Path(".cache"),
    run_id: Annotated[str | None, Parameter(help="Override run ID")] = None,
    limit: Annotated[int | None, Parameter(help="Limit dataset size")] = None,
) -> int:
    """Run a specific experiment from the project."""
    
    # Load project
    project = create_project()
    
    # Get experiment
    try:
        exp = project.get_experiment(experiment)
    except KeyError:
        print(f"Error: Experiment '{experiment}' not found")
        print(f"Available experiments: {', '.join(project.list_experiment_names())}")
        return 1
    
    # Load dataset
    dataset = load_dataset(exp.metadata.get("dataset", "demo"), limit=limit)
    
    # Build and run
    builder = ExperimentBuilder(
        extractor=extractors.JsonFieldExtractor(field_path="answer"),
        metrics=[metrics.ExactMatch(case_sensitive=False)],
    )
    
    exp_storage = storage_dir / project.project_id / exp.name
    built = builder.build(exp.definition, storage_dir=str(exp_storage))
    
    run_id = run_id or f"{project.project_id}-{exp.name}"
    report = built.orchestrator.run(dataset=dataset, run_id=run_id, resume=True)
    
    # Print results
    exact_match = report.evaluation_report.metrics.get("ExactMatch")
    print(f"Experiment: {exp.name}")
    print(f"Accuracy: {exact_match.mean:.2%}" if exact_match else "No metrics")
    
    return 0

@app.command()
def list_experiments() -> int:
    """List all experiments in the project."""
    project = create_project()
    print(f"Project: {project.name}")
    print(f"ID: {project.project_id}\n")
    print("Experiments:")
    for name in project.list_experiment_names():
        exp = project.get_experiment(name)
        tags = ", ".join(exp.tags) if exp.tags else "none"
        print(f"  - {name}: {exp.description} [tags: {tags}]")
    return 0

if __name__ == "__main__":
    app()
```

Usage:

```bash
# List all experiments
uv run python -m experiments.04_projects.cli list-experiments

# Run a specific experiment
uv run python -m experiments.04_projects.cli run --experiment math500-full

# Run with custom settings
uv run python -m experiments.04_projects.cli run \
  --experiment math500-quick \
  --storage-dir .cache/my-run \
  --run-id test-2024-01-15 \
  --limit 10
```

## Best Practices

### 1. Organize by Research Question

Group experiments that answer related questions:

```python
# Good: Clear research questions
project = Project(
    project_id="prompting-strategies-2024",
    name="Impact of Prompting Strategies on Math Reasoning",
)
project.create_experiment(name="baseline-zero-shot", ...)
project.create_experiment(name="few-shot-2-examples", ...)
project.create_experiment(name="few-shot-5-examples", ...)
project.create_experiment(name="chain-of-thought", ...)
```

### 2. Use Consistent Naming

Establish a naming convention:

```python
# Pattern: {dataset}-{method}-{variant}
project.create_experiment(name="math500-zero-shot", ...)
project.create_experiment(name="math500-few-shot", ...)
project.create_experiment(name="gsm8k-zero-shot", ...)
project.create_experiment(name="gsm8k-few-shot", ...)
```

### 3. Tag Experiments

Use tags for filtering and organization:

```python
project.create_experiment(
    name="math500-baseline",
    definition=...,
    tags=("math500", "baseline", "published"),
)

project.create_experiment(
    name="math500-experimental",
    definition=...,
    tags=("math500", "experimental", "in-progress"),
)
```

### 4. Share Common Configuration

Define shared components once:

```python
# Shared sampling strategies
SHARED_SAMPLINGS = [
    SamplingConfig(name="greedy", temperature=0.0, max_tokens=512),
    SamplingConfig(name="creative", temperature=0.8, max_tokens=512),
]

# Shared models
SHARED_MODELS = [...]

# Use in all experiments
for exp_name in ["exp1", "exp2", "exp3"]:
    project.create_experiment(
        name=exp_name,
        definition=ExperimentDefinition(
            sampling_parameters=SHARED_SAMPLINGS,
            model_bindings=SHARED_MODELS,
            ...
        ),
    )
```

### 5. Use Project Metadata

Store project-level information:

```python
project = Project(
    project_id="math-benchmark-2024",
    name="Math Benchmark Study",
    metadata={
        "version": "1.0.0",
        "date_started": "2024-01-15",
        "author": "Research Team",
        "paper": "https://arxiv.org/...",
        "repository": "https://github.com/...",
    },
)

# Access later
print(project.metadata["version"])
```

### 6. Organize Storage

Use consistent directory structure:

```
.cache/
└── {project_id}/
    ├── {experiment_1}/
    │   └── generations/
    ├── {experiment_2}/
    │   └── generations/
    └── results/
        ├── experiment_1_results.csv
        └── experiment_2_results.csv
```

## Advanced: Running All Experiments

Create a script to run all experiments systematically:

```python
from pathlib import Path
from themis.experiment.builder import ExperimentBuilder
from themis.evaluation import extractors, metrics
from .project_setup import create_project, load_dataset

def run_all_experiments(storage_root: Path, limit: int | None = None):
    """Run all experiments in the project."""
    
    project = create_project()
    builder = ExperimentBuilder(
        extractor=extractors.JsonFieldExtractor(field_path="answer"),
        metrics=[metrics.ExactMatch(case_sensitive=False)],
    )
    
    results = {}
    
    for exp_name in project.list_experiment_names():
        print(f"\n{'='*60}")
        print(f"Running experiment: {exp_name}")
        print(f"{'='*60}\n")
        
        exp = project.get_experiment(exp_name)
        dataset = load_dataset(exp.metadata.get("dataset", "demo"), limit=limit)
        
        storage_dir = storage_root / project.project_id / exp_name
        built = builder.build(exp.definition, storage_dir=str(storage_dir))
        
        report = built.orchestrator.run(
            dataset=dataset,
            run_id=f"{project.project_id}-{exp_name}",
            resume=True,
        )
        
        results[exp_name] = report.evaluation_report.metrics.get("ExactMatch").mean
        print(f"✓ {exp_name}: {results[exp_name]:.2%}")
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for exp_name, accuracy in results.items():
        print(f"{exp_name:30s} {accuracy:.2%}")
    
    return results

if __name__ == "__main__":
    run_all_experiments(Path(".cache"), limit=10)
```

## Troubleshooting

**Q: Can experiments share storage directories?**
A: No, use separate directories per experiment to avoid cache collisions.

**Q: How do I run experiments in parallel?**
A: Use Python's multiprocessing or run multiple CLI processes with different experiment names.

**Q: Can I add experiments after creating a project?**
A: Yes, use `project.add_experiment()` or `project.create_experiment()` at any time.

**Q: How do I version my project?**
A: Store version info in `project.metadata` and use different `project_id` values for major changes.

## Next Steps

- **05_advanced**: Learn how to customize generation loops and evaluation
- Review `themis.project.definitions` for the full API
- Check out real research projects using this structure

## File Structure

```
04_projects/
├── README.md           # This file
├── cli.py              # CLI for running experiments
├── project_setup.py    # Project definition
├── datasets.py         # Dataset loaders
└── configs/
    ├── project.json    # Project metadata
    └── experiments/
        ├── exp1.json
        └── exp2.json
```
