# Projects Example

This example (`examples/projects`) demonstrates how to organize multiple related experiments into a single Project structure. This is essential for larger research efforts.

## Key Features
- **Project Structure**: Hierarchy of `Project -> Experiment -> Run`.
- **Shared Configuration**: Define models and metrics at the project level.
- **Experiment Management**: List, run, and compare experiments within a project.

## Project Definition
Projects are defined in Python code to allow for dynamic configuration sharing:

```python
# examples/projects/project_setup.py
project = Project(
    project_id="math-benchmark-2024",
    name="Math Benchmark Study",
    ...
)
```

## CLI Usage

The CLI for this example lets you manage the project:

```bash
# List all experiments in the project
uv run python -m examples.projects.cli list-experiments

# Run a specific experiment by name
uv run python -m examples.projects.cli run --experiment math500-full
```
