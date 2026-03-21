# Author Project Files

Project files are for shared runtime policy, not for benchmark semantics.

## Keep in `project.toml`

- storage root
- storage backend
- execution policy
- project seed used as the default deterministic seed namespace for planning and execution

## Keep in Python

- benchmark slices
- dataset queries
- prompt variants
- parse pipelines
- score overlays

Load a project file with:

```python
orchestrator = Orchestrator.from_project_file(
    "project.toml",
    registry=registry,
    dataset_provider=dataset_provider,
)
```

Worked example: `examples/02_project_file.py`
