# Themis v4

Themis v4 is a Python-first evaluation runtime for compiling experiments into immutable `RunSnapshot` artifacts, executing them through a typed generation/evaluation pipeline, and inspecting the stored results afterward.

## Install

```bash
uv pip install -e ".[dev]"
```

Optional extras:

- `uv pip install -e ".[openai]"`
- `uv pip install -e ".[vllm]"` on Linux
- `uv pip install -e ".[langgraph]"`
- `uv pip install -e ".[datasets]"`
- `uv pip install -e ".[mongodb]"`
- `uv pip install -e ".[postgres]"`

## Quick Start

```python
from themis import evaluate
from themis.core.models import Case, Dataset

result = evaluate(
    model="builtin/demo_generator",
    data=[
        Dataset(
            dataset_id="sample",
            cases=[
                Case(
                    case_id="case-1",
                    input={"question": "2+2"},
                    expected_output={"answer": "4"},
                )
            ],
        )
    ],
    metric="builtin/exact_match",
    parser="builtin/json_identity",
)

print(result.run_id, result.status.value)
```

## Documentation

- Start with [docs/index.md](/Users/pittawat/projects/themis/docs/index.md)
- API layer chooser: [docs/start-here/choose-your-api-layer.md](/Users/pittawat/projects/themis/docs/start-here/choose-your-api-layer.md)
- Tutorials: [docs/tutorials/first-evaluate.md](/Users/pittawat/projects/themis/docs/tutorials/first-evaluate.md)
- Reference: [docs/reference/python-api.md](/Users/pittawat/projects/themis/docs/reference/python-api.md)
- Concepts: [docs/explanation/run-snapshot.md](/Users/pittawat/projects/themis/docs/explanation/run-snapshot.md)

Build the docs locally with:

```bash
uv run mkdocs build --strict
```
