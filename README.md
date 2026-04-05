# Themis

Themis is a Python package for running reproducible LLM evaluations. It gives you a typed scaffold for defining datasets, generators, parsers, metrics, judge workflows, and persistent run artifacts without forcing you into one provider or benchmark.

The published package name is `themis-eval`. The Python import namespace and CLI command are both `themis`.

## Install

```bash
pip install themis-eval
```

Optional extras:

- `pip install "themis-eval[openai]"`
- `pip install "themis-eval[vllm]"` on Linux
- `pip install "themis-eval[langgraph]"`
- `pip install "themis-eval[datasets]"`
- `pip install "themis-eval[mongodb]"`
- `pip install "themis-eval[postgres]"`
- `pip install "themis-eval[docs]"` for local documentation builds

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

## Custom Extensions

Themis is designed to be extended. You can plug in custom generators, parsers, reducers, metrics, judge models, and store backends through the Python API or config-driven workflows.

- Start with [`Experiment(...)`](docs/tutorials/first-experiment.md) when you want a reusable compiled evaluation definition.
- Start with [`evaluate(...)`](docs/tutorials/first-evaluate.md) when you want the shortest path from inline data to a completed run.
- Use [`docs/how-to/author-custom-components.md`](docs/how-to/author-custom-components.md) for custom component authoring.

## CLI

After installation, the package exposes the `themis` CLI:

```bash
themis quick-eval inline \
  --model builtin/demo_generator \
  --metric builtin/exact_match \
  --parser builtin/json_identity \
  --input '{"question":"2+2"}' \
  --expected-output '{"answer":"4"}'
```

## Documentation

- Start here: [`docs/index.md`](docs/index.md)
- Installation guide: [`docs/start-here/installation.md`](docs/start-here/installation.md)
- API layer chooser: [`docs/start-here/choose-your-api-layer.md`](docs/start-here/choose-your-api-layer.md)
- Python API reference: [`docs/reference/python-api.md`](docs/reference/python-api.md)
- Extension boundaries: [`docs/explanation/extension-boundaries.md`](docs/explanation/extension-boundaries.md)

Build the docs locally with:

```bash
pip install "themis-eval[docs]"
mkdocs build --strict
```

## Contributing

Contributor setup and release guidance live in [`CONTRIBUTING.md`](CONTRIBUTING.md).
