# Hello World Walkthrough

This tutorial mirrors `examples/01_hello_world.py`.

## Run It

```bash
uv run python examples/01_hello_world.py
```

Output:

```text
{'model_id': 'demo-model', 'slice_id': 'arithmetic', 'metric_id': 'exact_match', 'source': 'synthetic', 'prompt_variant_id': 'qa-default', 'mean': 1.0, 'count': 1}
```

## What To Notice

- `ArithmeticDatasetProvider.scan(...)` receives both `slice_spec` and `query`
- the benchmark declares semantic dimensions on the slice
- prompt selection is explicit through `prompt_variant_ids=["qa-default"]`
- the read side is aggregation-first through `result.aggregate(...)`

## Full Script

--8<-- "examples/01_hello_world.py"
