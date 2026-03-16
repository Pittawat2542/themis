# Quick Start

Run this when you want the smallest useful Themis workflow on screen
immediately: one dataset loader, one inference engine, one metric, and one
`Orchestrator`.

This is a run-the-example page, not a line-by-line lesson. If you want the
stepwise build, continue to the tutorial after you run this file once.

Run it from the repository root:

```bash
uv run python examples/01_hello_world.py
```

The source of truth for this workflow is the runnable example
`examples/01_hello_world.py`:

```python
--8<-- "examples/01_hello_world.py"
```

Expected output:

```text
Stored SQLite database: .cache/themis-examples/01-hello-world/themis.sqlite3
item-1: exact_match=1.0
item-2: exact_match=1.0
```

This example always writes to:

```text
.cache/themis-examples/01-hello-world/themis.sqlite3
```

Other runs can produce different hashes, timestamps, and overlay IDs. This
output is stable because the example uses only local demo components.

If you rerun the example against the same `.cache` directory, Themis reuses the
same storage root and overwrites the same example-sized SQLite store rather than
creating a new random path.

## Next Steps

- Read the [Hello World walkthrough](../tutorials/hello-world.md) if you want to
  build the same script step by step.
- Use the [dataset loader guide](../guides/dataset-loaders.md) to plug in real data.
- Add the `stats` extra and continue with [Analyze Results](../guides/analyze-results.md).
