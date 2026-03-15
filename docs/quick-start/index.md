# Quick Start

Run this when you want the smallest useful Themis workflow on screen
immediately: one dataset loader, one inference engine, one metric, and one
`Orchestrator`.

The source of truth for this workflow is the runnable example
`examples/01_hello_world.py`:

```python
--8<-- "examples/01_hello_world.py"
```

Expected output:

```text
item-1 1.0
item-2 1.0
```

After this run you should have a SQLite database under
`.cache/themis-examples/01-hello-world` and two successful trial projections.

## Next Steps

- Read the [Hello World walkthrough](../tutorials/hello-world.md) if you want to
  build the same script step by step.
- Use the [dataset loader guide](../guides/dataset-loaders.md) to plug in real data.
- Add the `stats` extra and continue with [Analyze Results](../tutorials/analyze-results.md).
