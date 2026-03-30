---
title: Adding new docs coverage
diataxis: project
audience: contributors adding public APIs, commands, builtins, or benchmarks
goal: Explain how to keep docs coverage aligned with the evolving public surface.
---

# Adding new docs coverage

Whenever you add a new public surface:

- root export: update the relevant Python API/reference page
- CLI command or subcommand: update CLI reference and inventory coverage
- builtin component or benchmark entry: update the relevant catalog docs and keep the inventory script truthful
- user-facing workflow: add or extend a runnable example and point a tutorial or guide at it when appropriate

Required local checks:

```bash
uv run pytest tests/test_docs_site.py tests/test_docs_examples.py tests/test_docs_inventory.py -q
uv run mkdocs build --strict
```
