---
title: Docs architecture
diataxis: project
audience: contributors and maintainers
goal: Document the structure and source-of-truth rules of the Themis docs system.
---

# Docs architecture

Themis docs are split into:

- public Diátaxis docs under `docs/`
- runnable example sources under `examples/docs/`
- tests that enforce docs inventory and example coverage
- generated API reference from docstrings
- the docs inventory script at `scripts/docs/build_inventory.py`

Source-of-truth rules:

- manifests and exported Python surface define reference coverage
- examples are source-of-truth for code snippets
- project/process docs live under `docs/project/` and `CONTRIBUTING.md`
