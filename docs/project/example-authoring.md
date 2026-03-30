---
title: Example authoring
diataxis: project
audience: contributors adding or changing runnable docs examples
goal: Define how docs examples should be structured and validated.
---

# Example authoring

Rules for `examples/docs/*.py`:

- expose `run_example()`
- keep the example focused on one primary decision
- prefer deterministic local behavior
- document prerequisites and artifacts in the docs page that embeds the example
- avoid undocumented internal IR helpers
- use tiny fixture data
- make provider-backed examples runnable with fake or injected clients when possible
