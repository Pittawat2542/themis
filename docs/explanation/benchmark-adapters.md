---
title: Benchmark adapters
diataxis: explanation
audience: users running named benchmark entries
goal: Explain how benchmark entries encapsulate dataset plus adapter-specific behavior.
---

# Benchmark adapters

What it is: the mapping between a benchmark name and the dataset, variant, and adapter behavior needed to run it correctly.

When it matters: whenever a benchmark requires more than “load the dataset and score exact match,” especially for rubric QA or code-execution workflows.

What you provide: the named benchmark plus any environment prerequisites such as code-execution support or dataset extras.

What Themis provides: a catalog entry that resolves the benchmark configuration and adapter assumptions.

What to inspect when it goes wrong: verify the benchmark entry, required extras, and any execution-backend assumptions associated with that entry.
