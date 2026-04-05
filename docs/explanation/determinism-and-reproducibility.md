---
title: Determinism and reproducibility
diataxis: explanation
audience: users designing stable examples and repeatable evaluation workflows
goal: Explain how seeds, stored artifacts, and provider behavior interact in practice.
---

# Determinism and reproducibility

What it is: the difference between deterministic local examples and reproducible real-world workflows.

When it matters: whenever you expect the same `run_id`, the same outputs, or the same downstream scores across executions.

What you provide: identity-bearing settings such as seeds, stable component fingerprints, and persisted artifacts when rerunning evaluation.

What Themis provides: snapshot identity, artifact persistence, and rejudge flows that avoid regenerating candidates.

What to inspect when it goes wrong: verify seeds, component identity, optional provider nondeterminism, and whether the workflow is rerunning generation or only reusing stored artifacts.
