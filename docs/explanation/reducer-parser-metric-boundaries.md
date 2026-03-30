---
title: Reducer vs parser vs metric
diataxis: explanation
audience: users authoring custom components
goal: Clarify responsibility boundaries between reduction, parsing, and scoring.
---

# Reducer vs parser vs metric boundaries

What it is: the responsibility split between choosing a candidate, normalizing it, and assigning a score.

When it matters: whenever a custom component feels overloaded or two components seem to be solving the same problem.

What you provide:

- reducers that choose or synthesize a reduced candidate
- parsers that normalize reduced output
- metrics that score parsed subjects

What Themis provides: stage ordering and typed inputs to each stage.

What to inspect when it goes wrong: check whether the component is changing data in the wrong stage or relying on information it should not own.
