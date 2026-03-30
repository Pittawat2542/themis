---
title: Generation vs evaluation
diataxis: explanation
audience: users designing or debugging component boundaries
goal: Explain the ownership boundary between candidate generation and Themis-owned evaluation.
---

# Generation vs evaluation

What it is: the split between candidate production and the runtime-owned scoring/evaluation pipeline.

When it matters: whenever you are deciding whether logic belongs in a generator, parser, reducer, or metric.

What you provide: a generator that produces one candidate per call plus any custom metrics or workflows you need.

What Themis provides: candidate fan-out, reduction, parsing, workflow execution, judge orchestration, persistence, and inspection.

What to inspect when it goes wrong: generation artifacts when the raw candidate is wrong, evaluation executions when scoring or judgment is wrong.
