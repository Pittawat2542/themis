---
title: Fan-out, reduction, parsing, and scoring
diataxis: explanation
audience: users composing multi-stage experiments
goal: Explain why Themis models these stages separately and what each one contributes.
---

# Fan-out, reduction, parsing, and scoring

What it is: the staged model that separates candidate multiplicity, reduced-candidate selection, output normalization, and final scoring.

When it matters: whenever `num_samples` is greater than one or metrics need a parsed shape that differs from raw model output.

What you provide: candidate policy, optional reducer, optional parser, and metrics.

What Themis provides: ordered stage execution, typed contexts, and persisted artifacts between stages.

Use this pipeline when you need to see exactly where the output changes shape.

```mermaid
flowchart LR
    A["One dataset case"] --> B["Fan out N candidates"]
    B --> C["Reduce to one candidate or summary"]
    C --> D["Parse into metric-ready shape"]
    D --> E["Score or run workflow-backed evaluation"]
    E --> F["Persist artifacts and projections"]
```

Each stage exists so multiplicity, selection, normalization, and scoring can evolve independently without collapsing into one overloaded component.

What to inspect when it goes wrong: generated candidates first, then reduction output, then parsed output, then scores or evaluation executions.
