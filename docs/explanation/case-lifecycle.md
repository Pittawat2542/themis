---
title: Case lifecycle
diataxis: explanation
audience: users understanding end-to-end runtime behavior
goal: Explain how one dataset case moves through generation, reduction, parsing, scoring, and persistence.
---

# Case lifecycle

What it is: the path one case takes from dataset input to final score rows and persisted artifacts.

When it matters: whenever you need to understand where data changes shape or where a failure occurred.

What you provide: a case inside a dataset, generation config, evaluation config, and optional seeds.

What Themis provides: planning, candidate fan-out, optional reduction, optional parsing, scoring, persistence, and projection refresh.

```mermaid
flowchart LR
    A["Dataset case"] --> B["Generation"]
    B --> C["Reduction"]
    C --> D["Parsing"]
    D --> E["Scoring / Evaluation workflow"]
    E --> F["Store events and projections"]
```

What to inspect when it goes wrong: generated candidates, reduced candidate output, parsed output, evaluation executions, and projection-backed reports.
