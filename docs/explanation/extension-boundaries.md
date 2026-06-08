---
title: Extension boundaries
diataxis: explanation
audience: users and contributors authoring custom components
goal: Explain what custom components own versus what the Themis runtime owns.
---

# Extension boundaries

What it is: the line between user-owned components and Themis-owned orchestration.

When it matters: whenever a custom component starts to replicate planning, persistence, or workflow execution logic that the runtime already provides.

What you provide: protocol-conforming components, store backends, reporters, or instrumentation with focused behavior.

What Themis provides: orchestration, fan-out, evaluation workflows, persistence, and projection-backed inspection.

Use this ownership map when a custom component starts to feel broader than one protocol boundary.

```mermaid
flowchart LR
    A["User-owned extensions"] --> B["Generator / Selector / Reducer / Parser / Metric / Judge"]
    A --> G["Store / Reporter / Instrumentation"]
    B --> C["Themis runtime"]
    G --> C
    C --> D["Planning and fan-out"]
    C --> E["Workflow execution"]
    C --> F["Persistence and inspection"]
```

Custom extensions should supply behavior at one boundary, not absorb orchestration responsibilities that belong to the runtime. Reporters are replaceable read-side boundaries over persisted evidence and projections; changing a reporter does not change `run_id`.

What to inspect when it goes wrong: check whether the custom component is trying to own orchestration concerns that belong in Themis.
