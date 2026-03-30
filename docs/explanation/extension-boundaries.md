---
title: Extension boundaries
diataxis: explanation
audience: users and contributors authoring custom components
goal: Explain what custom components own versus what the Themis runtime owns.
---

# Extension boundaries

What it is: the line between user-owned components and Themis-owned orchestration.

When it matters: whenever a custom component starts to replicate planning, persistence, or workflow execution logic that the runtime already provides.

What you provide: protocol-conforming components with stable identity and focused behavior.

What Themis provides: orchestration, fan-out, evaluation workflows, persistence, and projection-backed inspection.

What to inspect when it goes wrong: check whether the custom component is trying to own orchestration concerns that belong in Themis.
