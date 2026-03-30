---
title: Store backend model
diataxis: explanation
audience: users choosing persistence behavior
goal: Explain how different store backends shape inspection, resume, and operational workflows.
---

# Store backend model

What it is: the persistence model behind `memory`, `sqlite`, `jsonl`, `mongodb`, and `postgres` backends.

When it matters: whenever the run must survive process boundaries or integrate with a larger environment.

What you provide: the backend choice and backend-specific parameters.

What Themis provides: a common `RunStore` contract so planning, execution, reporting, and inspection can remain backend-agnostic.

What to inspect when it goes wrong: confirm the store backend supports the workflow you expect, then inspect whether the snapshot and events were persisted where that workflow reads them back.
