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

Use this capability table when you need to match a backend to the workflow it must support.

| Backend | Best fit | Cross-process reopen | Resume/report ergonomics |
| --- | --- | --- | --- |
| `memory` | tutorials, smoke tests, in-process debugging | no | lowest |
| `sqlite` | most local persisted work | yes | highest for single-machine workflows |
| `jsonl` | file-based operational handoff | yes | workflow-specific |
| `mongodb` | service-backed persistence | yes | environment-specific |
| `postgres` | shared relational persistence | yes | environment-specific |

The main difference is not API shape but how durable and operationally reusable the stored run becomes.

What to inspect when it goes wrong: confirm the store backend supports the workflow you expect, then inspect whether the snapshot and events were persisted where that workflow reads them back.
