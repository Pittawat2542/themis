---
title: Choose your storage backend
diataxis: landing
audience: users deciding how runs should persist
goal: Help readers decide between memory, sqlite, and external store backends.
---

# Choose your storage backend

Use `memory` when you want the smallest local run and you do not need to reopen the run from another process.

Use `sqlite` when you want persisted runs, later reporting, resume across processes, or a default local backend that works well for most single-machine workflows.

Use `jsonl`, `mongodb`, or `postgres` when you have a specific operational requirement that makes those stores a better fit than SQLite.

Use this chooser when persistence and later inspection matter more than the scoring logic itself.

```mermaid
flowchart TD
    A["What must the run survive?"]
    A --> B{"Only this process?"}
    B -->|Yes| C["memory"]
    B -->|No| D{"Single-machine local workflow?"}
    D -->|Yes| E["sqlite"]
    D -->|No| F["jsonl / mongodb / postgres"]
```

The decision is mostly about whether later resume, reporting, or handoff must happen outside the current process.

Decision rule:

- tutorials and smoke tests: `memory`
- most real local work: `sqlite`
- operational or integration-specific needs: external stores

Next:

- [First persisted run](../tutorials/first-persisted-run.md)
- [Choose the right store backend](../how-to/choose-the-right-store-backend.md)
- [Store backend model](../explanation/store-backend-model.md)
