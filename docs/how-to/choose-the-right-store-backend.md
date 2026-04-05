---
title: Choose the right store backend
diataxis: how-to
audience: active Themis users
goal: Help readers choose the storage backend that matches persistence, inspection, and operational needs.
---

# Choose the right store backend

Goal: choose between `memory`, `sqlite`, and external store backends.

When to use this:

Use this guide when the run lifecycle matters more than the scoring logic itself.

## Procedure

Choose `memory` when:

- the run is local and short-lived
- you do not need cross-process resume or later reporting

Choose `sqlite` when:

- you want the default persistent local backend
- you need `resume`, `report`, `compare`, or `export`
- you want something easy to inspect and copy locally

Choose `jsonl`, `mongodb`, or `postgres` when your environment or data lifecycle makes those backends a better operational fit.

## Variants

| Variant | Best when | Tradeoff | Related APIs / commands |
| --- | --- | --- | --- |
| Tutorial or smoke-test runs | The run is short-lived and does not need reopen support | No cross-process persistence or later reporting | `memory`, `InMemoryRunStore` |
| Most local persisted work | You want the default persistent backend for resume, report, and compare | Less flexible than environment-specific external stores | `sqlite`, `SqliteRunStore`, `sqlite_store` |
| Environment-driven persistence requirements | Team or infrastructure constraints already require an external backend | More setup and operational dependencies | `jsonl`, `mongodb`, `postgres`, `StorageConfig` |

## Expected result

You should know which store to configure and whether the run can be reopened outside the current process.

## Troubleshooting

- [Store backend model](../explanation/store-backend-model.md)
- [Config schema](../reference/config-schema.md)
- [Resume and inspect runs](resume-and-inspect-runs.md)
