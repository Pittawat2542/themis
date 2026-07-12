---
title: Launcher config reference
diataxis: reference
audience: automation and CLI users
goal: Document the operational launcher format without duplicating experiment meaning.
---

# Launcher config reference

Python is the canonical experiment definition. YAML and TOML only locate that
definition and supply operational settings:

The required shape is `definition: module:symbol`.

```yaml
definition: experiments.baseline:experiment
storage:
  target: sqlite
  kwargs:
    path: runs/themis.sqlite3
runtime:
  max_concurrency: 8
  provider_timeout_seconds: 120
  existing_run_policy: reuse
```

## Launcher fields

| Field | Required | Meaning | Affects `run_id` |
| --- | --- | --- | --- |
| `definition` | Yes | Import path to an `Experiment` object or zero-argument factory | The imported experiment does |
| `storage.target` | No | `memory`, `sqlite`, `jsonl`, `mongodb`, or `postgres` | No |
| `storage.kwargs` | No | Backend constructor settings; relative paths resolve beside the launcher | No |
| `runtime` | No | Fields accepted by `RunOptions` | No |

Notable runtime defaults are `evidence_retention: standard`,
`strict_determinism: false`, a 30-second persistence timeout, a 5-second
subscriber timeout, and an evidence queue capacity of 256. An omitted
`provider_rate_limits` entry means unlimited client-side RPM for that provider.

Credentials should use environment or secret-provider references and never be
embedded in an experiment identity or launcher committed to source control.
