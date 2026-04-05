---
title: Identity vs provenance
diataxis: explanation
audience: users debugging run identity and reproducibility
goal: Clarify which inputs change run_id and which are only recorded as metadata.
---

# Identity vs provenance

What it is: the split between logical run inputs and execution metadata.

When it matters: whenever a `run_id` changes unexpectedly or stays the same when you expected a new logical run.

What you provide: identity-bearing inputs such as dataset refs, component refs, candidate policy, judge config, workflow overrides, and seeds.

What Themis provides: provenance capture for version, platform, runtime, storage, environment metadata, and runtime-only execution wiring such as tracing or subscribers.

Use this split when you need to explain why two runs are logically the same or different.

```mermaid
flowchart TD
    A["Experiment inputs"] --> B["RunSnapshot.identity"]
    A --> C["RunSnapshot.provenance"]
    B --> D["dataset refs and fingerprints"]
    B --> E["component refs"]
    B --> F["candidate policy, judges, seeds"]
    C --> G["platform, version, storage, environment"]
    C --> J["subscribers, tracing backend"]
    B --> H["Changes run_id"]
    C --> I["Recorded metadata only"]
```

If the logical run changed, the difference should appear on the identity side; provenance explains where and how that same logical run happened. Changing a `LifecycleSubscriber` or `TracingProvider` changes runtime observation, not logical run identity.

What to inspect when it goes wrong: look at `RunSnapshot.identity` first. If the logical run should be the same, differences should only appear in `RunSnapshot.provenance`.
