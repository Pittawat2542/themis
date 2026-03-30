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

What Themis provides: provenance capture for version, platform, runtime, storage, and environment metadata.

What to inspect when it goes wrong: look at `RunSnapshot.identity` first. If the logical run should be the same, differences should only appear in `RunSnapshot.provenance`.
