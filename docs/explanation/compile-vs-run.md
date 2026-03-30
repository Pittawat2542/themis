---
title: Compile vs run
diataxis: explanation
audience: users distinguishing definition time from execution time
goal: Explain what compile freezes and what execution still controls.
---

# Compile vs run

What it is: the boundary between creating a `RunSnapshot` and executing it.

When it matters: whenever you need to reason about `run_id`, resume semantics, or runtime tuning.

What you provide: compile-time experiment inputs plus optional runtime overrides at execution time.

What Themis provides: immutable snapshot compilation and orchestrated execution over that snapshot.

What to inspect when it goes wrong: inspect the compiled snapshot first, then inspect execution state, stored events, and runtime settings.
