---
title: Public Experiment and snapshot boundaries
diataxis: explanation
audience: maintainers and extension authors
goal: Record the v6 public API boundary decision.
---

# ADR-0003: Public Experiment and snapshot boundaries

## Status

Accepted for Themis 6.0.0.

## Context

Earlier releases exposed both a convenience experiment and a core experiment.
Catalogs, presets, launchers, and planning helpers could return or require
different types. That split made an experiment authored through one entry path
difficult to inspect or reuse through another, and encouraged application code
to import private runtime models.

## Decision

`themis.Experiment` is the only public authoring model. Convenience helpers,
catalogs, benchmark kits, suites, and presets produce it. Config and CLI are
automation entry paths around an importable Python `Experiment`; custom
components and adapters plug into it as extension points.

`Experiment.compile(...)` produces the public, deeply immutable `RunSnapshot`.
The snapshot is the planning and execution handoff. Runtime orchestration,
evidence projection machinery, and other implementation models remain private.
Stores implement the public `RunStore` protocol, with `RunStoreBase` available
to backend authors. Analysis consumes stored snapshots and evidence.

## Consequences

- Python APIs that returned the old core experiment are intentionally broken.
- Catalog builders no longer accept embedded storage or runtime arguments.
- Runtime settings are supplied as `RunOptions` at compile or run time.
- Presets return a `PresetApplication` containing experiment, options, and ids.
- Stored snapshot schemas and `run_id` computation remain compatible.
- User documentation and examples may not import `themis.core`.
