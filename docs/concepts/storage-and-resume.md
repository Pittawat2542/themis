# Storage and Resume

Themis persists append-only events plus read-side projection tables in SQLite by
default.

## What Gets Stored

- concrete trial events
- projected trial summaries
- candidate score rows
- benchmark semantics: `benchmark_id`, `slice_id`, `prompt_variant_id`, and dimensions

## Why Resume Works

Themis hashes the compiled execution plan. When a rerun targets the same
storage root and the same benchmark shape, completed work is reused.

Typical changes:

- add a prompt variant: only new prompt combinations run
- add a model: only new model combinations run
- add a score overlay: existing generation can be reused
- change a query or dimensions: affected slices replan deterministically

Example: `examples/05_resume_run.py` shows repeated runs against the same
benchmark storage root.
