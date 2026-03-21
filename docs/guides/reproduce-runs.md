# Reproduce and Share Runs

Use this guide when you need to hand a run to a teammate, archive the exact
configuration for a paper, or rebuild analysis from persisted storage.

## What To Share

For a reproducible handoff, share:

- the project file or inline `ProjectSpec` payload
- the exact `ProjectSpec.global_seed`
- the `run_id` from `submit()` or `plan()`
- the storage root path, or a redacted Postgres connection description plus blob root
- a config report for the exact run snapshot
- any exported JSON or Markdown artifacts you expect reviewers to read

Do not send live Postgres credentials in docs-driven handoff bundles. Share the
database host, database name, and any required environment-variable names
instead of a secret-bearing DSN.

## Generate a Config Report From a Persisted Run

```bash
themis report \
  --project-file project.toml \
  --run-id <run_id> \
  --format markdown \
  --output config-report.md
```

Expected output artifact:

```text
config-report.md
```

Use the persisted-run mode when you want the exact stored `RunManifest`
snapshot rather than the current local Python code.

When you need to confirm the exact generation request that executed, inspect the
candidate-level `effective_seed` and `effective_inference_params_hash` fields
from the stored projection or candidate timeline view. Judge-backed evaluations
persist their executed seeded config inside the judge audit trail.

## Reload the Run in Python

```python
from themis import Orchestrator
from themis.orchestration.run_manifest import RunHandle
from themis.runtime import ExperimentResult

# Recreate `orchestrator` from the same project wiring used for the original
# run, for example with `Orchestrator.from_project_spec(...)`.
resumed = orchestrator.resume(run_id)

if isinstance(resumed, RunHandle):
    print(resumed.run_id, resumed.status, resumed.pending_work_items)
elif isinstance(resumed, ExperimentResult):
    print(resumed.trial_hashes[:3])
```

If no work is left, `resume(run_id)` returns the final `ExperimentResult`. If
pending work remains, it returns a `RunHandle`.

## Export a Portable Analysis Payload

```python
# `result` is the `ExperimentResult` returned by `orchestrator.run(...)` or by
# `orchestrator.resume(run_id)` once the run has fully completed.
evaluation_result = result.for_evaluation(result.evaluation_hashes[0])
payload = evaluation_result.export_json("result.json")
print(payload["overlay"])
```

Use this when you need a warehouse-ready JSON payload or want to attach the
active overlay selection to a handoff artifact.

## Minimum Portable Bundle

For a teammate or external reviewer, package:

- `project.toml` or the equivalent `ProjectSpec` payload
- `config-report.md`
- `report.md`, `report.csv`, or both
- `result.json`
- the storage root path or the instructions needed to mount it
- the `run_id`

If the handoff depends on local blobs or SQLite artifacts, ship the storage root
directory itself rather than only the high-level report files.

## Inspect the Run Without Hydrating Everything

Use quickcheck for a lightweight operator view:

```bash
themis-quickcheck scores \
  --db .cache/themis-examples/04-compare-models-benchmark-first/themis.sqlite3 \
  --metric exact_match
```

Expected output:

```text
ev:<evaluation-hash>	baseline	qa	exact_match	0.5000	4
ev:<evaluation-hash>	candidate	qa	exact_match	1.0000	4
```

## Recommended Handoff Bundle

For paper or review workflows, a practical bundle is the minimum portable bundle
above plus the exact example DB or artifact directory when you expect someone
else to re-run `themis-quickcheck` or hydrate timelines locally.

## Track Tool Handler Versions

When running agent evaluations with tool-using models, record which version of
each tool handler was active so the execution trace remains interpretable after
the handler logic changes:

```python
from themis.specs.experiment import RuntimeContext

runtime = RuntimeContext(
    tool_handlers={"search": my_search_fn, "calculator": my_calc_fn},
    tool_handler_versions={"search": "2.1.0", "calculator": "0.4.0"},
)
result = orchestrator.run_benchmark(benchmark, runtime=runtime)
```

`tool_handler_versions` is serialised and stored alongside the rest of the
runtime context, so it appears in config reports and timeline views. Unlike
`tool_handlers`, which is excluded from serialisation, the versions field is
persisted and queryable.

## Detect Spec Changes That Invalidate Resume Work

Before applying a modified `BenchmarkSpec` to an existing storage root, check
whether completed trials would be lost:

```python
diff = orchestrator.diff_specs(old_benchmark, new_benchmark)
if diff.has_invalidated_resume_work:
    print(f"Warning: {len(diff.removed_trial_hashes)} completed trial(s) would be abandoned.")
```

`RunDiff.has_invalidated_resume_work` is `True` whenever `removed_trial_hashes`
is non-empty.  Adding a new model or prompt variant never triggers this flag —
only removing or replacing existing ones does.

## Next Steps

- Use [Generate Config Reports](config-reports.md) for the full report API.
- Use [Compare and Export Results](compare-and-export.md) for report and JSON
  exports.
- Use [Use the Quickcheck CLI](quickcheck.md) for fast SQLite inspection.
