# Advanced Workflows

Use this page when the quick-start path is no longer enough and you need to
reason about partial reruns, external systems, scaling, or post-run diagnosis.
Each section routes to one canonical guide or concept page for the workflow.

## Resume and Rerun Safely

When you rerun the same experiment against the same storage root, Themis checks
deterministic stage identities and skips completed work instead of recomputing
everything.

What Themis does:

- stores stage-level projections and run manifests
- skips completed generation, transform, and evaluation overlays independently
- exposes `plan()`, `resume()`, and `diff_specs()` so you can see what changed

What you provide:

- stable `ProjectSpec` and `ExperimentSpec` inputs
- the decision about whether a change should count as a new run or a resume

Canonical pages:

- [Resume and Inspect Runs](resume-and-inspect.md)
- [Storage and Resume](../concepts/storage-and-resume.md)
- [Evolve an Experiment](evolve-an-experiment.md)

## Hand Off Work to External Systems

Themis can keep deterministic planning, storage, and result analysis even when
generation or evaluation happens somewhere else.

What Themis does:

- exports only missing generation or evaluation work items
- persists run manifests and pending stage work
- imports Themis-compatible generation or evaluation records back into storage

What you provide:

- the external engine, batch job, judge pipeline, or worker system
- the adapter script that maps external outputs back into `TrialRecord` /
  `CandidateRecord` / `EvaluationRecord`

Canonical page:

- [Hand Off Generation or Evaluation](external-stage-handoffs.md)

## Evolve an Experiment Over Time

Themis treats experiment specs as immutable snapshots. When you add a model,
prompt, metric, or slice later, only the new deterministic work should remain
pending.

What Themis does:

- hashes models, prompts, params, transforms, and evaluations into stable IDs
- reuses existing generation artifacts when only downstream stages change
- lets you diff two experiment specs before running them

What you provide:

- the updated experiment matrix
- explicit versioning for your experiment builder code and project file

Canonical pages:

- [Evolve an Experiment](evolve-an-experiment.md)
- [Specs and Records](../concepts/specs-and-records.md)

## Scale Execution

Themis can schedule many work items concurrently and persist pending work for
async backends, but it does not hide provider-specific throughput and retry
limits.

What Themis does:

- bounds in-flight stage work with `ExecutionPolicySpec.max_in_flight_work_items`
- persists worker-pool and batch run manifests in shared storage
- keeps retry and resume state in the same run metadata

What you provide:

- the engine-level concurrency, rate limiting, and provider retry logic
- any external workers or provider-specific batch adapters

Canonical page:

- [Scale Execution](scaling-execution.md)

## Analyze Failures and Compare Runs

Stored projections support both aggregate comparison and targeted qualitative
inspection after the run finishes.

What Themis does:

- builds leaderboard rows, reports, and paired comparisons from projections
- surfaces invalid extractions, failed candidates, and tagged examples
- keeps timeline views for trial- and candidate-level drilldown

What you provide:

- the tagging conventions in metric details or candidate payloads
- any downstream dashboarding or custom database ingestion on top of the stored
  SQLite/Postgres data

Canonical pages:

- [Compare and Export Results](compare-and-export.md)
- [Analyze Results](../tutorials/analyze-results.md)
- [Storage and Resume](../concepts/storage-and-resume.md)

## Quick Map

Use these pages when you need a direct answer:

- Resume an interrupted eval without rerunning generation: [Storage and Resume](../concepts/storage-and-resume.md)
- Run generation only or evaluation only: [Hand Off Generation or Evaluation](external-stage-handoffs.md)
- Add new models, prompts, slices, or metrics later: [Evolve an Experiment](evolve-an-experiment.md)
- Estimate work or token budget before running: [Resume and Inspect Runs](resume-and-inspect.md)
- Compare models with confidence intervals and p-values: [Compare and Export Results](compare-and-export.md)
- Diagnose null extractions, tagged failures, or weak examples: [Analyze Results](../tutorials/analyze-results.md)
