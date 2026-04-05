# Reason About Execution And Persistence

Use this file when the task touches `run_id`, stores, replay, inspection, or CLI behavior.

## Compile Versus Run

- `compile()` freezes a `RunSnapshot` and its `run_id`.
- `run()` executes work against that frozen snapshot.
- `RuntimeConfig` changes execution behavior, not experiment identity.
- Inspect the compiled snapshot first when it is unclear whether a change should alter logical run identity.

## Identity Versus Provenance

Identity-bearing inputs change `run_id`. Important examples:

- dataset refs and fingerprints
- component refs
- candidate policy
- judge config and workflow overrides
- seeds
- prompt specs used by generation or judge workflows

Provenance captures how and where the run happened without changing `run_id`. Important examples:

- platform and package version
- storage parameters
- environment metadata
- tracing and subscriber wiring
- concurrency, retry, and rate-limit settings

If `run_id` drift is unexpected, compare `RunSnapshot.identity` before debugging runtime behavior.

Stage caches are keyed by stage inputs, not by `run_id`.

## Store Selection

- `memory`: local and short-lived only
- `sqlite`: default persistent local backend
- `jsonl`, `mongodb`, `postgres`: use when the environment or operational model requires them

Persistent stores are required for:

- cross-process resume
- reporting and comparison
- export and import workflows
- replay from stored upstream artifacts
- cross-run cache reuse

`InMemoryRunStore` does not provide cross-process persistence or cross-run stage-cache behavior.

`existing_run_policy` controls what happens when the same compiled `run_id` is
submitted again:

- `auto`: reuse completed runs and resume incomplete runs
- `error`: fail if the compiled run already exists
- `rerun`: clear stored state and run again

## Replay And Rejudge

- `Experiment.replay(stage="judge")` reruns downstream judge workflows from stored upstream artifacts.
- `Experiment.rejudge()` is shorthand for `replay(stage="judge")`.
- Downstream stage bundle handoff for reduction, parse, and score exists in Python even when the CLI surface is narrower.
- Imported artifacts are normalized back into standard event history, so reporting and resume treat them like locally produced data.

## CLI Boundary

Primary command groups:

- `quick-eval`
- `run`
- `replay`
- `submit`
- `resume`
- `estimate`
- `report`
- `inspect`
- `quickcheck`
- `compare`
- `export`
- `init`
- `worker`
- `batch`

Important behavior notes:

- `quick-eval benchmark --name ...` executes a shipped named benchmark recipe
  through the catalog runtime.
- `run --config ... [--until-stage ...]` executes and prints compact JSON.
- `replay --config ... --stage reduce|parse|score|judge` reruns downstream
  stages from stored upstream artifacts.
- `resume --config ...` and `inspect ...` require a persistent store unless the original memory store instance is still in process.
- `report` and exported score tables include outcome and error fields alongside metric values.
- The CLI currently exposes only `generation` and `evaluation` bundle export, even though Python APIs support reduction, parse, and score bundles too.

## Validation Heuristics

- Use deterministic builtins when you only need to validate experiment wiring.
- Use `sqlite` instead of `memory` when a test spans multiple commands or processes.
- Prefer targeted tests around the changed stage boundary instead of broad end-to-end provider runs.
