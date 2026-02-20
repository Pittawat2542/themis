# Storage Guide

Themis stores run artifacts under a storage root (default: `.cache/experiments`).

## Storage Lifecycle

```mermaid
flowchart LR
    A["Start run"] --> B["Persist run metadata"]
    B --> C["Cache generation records"]
    C --> D["Cache evaluation records"]
    D --> E["Mark completed/failed"]
    E --> F["Reuse via resume/cache"]
```

## Configure Storage

### Python API

```python
report = evaluate(
    "gsm8k",
    model="gpt-4",
    storage=".cache/experiments",
    run_id="gsm8k-gpt4",
    resume=True,
)
```

### CLI

```bash
themis eval gsm8k --model gpt-4 --storage .cache/experiments
```

You can also set a default path:

```bash
export THEMIS_STORAGE="~/.themis/experiments"
```

## How to Choose a Storage Path

```mermaid
flowchart TD
    A["Choose storage root"] --> B{"Single-project local work?"}
    B -- "Yes" --> C["Use project-local .cache/experiments"]
    B -- "No" --> D["Use shared stable path (for example ~/.themis/experiments)"]
    C --> E["Keep run IDs descriptive and stable"]
    D --> E
```

## Run Management

```bash
# list runs
themis list runs --verbose

# preview cleanup
themis clean --older-than 30 --dry-run

# delete old runs
themis clean --older-than 30
```

## Practical Recommendations

- Use explicit `run_id` values to make comparisons repeatable.
- Keep one storage root per environment (dev/staging/prod) to avoid mixing runs.
- Leave `resume=True` when iterating on prompt/model tweaks to save cost.
- Use `--dry-run` before cleanup commands.

## Stored Artifacts

Typical per-run artifacts include generation/evaluation caches and exported reports.
Use `themis share <RUN_ID>` to create shareable badge + markdown outputs.
