# Write a Dataset Loader

Themis expects a dataset loader object that satisfies the `DatasetLoader`
protocol: one `load_task_items(task: TaskSpec)` method returning a sequence of
execution items.

Use [Validate Dataset Loaders](dataset-validation.md) before large runs when you
want to inspect IDs, metadata, and JSON-safety directly against the raw items.

## Minimal Contract

```python
class MyDatasetLoader:
    def load_task_items(self, task):
        return [
            {"item_id": "item-1", "question": "6 * 7", "answer": "42"},
            {"item_id": "item-2", "question": "8 * 8", "answer": "64"},
        ]
```

That means:

- accept the resolved `TaskSpec`
- return deterministic items for that task
- keep the return values JSON-safe or `DataItemContext` objects

Each returned item can be:

- a `dict`
- a `DataItemContext`
- a JSON-safe scalar

Mapping-shaped rows are the most ergonomic teaching path because they carry
explicit IDs, payload fields, and `metadata`. `DataItemContext` is the typed
equivalent when you want validation and helper methods. Scalar items remain
valid for minimal tasks, but they are harder to inspect and slice because you
must derive identity from the value itself.

If you return a mapping, Themis will:

- use `item_id` or `id` when present
- otherwise derive a stable fallback ID from the payload
- preserve a string-only `metadata` sub-dict when provided

## Using the `datasets` extra

Install `themis-eval[datasets]` when your loader imports the Hugging Face
`datasets` package:

```bash
uv add "themis-eval[datasets]"
```

Example:

```python
from datasets import load_dataset


class HFDatasetLoader:
    def load_task_items(self, task):
        dataset = load_dataset(task.dataset.dataset_id, split="test")
        return [
            {
                "item_id": row["id"],
                "question": row["question"],
                "answer": row["answer"],
            }
            for row in dataset
        ]
```

Pass this loader explicitly to `Orchestrator`; the extra installs the
dependency, not a hidden loader implementation.

`dataset_loader` is part of the public orchestration boundary. Storage and
projection modules stay internal-by-convention even if they are importable.

## Sampling

`ExperimentSpec.item_sampling` controls how items are selected:

- `kind="all"`
- `kind="subset"` with `count`
- `kind="stratified"` with `count` and `strata_field`
- `item_ids=[...]` to pin the run to explicit rows before sampling
- `metadata_filters={...}` to keep only rows whose string metadata matches

Sampling is deterministic when you pass `seed`.

`item_ids` and `metadata_filters` are applied before subset or stratified
sampling, so you can define reproducible slices like "only hard examples" or
"this exact regression list" without changing the dataset loader itself.

Use [Evolve an Experiment](evolve-an-experiment.md) when you want those slices
to remain stable over time.
