# Write a Dataset Loader

Themis expects a dataset loader object with a `load_task_items(task)` method.

## Minimal Contract

```python
class MyDatasetLoader:
    def load_task_items(self, task):
        return [
            {"item_id": "item-1", "question": "6 * 7", "answer": "42"},
            {"item_id": "item-2", "question": "8 * 8", "answer": "64"},
        ]
```

Each returned item can be:

- a `dict`
- a `DataItemContext`
- a JSON-safe scalar

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

Themis still expects you to pass this loader explicitly to `Orchestrator`; the
extra installs the dependency, not a hidden loader implementation.

## Sampling

`ExperimentSpec.item_sampling` controls how items are selected:

- `kind="all"`
- `kind="subset"` with `count`
- `kind="stratified"` with `count` and `strata_field`

Sampling is deterministic when you pass `seed`.
