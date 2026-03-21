# Validate Dataset Loaders

Use this guide before a large run when you want to verify that your
`dataset_loader` is deterministic, JSON-safe, and compatible with sampling.
For query-aware dataset providers, the same determinism requirement applies to
`scan(slice_spec, query)` as well.

## Start From the Loader Directly

Use the same loader object you plan to pass to `Orchestrator`:

```python
items = list(dataset_loader.load_task_items(experiment.tasks[0]))
print(len(items))
print(items[0])
```

Expected outcome:

- the sequence length is stable across repeated calls
- each item is a JSON-safe scalar, mapping, or `DataItemContext`
- the first few rows look like real benchmark items rather than provider payloads

## Check IDs and Metadata Early

Use type-aware checks so the validation path matches the item shape your loader
actually returns. The examples below prefer explicit `Mapping` checks first,
then fall back to `DataItemContext`, then scalar item handling.

```python
from collections.abc import Mapping

from themis.specs import DataItemContext


for item in items[:3]:
    if isinstance(item, Mapping):
        print(item.get("item_id") or item.get("id"))
        print(item.get("metadata", {}))
    elif isinstance(item, DataItemContext):
        print(item.item_id)
        print(item.metadata)
    else:
        print("scalar item", item)
```

Themis expects:

- `item_id` or `id` on mappings, or `item_id` on `DataItemContext`, when you
  want stable human-readable IDs
- `metadata` as a string-to-string mapping when you plan to use
  `metadata_filters`

If IDs are missing, Themis derives one from the payload. That is deterministic,
but it is harder to debug and harder to share with teammates.

## Validate JSON-Safe Payloads

The runtime persists dataset context, event metadata, and exported bundles as
JSON. Catch non-serializable values before the run:

```python
from themis.types.json_validation import validate_json_value

for item in items:
    validate_json_value(item)
```

Typical bad payloads are:

- raw `datetime` objects
- custom classes
- sets
- nested metadata values that are not strings

## Preview Sampling Inputs

Sampling happens after `item_ids` and `metadata_filters` are applied. Validate
those fields against the raw items before you trust a slice. Use duck-typing
only after you have ruled out the explicit `Mapping` or `DataItemContext`
shapes:

```python
from collections.abc import Mapping

from themis.specs import DataItemContext

hard_items = [
    item
    for item in items
    if (
        (
            isinstance(item, Mapping)
            and ((item.get("metadata", {}) or {}).get("difficulty") == "hard")
        )
        or (
            isinstance(item, DataItemContext)
            and item.metadata.get("difficulty") == "hard"
        )
    )
]
print(len(hard_items))
```

Then mirror the same intent in `DatasetQuerySpec` on the relevant slice:

```python
from themis import DatasetQuerySpec

benchmark = benchmark.model_copy(
    update={
        "slices": [
            benchmark.slices[0].model_copy(
                update={
                    "dataset_query": DatasetQuerySpec(
                        kind="subset",
                        count=100,
                        seed=7,
                        metadata_filters={"difficulty": "hard"},
                    )
                }
            )
        ]
    }
)
```

## Common Failure Patterns

- changing item order between runs: breaks reproducibility and makes slices drift
- treating `seed=None` as permission to randomize: Themis expects deterministic
  order-based behavior when count-based sampling omits a seed
- non-string metadata values: makes `metadata_filters` ineffective
- missing IDs on regression sets: makes failure triage painful
- huge provider objects in the payload: bloats storage and export bundles
- relying on `item.get(...)` for every loader shape: breaks as soon as the
  loader returns `DataItemContext` or a scalar item

## Next Steps

- Use [Write a Dataset Loader](dataset-loaders.md) for the core loader contract.
- Use [Evolve an Experiment](evolve-an-experiment.md) when you want deterministic
  dataset slices over time.
