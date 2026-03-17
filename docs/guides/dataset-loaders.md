# Build a Dataset Provider

The benchmark-first dataset contract is `DatasetProvider.scan(slice_spec, query)`.

## Minimum Shape

```python
class MyDatasetProvider:
    def scan(self, slice_spec, query):
        del slice_spec, query
        return [
            {"item_id": "item-1", "question": "2 + 2", "answer": "4"},
        ]
```

Register it when building the orchestrator:

```python
orchestrator = Orchestrator.from_project_spec(
    project,
    registry=registry,
    dataset_provider=MyDatasetProvider(),
)
```

## What `query` Owns

Use `DatasetQuerySpec` to keep selection logic out of the payload:

- subset size and seed
- pinned item IDs
- metadata filters
- provider-specific sampling hints

Example:

```python
SliceSpec(
    slice_id="smoke",
    dataset_query=DatasetQuerySpec.subset(5, seed=13),
    ...
)
```

## Provider Rules

- push the query into the remote system when possible
- return stable `item_id` values
- keep prompt-only configuration out of dataset payloads
- prefer slice dimensions for benchmark semantics such as `source` or `format`
