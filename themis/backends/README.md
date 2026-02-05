# Themis Backends

This module provides pluggable backend interfaces for extending Themis functionality.

## Overview

Themis supports custom backends for:
- **Storage**: Where and how experiment data is persisted
- **Execution**: How tasks are executed (parallel, async, or custom schedulers)

## Quick Start

### Using Built-in Backends

```python
from themis import evaluate
from themis.backends import LocalExecutionBackend

# Use default file storage + local threading
result = evaluate(
    "demo",
    model="gpt-4",
    workers=8,  # Uses LocalExecutionBackend with 8 threads
)
```

### Storage Backends with `evaluate()`

```python
from themis import evaluate
from themis.backends.storage import LocalFileStorageBackend

result = evaluate(
    "demo",
    model="gpt-4",
    storage_backend=LocalFileStorageBackend(".cache/experiments"),
)
```

Notes:
- `evaluate()` currently expects storage backends that are ExperimentStorage-compatible.
- Full custom `StorageBackend` integration is still evolving at the high-level API.

### Custom Execution Backend

```python
from themis import evaluate
from my_backends import RayExecutionBackend

# Distributed execution with Ray
executor = RayExecutionBackend(num_cpus=32)

result = evaluate(
    "math500",
    model="gpt-4",
    execution_backend=executor,  # Custom execution
)
```

## Available Backends

### Storage Backends

- **`LocalFileStorageBackend`** (default): File-based storage with SQLite metadata
  - Fast, reliable, no dependencies
  - Good for: Individual research, local experiments

### Execution Backends

- **`LocalExecutionBackend`** (default): Multi-threaded execution using `ThreadPoolExecutor`
  - Good for: Most use cases, < 1000 samples
  
- **`SequentialExecutionBackend`**: Single-threaded execution
  - Good for: Debugging, testing

## Creating Custom Backends

See [docs/api/backends.md](../../docs/api/backends.md) for backend API documentation.

### Example: S3 Storage

```python
from themis.backends import StorageBackend
import boto3

class S3StorageBackend(StorageBackend):
    def __init__(self, bucket: str):
        self.bucket = bucket
        self.s3 = boto3.client('s3')
    
    def save_generation_record(self, run_id, record):
        key = f"runs/{run_id}/generations/{record.id}.json"
        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=record.to_json(),
        )
    
    # ... implement other methods
```

### Example: Ray Execution

```python
from themis.backends import ExecutionBackend
import ray

class RayExecutionBackend(ExecutionBackend):
    def __init__(self):
        ray.init()
    
    def map(self, func, items, **kwargs):
        @ray.remote
        def remote_func(item):
            return func(item)
        
        futures = [remote_func.remote(item) for item in items]
        for future in futures:
            yield ray.get(future)
    
    def shutdown(self):
        ray.shutdown()
```

## When to Use Custom Backends

**Use custom storage when:**
- You need team collaboration (S3, GCS)
- You want to query results with SQL (PostgreSQL)
- You can provide an ExperimentStorage-compatible adapter

**Use custom execution when:**
- You have > 10,000 samples
- You have multiple machines available
- You need GPU batching
- You want async I/O

**Stick with defaults when:**
- Individual research experiments
- < 1,000 samples per run
- Working on single machine

## Architecture

```
┌─────────────────────────────────────────┐
│           themis.evaluate()              │
│  (High-level API - manages orchestration)│
└────────────────┬────────────────────────┘
                 │
         ┌───────┴────────┐
         │                 │
    ┌────▼─────┐     ┌────▼─────┐
    │ Storage  │     │Execution │
    │ Backend  │     │ Backend  │
    └────┬─────┘     └────┬─────┘
         │                 │
    ┌────▼─────┐     ┌────▼─────┐
    │ LocalFile│     │  Local   │
    │ (default)│     │Threading │
    └──────────┘     │(default) │
                     └──────────┘
         │                 │
    ┌────▼─────┐     ┌────▼─────┐
    │Adapter-  │     │Custom Ray│
    │backed    │     │ Backend  │
    │ Storage  │     │(example) │
    └──────────┘     └──────────┘
```

## Best Practices

1. **Start with defaults** - They're optimized for 90% of use cases
2. **Benchmark first** - Measure before adding complexity
3. **Test thoroughly** - Custom backends should have comprehensive tests
4. **Document edge cases** - Note any limitations or requirements
5. **Handle errors gracefully** - Raise appropriate exceptions

## Contributing

Have a useful backend implementation? Consider contributing it:
1. Implement the interface
2. Add tests
3. Document usage
4. Open a PR!

Example backends to contribute:
- RedisStorageBackend
- DaskExecutionBackend  
- AzureBlobStorageBackend
- AsyncExecutionBackend
