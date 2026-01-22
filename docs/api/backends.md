# Backends API

Pluggable interfaces for custom storage and execution.

## Overview

Themis provides abstract interfaces for:
- **StorageBackend**: Where results are stored
- **ExecutionBackend**: How tasks are executed

## Execution Backends

### ExecutionBackend

Abstract interface for execution strategies.

```python
from themis.backends import ExecutionBackend

class MyBackend(ExecutionBackend):
    def map(self, func, items, max_workers=None, timeout=None, **kwargs):
        # Your execution logic
        for result in execute(func, items):
            yield result
    
    def shutdown(self):
        # Cleanup
        pass
```

---

### LocalExecutionBackend

Multi-threaded execution using `ThreadPoolExecutor` (default).

```python
from themis.backends import LocalExecutionBackend

backend = LocalExecutionBackend(max_workers=8)

# Use directly
results = list(backend.map(lambda x: x * 2, [1, 2, 3]))

# Or in evaluate()
from themis import evaluate

result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    execution_backend=backend,
)
```

**Parameters:**
- **`max_workers`** : `int` - Number of worker threads

---

### SequentialExecutionBackend

Single-threaded execution for debugging.

```python
from themis.backends import SequentialExecutionBackend

backend = SequentialExecutionBackend()

# Results are yielded in order
results = list(backend.map(func, items))
```

**Use for:**
- Debugging
- Testing
- When parallelism causes issues

---

## Storage Backends

### StorageBackend

Abstract interface for storage implementations.

```python
from themis.backends import StorageBackend

class S3Storage(StorageBackend):
    def save_run_metadata(self, run_id, metadata):
        # Upload to S3
        pass
    
    def load_run_metadata(self, run_id):
        # Download from S3
        pass
    
    def save_generation_record(self, run_id, record):
        # Save record
        pass
    
    # ... implement other methods
```

**Required Methods:**

- `save_run_metadata(run_id, metadata)` - Save run info
- `load_run_metadata(run_id)` - Load run info
- `save_generation_record(run_id, record)` - Save generation
- `load_generation_records(run_id)` - Load generations
- `save_evaluation_record(run_id, record)` - Save evaluation
- `load_evaluation_records(run_id)` - Load evaluations
- `save_report(run_id, report)` - Save report
- `load_report(run_id)` - Load report
- `list_runs()` - List all runs
- `run_exists(run_id)` - Check if run exists
- `delete_run(run_id)` - Delete run
- `close()` - Cleanup (optional)

---

### LocalFileStorageBackend

Adapter for existing file-based storage.

```python
from themis.backends import LocalFileStorageBackend

backend = LocalFileStorageBackend(storage_path=".cache")

# Use in evaluate()
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    storage_backend=backend,
)
```

---

## Using Custom Backends

### Example: S3 Storage

```python
from themis.backends import StorageBackend
from themis import evaluate
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
            Body=json.dumps(record.__dict__),
        )
    
    # ... implement other methods

# Use it
result = evaluate(
    benchmark="gsm8k",
    model="gpt-4",
    storage_backend=S3StorageBackend(bucket="my-experiments"),
)
```

### Example: Ray Execution

```python
from themis.backends import ExecutionBackend
from themis import evaluate
import ray

class RayExecutionBackend(ExecutionBackend):
    def __init__(self, num_cpus=None):
        if not ray.is_initialized():
            ray.init(num_cpus=num_cpus)
    
    def map(self, func, items, **kwargs):
        @ray.remote
        def remote_func(item):
            return func(item)
        
        futures = [remote_func.remote(item) for item in items]
        for future in futures:
            yield ray.get(future)
    
    def shutdown(self):
        pass

# Use it
result = evaluate(
    benchmark="math500",
    model="gpt-4",
    execution_backend=RayExecutionBackend(num_cpus=32),
)
```

---

## Context Manager Protocol

Backends support context managers:

```python
from themis.backends import LocalExecutionBackend

with LocalExecutionBackend(max_workers=8) as backend:
    results = list(backend.map(func, items))
# Automatically calls shutdown()
```

---

## Thread Safety

### Storage Backends

Storage backends should be thread-safe:

```python
class ThreadSafeStorage(StorageBackend):
    def __init__(self):
        self.lock = threading.Lock()
    
    def save_generation_record(self, run_id, record):
        with self.lock:
            # Thread-safe write
            self._write(run_id, record)
```

### Execution Backends

Execution backends handle concurrency internally:

```python
class LocalExecutionBackend(ExecutionBackend):
    def map(self, func, items, max_workers=None, **kwargs):
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Handles thread safety
            futures = [executor.submit(func, item) for item in items]
            for future in as_completed(futures):
                yield future.result()
```

---

## Testing Backends

### Test Storage

```python
def test_storage_backend():
    backend = MyStorageBackend()
    
    # Test save/load
    record = GenerationRecord(id="test", prompt="...", response="...")
    backend.save_generation_record("test-run", record)
    
    loaded = backend.load_generation_records("test-run")
    assert len(loaded) == 1
    assert loaded[0].id == "test"
```

### Test Execution

```python
def test_execution_backend():
    backend = MyExecutionBackend()
    
    # Test parallel execution
    results = list(backend.map(lambda x: x * 2, [1, 2, 3]))
    assert set(results) == {2, 4, 6}
    
    backend.shutdown()
```

---

## See Also

- [Extending Backends Guide](../EXTENDING_BACKENDS.md) - Complete guide
- [Examples](../EXTENDING_BACKENDS.md#example-implementations) - Working implementations
- [Storage Architecture](../STORAGE.md) - Storage V2 details
