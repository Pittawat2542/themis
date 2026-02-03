# Extending Themis with Custom Backends

Themis provides pluggable backend interfaces that allow you to customize storage and execution without modifying core code. This guide shows you how to implement custom backends for advanced use cases.

## Table of Contents

- [Storage Backends](#storage-backends)
- [Execution Backends](#execution-backends)
- [Example Implementations](#example-implementations)
- [Testing Your Backend](#testing-your-backend)

---

## Storage Backends

Storage backends control where and how experiment data is persisted.

### Use Cases

- **Cloud Storage**: Store results in S3, GCS, or Azure Blob Storage for team collaboration
- **Databases**: Use PostgreSQL, MongoDB, or other databases for structured querying
- **Distributed Cache**: Use Redis or Memcached for fast distributed access
- **Custom Formats**: Write data in custom formats for integration with existing systems

### Interface Definition

```python
from themis.backends import StorageBackend
from themis.core.entities import (
    GenerationRecord, EvaluationRecord, ExperimentReport
)
from typing import Any, Dict

class MyStorageBackend(StorageBackend):
    def save_run_metadata(self, run_id: str, metadata: Dict[str, Any]) -> None:
        """Save run metadata."""
        pass
    
    def load_run_metadata(self, run_id: str) -> Dict[str, Any]:
        """Load run metadata."""
        pass
    
    def save_generation_record(self, run_id: str, record: GenerationRecord) -> None:
        """Save a generation record (must be thread-safe)."""
        pass
    
    def load_generation_records(self, run_id: str) -> list[GenerationRecord]:
        """Load all generation records for a run."""
        pass
    
    def save_evaluation_record(self, run_id: str, record: EvaluationRecord) -> None:
        """Save an evaluation record (must be thread-safe)."""
        pass
    
    def load_evaluation_records(self, run_id: str) -> dict[str, EvaluationRecord]:
        """Load all evaluation records (returns dict of cache_key -> record)."""
        pass
    
    def save_report(self, run_id: str, report: ExperimentReport) -> None:
        """Save experiment report."""
        pass
    
    def load_report(self, run_id: str) -> ExperimentReport:
        """Load experiment report."""
        pass
    
    def list_runs(self) -> list[str]:
        """List all run IDs."""
        pass
    
    def run_exists(self, run_id: str) -> bool:
        """Check if run exists."""
        pass
    
    def delete_run(self, run_id: str) -> None:
        """Delete all data for a run."""
        pass
    
    def close(self) -> None:
        """Optional: cleanup resources."""
        pass
```

### Example: S3 Storage Backend

```python
import json
import boto3
from themis.backends import StorageBackend
from themis.core.entities import GenerationRecord, EvaluationRecord

class S3StorageBackend(StorageBackend):
    """Store experiment data in AWS S3."""
    
    def __init__(self, bucket: str, prefix: str = "themis-experiments"):
        self.bucket = bucket
        self.prefix = prefix
        self.s3 = boto3.client('s3')
    
    def _key(self, run_id: str, *parts: str) -> str:
        """Build S3 key."""
        return "/".join([self.prefix, run_id, *parts])
    
    def save_run_metadata(self, run_id: str, metadata: dict[str, Any]) -> None:
        """Save run metadata to S3."""
        key = self._key(run_id, "metadata.json")
        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=json.dumps(metadata),
            ContentType="application/json",
        )
    
    def load_run_metadata(self, run_id: str) -> dict[str, Any]:
        """Load run metadata from S3."""
        key = self._key(run_id, "metadata.json")
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=key)
            data = json.loads(response['Body'].read())
            return data
        except self.s3.exceptions.NoSuchKey:
            raise FileNotFoundError(f"Run metadata not found: {run_id}")
    
    def save_generation_record(self, run_id: str, record: GenerationRecord) -> None:
        """Append generation record to JSONL file in S3."""
        key = self._key(run_id, "generations.jsonl")
        
        # S3 doesn't support append, so we need to:
        # 1. Download existing file
        # 2. Append new record
        # 3. Upload back
        # For production, consider using S3 streaming or batching
        
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=key)
            existing = response['Body'].read().decode('utf-8')
        except self.s3.exceptions.NoSuchKey:
            existing = ""
        
        new_line = json.dumps(record.__dict__) + "\n"
        updated = existing + new_line
        
        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=updated,
            ContentType="application/x-ndjson",
        )
    
    def load_generation_records(self, run_id: str) -> list[GenerationRecord]:
        """Load all generation records from S3."""
        key = self._key(run_id, "generations.jsonl")
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=key)
            content = response['Body'].read().decode('utf-8')
            
            records = []
            for line in content.strip().split('\n'):
                if line:
                    data = json.loads(line)
                    records.append(GenerationRecord(**data))
            return records
        except self.s3.exceptions.NoSuchKey:
            return []
    
    # ... implement other methods similarly
    
    def list_runs(self) -> list[str]:
        """List all runs by listing S3 prefixes."""
        response = self.s3.list_objects_v2(
            Bucket=self.bucket,
            Prefix=self.prefix + "/",
            Delimiter="/",
        )
        
        runs = []
        for prefix in response.get('CommonPrefixes', []):
            run_id = prefix['Prefix'].rstrip('/').split('/')[-1]
            runs.append(run_id)
        return runs
    
    def run_exists(self, run_id: str) -> bool:
        """Check if run exists by checking for metadata."""
        key = self._key(run_id, "metadata.json")
        try:
            self.s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except:
            return False
```

### Example: PostgreSQL Storage Backend

```python
import json
import psycopg2
from themis.backends import StorageBackend
from themis.core.entities import GenerationRecord

class PostgresStorageBackend(StorageBackend):
    """Store experiment data in PostgreSQL."""
    
    def __init__(self, connection_string: str):
        self.conn = psycopg2.connect(connection_string)
        self._create_tables()
    
    def _create_tables(self):
        """Create tables if they don't exist."""
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id VARCHAR(255) PRIMARY KEY,
                    metadata JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW()
                );
                
                CREATE TABLE IF NOT EXISTS generations (
                    id SERIAL PRIMARY KEY,
                    run_id VARCHAR(255) REFERENCES runs(run_id),
                    record JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW()
                );
                
                CREATE INDEX IF NOT EXISTS idx_generations_run 
                ON generations(run_id);
            """)
            self.conn.commit()
    
    def save_run_metadata(self, run_id: str, metadata: dict[str, Any]) -> None:
        """Save run metadata to PostgreSQL."""
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO runs (run_id, metadata) VALUES (%s, %s)"
                "ON CONFLICT (run_id) DO UPDATE SET metadata = EXCLUDED.metadata",
                (run_id, json.dumps(metadata))
            )
            self.conn.commit()
    
    def save_generation_record(self, run_id: str, record: GenerationRecord) -> None:
        """Save generation record to PostgreSQL."""
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO generations (run_id, record) VALUES (%s, %s)",
                (run_id, json.dumps(record.__dict__))
            )
            self.conn.commit()
    
    def load_generation_records(self, run_id: str) -> list[GenerationRecord]:
        """Load generation records from PostgreSQL."""
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT record FROM generations WHERE run_id = %s ORDER BY id",
                (run_id,)
            )
            records = []
            for (record_json,) in cur.fetchall():
                data = json.loads(record_json)
                records.append(GenerationRecord(**data))
            return records
    
    # ... implement other methods
    
    def close(self):
        """Close database connection."""
        self.conn.close()
```

---

## Execution Backends

Execution backends control how tasks are executed (sequential, parallel, distributed).

### Use Cases

- **Distributed Execution**: Use Ray or Dask for multi-machine parallelism
- **GPU Batching**: Batch requests for efficient GPU utilization
- **Async Execution**: Use async/await for I/O-bound tasks
- **Custom Scheduling**: Implement custom task scheduling logic

### Interface Definition

```python
from themis.backends import ExecutionBackend
from typing import Callable, Iterable, Iterator

class MyExecutionBackend(ExecutionBackend):
    def map(
        self,
        func: Callable[[T], R],
        items: Iterable[T],
        *,
        max_workers: int | None = None,
        timeout: float | None = None,
        **kwargs,
    ) -> Iterator[R]:
        """Execute function over items in parallel.
        
        Yields results as they complete (order not guaranteed).
        """
        pass
    
    def shutdown(self) -> None:
        """Cleanup resources."""
        pass
```

### Example: Ray Execution Backend

```python
from themis.backends import ExecutionBackend
import ray

class RayExecutionBackend(ExecutionBackend):
    """Distributed execution using Ray."""
    
    def __init__(self, num_cpus: int | None = None, address: str | None = None):
        """Initialize Ray cluster.
        
        Args:
            num_cpus: Number of CPUs for local cluster (None = use all)
            address: Ray cluster address (None = start local cluster)
        """
        if not ray.is_initialized():
            if address:
                ray.init(address=address)
            else:
                ray.init(num_cpus=num_cpus)
    
    def map(self, func, items, max_workers=None, timeout=None, **kwargs):
        """Execute function using Ray remote tasks."""
        # Convert function to Ray remote
        @ray.remote
        def remote_func(item):
            return func(item)
        
        # Submit all tasks
        items_list = list(items)
        futures = [remote_func.remote(item) for item in items_list]
        
        # Get results as they complete
        remaining = futures
        while remaining:
            ready, remaining = ray.wait(remaining, timeout=timeout)
            for future in ready:
                yield ray.get(future)
    
    def shutdown(self):
        """Shutdown Ray (optional, as Ray persists across runs)."""
        # Usually you don't want to shutdown Ray between runs
        pass
```

### Example: Async Execution Backend

```python
import asyncio
from themis.backends import ExecutionBackend

class AsyncExecutionBackend(ExecutionBackend):
    """Async execution for I/O-bound tasks."""
    
    def __init__(self, max_concurrent: int = 100):
        self.max_concurrent = max_concurrent
    
    def map(self, func, items, max_workers=None, timeout=None, **kwargs):
        """Execute function using asyncio."""
        async def _run():
            semaphore = asyncio.Semaphore(max_workers or self.max_concurrent)
            
            async def _run_one(item):
                async with semaphore:
                    # Wrap sync function in executor
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, func, item)
            
            items_list = list(items)
            tasks = [_run_one(item) for item in items_list]
            
            for coro in asyncio.as_completed(tasks):
                yield await coro
        
        # Run in event loop
        loop = asyncio.get_event_loop()
        async_gen = _run()
        
        # Convert async generator to sync generator
        while True:
            try:
                yield loop.run_until_complete(async_gen.__anext__())
            except StopAsyncIteration:
                break
    
    def shutdown(self):
        """No cleanup needed."""
        pass
```

---

## Testing Your Backend

### Unit Tests

```python
import pytest
from themis.backends import StorageBackend
from themis.core.entities import GenerationRecord

def test_storage_backend(backend: StorageBackend):
    """Test storage backend implementation."""
    run_id = "test-run-123"
    
    # Test save and load
    record = GenerationRecord(
        id="sample-1",
        prompt="What is 2+2?",
        response="4",
        metadata={},
    )
    
    backend.save_generation_record(run_id, record)
    loaded = backend.load_generation_records(run_id)
    
    assert len(loaded) == 1
    assert loaded[0].id == record.id
    
    # Test list runs
    runs = backend.list_runs()
    assert run_id in runs
    
    # Test run exists
    assert backend.run_exists(run_id)
    assert not backend.run_exists("nonexistent-run")
```

### Integration Tests

```python
def test_full_evaluation_with_custom_backend():
    """Test full evaluation with custom backend."""
    from themis import evaluate
    from my_backends import S3StorageBackend
    
    backend = S3StorageBackend(bucket="my-test-bucket")
    
    result = evaluate(
        benchmark="demo",
        model="fake-math-llm",
        limit=10,
        storage_backend=backend,  # Use custom backend
    )
    
    assert result.evaluation_report.metrics['ExactMatch'].mean > 0
    
    # Verify data in S3
    assert backend.run_exists(result.run_id)
```

---

## Best Practices

1. **Thread Safety**: Ensure `save_*` methods are thread-safe if using parallel execution
2. **Error Handling**: Raise appropriate exceptions (`FileNotFoundError`, etc.)
3. **Resource Cleanup**: Implement `close()` for proper resource management
4. **Idempotency**: Make operations idempotent where possible
5. **Testing**: Write comprehensive tests for your backend
6. **Documentation**: Document any backend-specific configuration options

---

## When to Use Custom Backends

**You DON'T need custom backends if:**
- You're doing individual research experiments (default file storage is fine)
- You have < 10,000 samples per run (default threading is sufficient)
- You're working on a single machine

**You MIGHT need custom backends if:**
- **Storage**: You need team collaboration on shared results (→ S3/GCS backend)
- **Storage**: You want to query results with SQL (→ PostgreSQL backend)
- **Execution**: You have > 10,000 samples and multiple machines (→ Ray backend)
- **Execution**: You need GPU batching for efficiency (→ Custom GPU backend)

---

## Getting Help

- Check existing implementations in `themis/backends/`
- Ask questions in GitHub Discussions
- Share your backend implementations with the community!
