# Metadata Propagation: Analysis and General Solution

## The Problem

**Issue**: Task metadata is lost during the evaluation pipeline, preventing custom metrics from accessing dataset-specific fields.

**Current Behavior** (line 236 in `standard_pipeline.py`):
```python
metadata = {"sample_id": sample_id}
score = metric.compute(
    prediction=prediction,
    references=references,
    metadata=metadata  # ❌ Only contains sample_id
)
```

**Expected Behavior**:
```python
metadata = {**record.task.metadata, "sample_id": sample_id}
score = metric.compute(
    prediction=prediction,
    references=references,
    metadata=metadata  # ✅ Contains full task metadata + sample_id
)
```

## Root Cause Analysis

### Data Flow

```
Dataset Sample (with fields: numbers, target, etc.)
    ↓
GenerationTask.metadata (all fields preserved)
    ↓
GenerationRecord.task.metadata (all fields available)
    ↓
Evaluation Pipeline
    ↓
metric.compute(metadata=???)  # ❌ Only sample_id passed
```

### The Pattern: **Information Loss at Pipeline Boundaries**

This is a **data transformation anti-pattern** where:
1. Information exists upstream (task.metadata)
2. A pipeline stage explicitly filters/reduces it (line 236)
3. Downstream consumers (metrics) lose access to critical data

## General Solution: **Preserve-by-Default, Filter-on-Demand**

### Principle

**When passing data between pipeline stages, preserve all information by default unless there's an explicit reason to filter.**

### Implementation Strategy

#### 1. **Metadata Merging Pattern**

```python
# ✅ Good: Preserve all metadata, add pipeline-specific fields
metadata = {
    **upstream_metadata,  # Preserve everything
    "pipeline_field": pipeline_value,  # Add new fields
}

# ❌ Bad: Select only specific fields
metadata = {"pipeline_field": pipeline_value}  # Loses upstream data
```

#### 2. **Explicit Filtering When Needed**

If filtering is necessary (e.g., for privacy, size limits), make it explicit:

```python
# Define what to exclude (whitelist approach)
METADATA_EXCLUDE = {"sensitive_field", "large_blob"}

metadata = {
    k: v for k, v in record.task.metadata.items()
    if k not in METADATA_EXCLUDE
}
metadata["sample_id"] = sample_id
```

## Where This Pattern Occurs in Themis

### 1. ✅ **Evaluation Pipeline** (CONFIRMED - needs fix)

**Location**: `themis/evaluation/pipelines/standard_pipeline.py:236`

**Issue**: Only `sample_id` passed to metrics

**Fix**:
```python
# Current (line 236)
metadata = {"sample_id": sample_id}

# Fixed
metadata = {**record.task.metadata, "sample_id": sample_id}
```

### 2. ⚠️ **Evaluation Strategies** (POTENTIAL ISSUE)

**Locations**:
- `themis/evaluation/strategies/attempt_aware_evaluation_strategy.py:46`
- `themis/evaluation/strategies/judge_evaluation_strategy.py:61`

**Pattern**:
```python
metadata={
    "attempts": len(group),
    "sample_id": group[0].metadata.get("sample_id"),
}
```

**Risk**: If aggregation strategies need access to original task metadata, they won't have it.

**Recommended Fix**:
```python
# Preserve original metadata during aggregation
base_metadata = group[0].metadata.copy()  # Get original metadata
metadata = {
    **base_metadata,  # Preserve all
    "attempts": len(group),  # Add aggregation-specific fields
    "sample_id": base_metadata.get("sample_id"),
}
```

### 3. ⚠️ **Storage/Export** (CHECK NEEDED)

**Locations**: 
- `themis/experiment/export.py`
- `themis/experiment/export_csv.py`
- `themis/experiment/storage.py`

**Potential Issue**: When exporting results, task metadata might be filtered

**Check**: Do exported results include full task metadata?

### 4. ⚠️ **Integration Hooks** (CHECK NEEDED)

**Locations**:
- `themis/integrations/wandb.py`
- `themis/integrations/huggingface.py`

**Potential Issue**: When logging to external systems, metadata might be filtered

**Check**: Are all relevant fields logged, or just a subset?

### 5. ⚠️ **Extractors** (LOW RISK)

**Extractors receive `**context`** which includes task data, but check if context is properly passed.

## Systematic Fix Plan

### Phase 1: Critical Fix (Immediate)
- [x] Fix evaluation pipeline metadata passing (line 236)
- [x] Add test for metadata propagation to metrics
- [x] Document in CHANGELOG

### Phase 2: Comprehensive Audit
- [ ] Audit all evaluation strategies for metadata handling
- [ ] Audit storage/export for metadata preservation
- [ ] Audit integration hooks for metadata logging
- [ ] Create test suite for metadata propagation across all stages

### Phase 3: Design Pattern Enforcement
- [ ] Add linting rule to catch `metadata = {` patterns (flag for review)
- [ ] Document metadata propagation best practices
- [ ] Add metadata propagation tests to CI

## Testing Strategy

### 1. Unit Test for Metadata Propagation

```python
def test_metric_receives_full_metadata():
    """Ensure metrics receive complete task metadata."""
    
    @dataclass
    class TestMetric(Metric):
        def __post_init__(self):
            self.name = "test_metadata"
        
        def compute(self, *, prediction, references=None, metadata=None):
            # Check that metadata contains task-specific fields
            assert "numbers" in metadata
            assert "target" in metadata
            assert "sample_id" in metadata
            return MetricScore(
                metric_name=self.name,
                value=1.0,
                details={},
                metadata=metadata,
            )
    
    themis.register_metric("test_metadata", TestMetric)
    
    dataset = [
        {
            "id": "1",
            "question": "Test",
            "answer": "42",
            "numbers": [1, 2, 3],  # Task-specific field
            "target": 6,            # Task-specific field
        }
    ]
    
    report = themis.evaluate(
        dataset,
        model="fake",
        prompt="Q: {question}\nA:",
        metrics=["test_metadata"],
    )
    
    # If this passes, metadata propagation works
    assert report.evaluation_report.metrics["test_metadata"].mean == 1.0
```

### 2. Integration Test Across Pipeline

```python
def test_metadata_preserved_through_pipeline():
    """Ensure metadata flows through entire pipeline."""
    
    # Create dataset with rich metadata
    dataset = [
        {
            "id": "sample1",
            "question": "Q",
            "answer": "A",
            "custom_field_1": "value1",
            "custom_field_2": 42,
            "nested": {"key": "value"},
        }
    ]
    
    # Track metadata at each stage
    metadata_log = []
    
    @dataclass
    class MetadataTracker(Metric):
        def __post_init__(self):
            self.name = "tracker"
        
        def compute(self, *, prediction, references=None, metadata=None):
            metadata_log.append(metadata.copy())
            return MetricScore(
                metric_name=self.name,
                value=1.0,
                details={},
                metadata=metadata,
            )
    
    themis.register_metric("tracker", MetadataTracker)
    
    report = themis.evaluate(
        dataset,
        model="fake",
        prompt="Q: {question}\nA:",
        metrics=["tracker"],
    )
    
    # Verify all custom fields present
    assert len(metadata_log) == 1
    received = metadata_log[0]
    assert "custom_field_1" in received
    assert received["custom_field_1"] == "value1"
    assert "custom_field_2" in received
    assert received["custom_field_2"] == 42
    assert "nested" in received
    assert received["nested"]["key"] == "value"
```

## Best Practices for Future Development

### 1. **Metadata Contract**

Document what metadata each component expects and provides:

```python
class Component:
    """
    Metadata Contract:
      Expects: sample_id, dataset_id (optional)
      Provides: sample_id, component_specific_field
      Preserves: All input metadata + added fields
    """
```

### 2. **Defensive Metadata Handling**

```python
# ✅ Good: Safe metadata merging
def process(record):
    base_metadata = getattr(record.task, 'metadata', {})
    metadata = {
        **base_metadata,  # Preserve all
        "processor_id": "my_processor",
    }
    return metadata

# ❌ Bad: Assumes specific fields exist
def process(record):
    return {
        "sample_id": record.task.metadata["sample_id"],  # Breaks if missing
    }
```

### 3. **Metadata Schema Validation**

For critical fields, validate but don't filter:

```python
# ✅ Good: Validate and preserve
def validate_and_preserve(metadata):
    # Validate required fields
    if "sample_id" not in metadata:
        raise ValueError("sample_id required")
    
    # Return all metadata
    return metadata

# ❌ Bad: Validate and filter
def validate_and_filter(metadata):
    return {"sample_id": metadata["sample_id"]}  # Loses other fields
```

## Summary

### The Core Issue
**Information loss at pipeline boundaries** due to explicit filtering instead of preservation.

### The Solution
**Preserve-by-default**: Merge upstream metadata with pipeline-specific fields.

### The Pattern to Watch
Anytime you see:
```python
metadata = {"field": value}
```

Ask: **Should this preserve upstream metadata?**

Change to:
```python
metadata = {**upstream_metadata, "field": value}
```

### Impact
- ✅ Custom metrics can access all dataset fields
- ✅ Strategies can access task metadata during aggregation
- ✅ Export/logging includes complete context
- ✅ Debugging easier with full metadata trail
- ✅ More flexible and extensible system

### Priority Locations to Fix
1. **HIGH**: `standard_pipeline.py:236` - Breaks custom metrics
2. **MEDIUM**: Evaluation strategies - May break aggregation
3. **LOW**: Export/integration - Completeness issue, not breaking
