# Design Patterns and Anti-Patterns in Themis

This document describes common design patterns in Themis and identifies anti-patterns to avoid.

## Pattern: Metadata Propagation

### The Problem

**Anti-Pattern**: Information loss at pipeline boundaries

When data flows through a pipeline, intermediate stages may filter or reduce information, causing downstream consumers to lose access to important data.

### Example Issue

**Location**: Generation plan metadata building (`themis/generation/plan.py`)

**Problem**: Dataset fields were not being passed to metrics

```python
# Anti-pattern: Explicit filtering
def _build_metadata(template, dataset_id, row):
    metadata = {"dataset_id": dataset_id}
    
    # Only include fields explicitly listed
    for field_name in self.metadata_fields:  # ❌ Empty for custom datasets
        if field_name in row:
            metadata[field_name] = row[field_name]
    
    return metadata  # ❌ Missing most dataset fields
```

**Impact**:
- Custom metrics couldn't access dataset-specific fields (e.g., `numbers`, `target`)
- Metrics received only `{" sample_id": "..."}` instead of full context
- Broke legitimate use cases like validating structured outputs

### The Solution

**Pattern**: Preserve-by-Default, Filter-Explicitly

```python
def _build_metadata(template, dataset_id, row):
    metadata = {"dataset_id": dataset_id}
    
    # If explicit filter provided, respect it (backward compatibility)
    if self.metadata_fields:
        for field_name in self.metadata_fields:
            if field_name in row:
                metadata[field_name] = row[field_name]
    else:
        # No filter: include all fields except those used elsewhere
        for field_name, field_value in row.items():
            if field_name not in (self.dataset_id_field, self.reference_field):
                metadata[field_name] = field_value
    
    return metadata  # ✅ Complete context available
```

### Design Principles

1. **Preserve Information by Default**
   - Don't filter data unless there's an explicit reason
   - Downstream components should have access to upstream context
   
2. **Make Filtering Explicit**
   - If filtering is needed, make it opt-in and obvious
   - Document why filtering is necessary (privacy, size, performance)
   
3. **Backward Compatibility**
   - Respect existing explicit filters
   - New behavior only applies when filter is not specified

### Where This Pattern Applies

#### 1. ✅ Generation Plan → Task Metadata (FIXED)

**Location**: `themis/generation/plan.py`

**Fixed**: Now includes all dataset fields when `metadata_fields` is empty

**Behavior**:
- Explicit `metadata_fields=["field1", "field2"]` → Only those fields
- Empty `metadata_fields=[]` or `None` → All fields except id/reference

#### 2. ✅ Task Metadata → Metric Metadata (FIXED)

**Location**: `themis/evaluation/pipelines/standard_pipeline.py:236`

**Fixed**: Passes complete task metadata to metrics

```python
# Before: metadata = {"sample_id": sample_id}
# After:  metadata = {**record.task.metadata, "sample_id": sample_id}
```

**Impact**: Metrics now receive ALL task metadata fields

#### 3. ✅ Metric Aggregation Strategies (FIXED)

**Locations**:
- `themis/evaluation/strategies/attempt_aware_evaluation_strategy.py:44` - FIXED
- `themis/evaluation/strategies/judge_evaluation_strategy.py:61` - FIXED

**Was**: Only `sample_id` and aggregation-specific fields

**Now**: Preserves all original metadata + adds aggregation fields

```python
# Fixed implementation
base_metadata = group[0].metadata.copy() if group[0].metadata else {}
metadata = {
    **base_metadata,  # Preserve all original fields
    "attempts": len(group),  # Add aggregation-specific info
    "sample_id": base_metadata.get("sample_id"),
}
```

**Impact**: Aggregated scores now retain full task context

**Tests**: `tests/evaluation/test_metadata_strategies.py` - 4 tests added

#### 4. ✅ Storage/Export (VERIFIED - NO ISSUES)

**Locations**:
- `themis/experiment/export.py:535` - Uses `dict(record.task.metadata)` ✅
- `themis/experiment/export_csv.py:152` - Uses `dict(record.task.metadata)` ✅
- `themis/experiment/storage.py` - Uses dataclass serialization ✅

**Status**: GOOD - All metadata fields are preserved

**Implementation**:
```python
# Both export files use this pattern
metadata = dict(record.task.metadata)  # Preserves all fields
metadata.setdefault("model_identifier", record.task.model.identifier)
# Adds additional fields without losing existing ones
```

**Conclusion**: No changes needed. Export functions correctly preserve metadata.

#### 5. ✅ Integration Hooks (VERIFIED - NO ISSUES)

**Locations**:
- `themis/integrations/wandb.py` - Accesses specific fields but logs complete records
- `themis/integrations/huggingface.py` - Uses `asdict()` for serialization ✅

**Status**: GOOD - Full records are uploaded to integrations

**Implementation**:
```python
# HuggingFace integration (line 20-28)
def to_dict(obj):
    if is_dataclass(obj):
        return asdict(obj)  # Preserves ALL fields including metadata
    # ... handles nested structures ...

# Usage (line 61)
record_dict = to_dict(record)  # Complete record with all metadata
```

**Conclusion**: No changes needed. Integrations preserve complete metadata through serialization.

## Pattern: Registry Design

### The Pattern

**Consistent Registration API**: All extensible components follow the same pattern

```python
# Module-level registry
_REGISTRY = {}

# Registration function
def register_component(name, factory):
    """Register a custom component."""
    _REGISTRY[name] = factory

# Creation function
def create_component(name, **options):
    """Create component from registry."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown: {name}")
    return _REGISTRY[name](options)

# Query function
def list_components():
    """List registered components."""
    return list(_REGISTRY.keys())
```

### Applied To

- ✅ Metrics: `register_metric()`, `get_registered_metrics()`
- ✅ Datasets: `register_dataset()`, `list_datasets()`, `is_dataset_registered()`
- ✅ Providers: `register_provider()`, `create_provider()`
- ✅ Benchmarks: `register_benchmark()`, `list_benchmarks()`, `get_benchmark_preset()`

### Benefits

- **Consistency**: Same pattern everywhere
- **Discoverability**: Easy to find what's available
- **Extensibility**: Users can add components without modifying source
- **Type Safety**: Validation at registration time

## Pattern: Direct Usage vs Registration

### When to Register

✅ **Register when**:
- Component is referenced by name: `metrics=["my_metric"]`
- Component is reused across multiple evaluations
- Component should be discoverable via list/query APIs

### When to Use Directly

✅ **Use directly when**:
- Component is used once
- Component is specific to single evaluation
- Component doesn't need a name

### Examples

```python
# Register for reuse
themis.register_metric("word_count", WordCountMetric)
themis.evaluate(..., metrics=["word_count"])

# Use directly for one-off
class OneOffExtractor(Extractor):
    def extract(self, output, **context):
        return output.upper()

themis.evaluate(..., extractor=OneOffExtractor())
```

## Anti-Pattern: Implicit Filtering

### What Not To Do

```python
# ❌ Anti-pattern: Silently filter data
def process_record(record):
    return {
        "id": record.id,
        "output": record.output,
        # Missing: All other fields from record
    }
```

### What To Do

```python
# ✅ Pattern: Preserve by default
def process_record(record):
    result = record.to_dict()  # Start with everything
    result["processed"] = True  # Add processing flag
    return result

# ✅ Pattern: Explicit filtering
def process_record(record, fields_to_include=None):
    if fields_to_include:
        # Explicit filter
        return {k: v for k, v in record.to_dict().items() if k in fields_to_include}
    else:
        # No filter - preserve all
        return record.to_dict()
```

## Testing Strategy for Metadata Propagation

### Test Template

```python
def test_component_preserves_metadata():
    """Test that component preserves all metadata fields."""
    
    # Create input with rich metadata
    input_data = {
        "id": "test",
        "required_field": "value",
        "custom_field_1": "custom1",
        "custom_field_2": 42,
        "nested": {"key": "value"},
    }
    
    # Process through component
    output = component.process(input_data)
    
    # Verify all custom fields preserved
    assert "custom_field_1" in output.metadata
    assert output.metadata["custom_field_1"] == "custom1"
    assert "custom_field_2" in output.metadata
    assert output.metadata["custom_field_2"] == 42
    assert "nested" in output.metadata
```

### Test Checklist

For each pipeline stage:
- [ ] Input metadata with custom fields
- [ ] Process through stage
- [ ] Verify custom fields present in output
- [ ] Verify values unchanged
- [ ] Test with nested structures
- [ ] Test with various data types (str, int, list, dict)

## Summary

### Key Takeaways

1. **Preserve Information**: Don't filter unless necessary
2. **Explicit Filtering**: Make filtering opt-in and documented
3. **Backward Compatibility**: Respect existing explicit filters
4. **Test Coverage**: Test metadata propagation at every stage
5. **Consistent APIs**: Follow established patterns (registration, creation, query)

### When Adding New Components

Ask these questions:
1. Does this component receive metadata?
2. Does it pass metadata to next stage?
3. Am I filtering any fields? Why?
4. Could downstream consumers need these fields?
5. Is filtering explicit and documented?

### Code Review Checklist

When reviewing PRs, watch for:
- [ ] `metadata = {specific_field: value}` without preserving upstream
- [ ] Loops that selectively copy fields
- [ ] Processing that transforms but doesn't preserve original
- [ ] Missing tests for metadata propagation

---

## Audit Results Summary

### ✅ Completed Audit (2026-01-24)

All identified locations have been audited and fixed where necessary:

| Location | Status | Action Taken |
|----------|--------|--------------|
| `generation/plan.py` | ✅ FIXED | Smart filtering: preserve all fields when `metadata_fields` empty |
| `evaluation/pipelines/standard_pipeline.py` | ✅ FIXED | Merge task metadata: `{**record.task.metadata, "sample_id": sample_id}` |
| `evaluation/strategies/attempt_aware_evaluation_strategy.py` | ✅ FIXED | Preserve base metadata during aggregation |
| `evaluation/strategies/judge_evaluation_strategy.py` | ✅ FIXED | Preserve base metadata during aggregation |
| `experiment/export.py` | ✅ VERIFIED | Already using `dict(record.task.metadata)` - no issues |
| `experiment/export_csv.py` | ✅ VERIFIED | Already using `dict(record.task.metadata)` - no issues |
| `experiment/storage.py` | ✅ VERIFIED | Dataclass serialization preserves all fields - no issues |
| `integrations/wandb.py` | ✅ VERIFIED | Logs complete records - no issues |
| `integrations/huggingface.py` | ✅ VERIFIED | Uses `asdict()` for complete serialization - no issues |

### Test Coverage Added

- ✅ `tests/evaluation/test_metadata_simple.py` - End-to-end metadata propagation
- ✅ `tests/evaluation/test_metadata_strategies.py` - Strategy aggregation preservation (4 tests)

### Test Results

**Total**: 448 tests passing (1 skipped)

---

**Audit Completed**: 2026-01-24
**All High and Medium Priority Items**: RESOLVED
