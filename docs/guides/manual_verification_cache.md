# Evaluation Cache Invalidation

> Technical documentation for the evaluation cache invalidation system

## Problem Statement

The resumability and skip management system had a critical flaw where adding new metrics or changing extractors would NOT trigger re-evaluation. The evaluation cache was keyed only by generation task parameters (model, temperature, prompt), ignoring evaluation configuration.

### Failing Scenarios (Before Fix)

1. **Adding new metrics**: Would reuse old evaluations, never computing the new metric
2. **Changing extractors**: Would reuse old evaluations with incorrect extraction
3. **Modifying evaluation strategies**: Would use stale evaluation results

### Working Scenarios (Already Worked)

- Changing temperature/sampling: Would regenerate responses (task key included these)
- Changing prompts: Would regenerate responses (task key included prompt hash)

## Solution Implemented

Created `evaluation_cache_key()` function that combines:
1. **Task cache key** (generation parameters: model, temperature, prompt, etc.)
2. **Evaluation config hash** (metrics, extractor type, evaluation settings)

This ensures evaluation cache invalidates when evaluation configuration changes, while still allowing generation cache reuse.

## Files Modified

### 1. `themis/experiment/storage.py`

**Added `evaluation_cache_key()` function** (lines 1386-1422):
- Combines task key with evaluation config hash
- Falls back to task key when no config (backward compatibility)
- Format: `{task_key}::eval:{config_hash}`

**Updated `append_evaluation()`** (line 738):
- Added `evaluation_config: dict | None = None` parameter
- Uses `evaluation_cache_key()` instead of task key only

**Updated `load_cached_evaluations()`** (line 769):
- Added `evaluation_config: dict | None = None` parameter
- Documentation explains cache matching behavior

**Updated exports** (line 1435):
- Added `evaluation_cache_key` to `__all__`

### 2. `themis/experiment/cache_manager.py`

**Updated `load_cached_evaluations()`** (line 68):
- Added `evaluation_config: dict | None = None` parameter
- Passes config through to storage layer

**Updated `save_evaluation_record()`** (line 97):
- Added `evaluation_config: dict | None = None` parameter
- Passes config through to storage layer

### 3. `themis/experiment/orchestrator.py`

**Added `_build_evaluation_config()` method** (line 284):
- Extracts evaluation configuration from pipeline
- Includes metric names/types and extractor configuration
- Returns dict for cache key generation

**Updated `run()` method**:
- Builds evaluation config before loading cache
- Passes config to `load_cached_evaluations()`
- Uses `evaluation_cache_key()` to check cached evaluations
- Passes config to `save_evaluation_record()`

### 4. `tests/experiment/test_storage.py`

**Enhanced `test_experiment_storage_roundtrip()`** (line 60):
- Tests evaluation with and without config
- Verifies `evaluation_cache_key()` works correctly

**Added `test_evaluation_cache_invalidation_on_config_change()`** (line 92):
- Comprehensive test for cache invalidation
- Tests config changes (adding metrics, changing extractors)
- Verifies cache keys differ when config changes
- Verifies both old and new evaluations can coexist

### 5. `tests/experiment/test_evaluation_cache_key.py` (NEW)

Created comprehensive unit test suite:
- `test_evaluation_cache_key_without_config`: No config returns task key
- `test_evaluation_cache_key_with_config`: With config includes eval marker
- `test_evaluation_cache_key_invalidates_on_metric_change`: Metric changes invalidate
- `test_evaluation_cache_key_invalidates_on_extractor_change`: Extractor changes invalidate
- `test_evaluation_cache_key_stable_for_same_config`: Same config produces same key
- `test_evaluation_cache_key_handles_unsorted_config`: Config order doesn't matter
- `test_task_cache_key_includes_sampling`: Temperature changes affect task key
- `test_task_cache_key_format`: Validates task key format
- `test_evaluation_cache_key_format`: Validates eval key format
- `test_evaluation_cache_key_with_empty_config`: Empty config behaves like None

### 6. `docs/STORAGE.md`

**Added comprehensive "Cache Invalidation" section** (line 582):
- Explains evaluation cache key concept
- Shows cache key formats
- Behavior table for different changes
- Example usage with code snippets
- Scenario walkthroughs (adding metrics, changing extractors)
- Technical details from orchestrator
- Backward compatibility notes

### 7. `README.md`

**Updated feature highlights** (line 16):
- Added note about smart cache invalidation
- Links cache invalidation to both generation and evaluation

## Behavior Changes

### What Now Works (Previously Broken)

| Change | Generation | Evaluation | Result |
|--------|-----------|------------|--------|
| Add/remove metric | ✓ Reuses cache | ✓ Re-evaluates | **FIXED** |
| Change extractor | ✓ Reuses cache | ✓ Re-evaluates | **FIXED** |
| Change extractor field | ✓ Reuses cache | ✓ Re-evaluates | **FIXED** |

### What Still Works (Already Worked)

| Change | Generation | Evaluation | Result |
|--------|-----------|------------|--------|
| Change temperature | ✓ Regenerates | ✓ Re-evaluates | Still works |
| Change prompt | ✓ Regenerates | ✓ Re-evaluates | Still works |
| Change model | ✓ Regenerates | ✓ Re-evaluates | Still works |
| No changes | ✓ Reuses cache | ✓ Reuses cache | Still works |

## Test Results

All tests pass:

```bash
$ uv run python -m pytest tests/experiment/test_evaluation_cache_key.py -v
============================== 10 passed in 1.08s ==============================
```

Test coverage:
- ✓ Basic functionality (with/without config)
- ✓ Cache invalidation on metric changes
- ✓ Cache invalidation on extractor changes
- ✓ Stability for same config
- ✓ Deterministic handling (config order doesn't matter)
- ✓ Task key includes sampling parameters
- ✓ Key format validation
- ✓ Edge cases (empty config, None config)

## Usage Examples

### Before (Broken Behavior)

```python
# Run 1: exact_match only
pipeline = EvaluationPipeline(
    extractor=JsonFieldExtractor("answer"),
    metrics=[ExactMatch()]
)
orchestrator.run(dataset, run_id="exp-1")

# Run 2: Add f1_score
pipeline = EvaluationPipeline(
    extractor=JsonFieldExtractor("answer"),
    metrics=[ExactMatch(), F1Score()]  # NEW METRIC
)
orchestrator.run(dataset, run_id="exp-1", resume=True)
# ❌ Would reuse old evaluations, f1_score never computed!
```

### After (Fixed Behavior)

```python
# Run 1: exact_match only
pipeline = EvaluationPipeline(
    extractor=JsonFieldExtractor("answer"),
    metrics=[ExactMatch()]
)
orchestrator.run(dataset, run_id="exp-1")

# Run 2: Add f1_score
pipeline = EvaluationPipeline(
    extractor=JsonFieldExtractor("answer"),
    metrics=[ExactMatch(), F1Score()]  # NEW METRIC
)
orchestrator.run(dataset, run_id="exp-1", resume=True)
# ✓ Reuses cached generations (saves cost)
# ✓ Re-evaluates with both metrics (correct results)
```

## Cache Key Examples

### Task Cache Key (Generation)

```
test-1::math::gpt-4::0.700-1.000-100::02409cd1139f

Components:
- dataset_id: test-1
- template: math
- model: gpt-4
- sampling: 0.700-1.000-100 (temp-top_p-max_tokens)
- prompt_hash: 02409cd1139f (first 12 chars of SHA256)
```

### Evaluation Cache Key

```
test-1::math::gpt-4::0.700-1.000-100::02409cd1139f::eval:41350460b0e7

Components:
- task_key: test-1::math::gpt-4::0.700-1.000-100::02409cd1139f
- eval_marker: ::eval:
- config_hash: 41350460b0e7 (first 12 chars of config SHA256)
```

## Backward Compatibility

Code without evaluation config still works:

```python
# Old style (no config)
storage.append_evaluation("run-1", record, evaluation)
# Uses task_cache_key only (fallback behavior)

# New style (with config)
config = {"metrics": ["exact_match"], "extractor": "json"}
storage.append_evaluation("run-1", record, evaluation, evaluation_config=config)
# Uses evaluation_cache_key with config hash
```

The system automatically handles both cases, providing smooth migration path.

## Implementation Quality

- ✅ No breaking changes (backward compatible)
- ✅ Clean separation of concerns
- ✅ Comprehensive test coverage (10 unit tests + 2 integration tests)
- ✅ Zero linter errors
- ✅ Proper documentation (in-code, tests, docs/STORAGE.md)
- ✅ Syntax verified on all modified files
- ✅ Follows existing code patterns and style

## Documentation

### Code Documentation
- Function docstrings with examples
- Inline comments explaining key logic
- Type hints for all parameters

### User Documentation
- `docs/STORAGE.md`: Comprehensive cache invalidation section
- `README.md`: Updated feature highlights
- Test files serve as usage examples

### Developer Documentation
- Test names explain behavior
- Test comments explain edge cases
- This implementation summary

## Files Cleaned Up

Removed temporary/draft files:
- ✓ Deleted `EVALUATION_CACHE_FIX.md` (draft documentation)
- ✓ Deleted `CHANGES_SUMMARY.md` (draft summary)
- ✓ Deleted `example_evaluation_cache.py` (standalone demo)

Content integrated into:
- `docs/STORAGE.md` (permanent documentation)
- `tests/experiment/test_evaluation_cache_key.py` (unit tests)
- `tests/experiment/test_storage.py` (integration tests)

## Summary

The implementation successfully fixes the evaluation cache invalidation issue while maintaining backward compatibility. The system now correctly handles:

1. ✅ Adding/removing metrics → Re-evaluates
2. ✅ Changing extractors → Re-evaluates  
3. ✅ Changing extraction fields → Re-evaluates
4. ✅ Changing temperature → Regenerates + re-evaluates
5. ✅ No changes → Full resume

All changes are well-tested, documented, and production-ready.
