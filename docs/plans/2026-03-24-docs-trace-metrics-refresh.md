# Documentation and Agent Skill Refresh Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Update Themis documentation, runnable examples, and local agent skill references so they describe metric-ref shorthand inputs, trace-score persistence, result-analysis surfaces, and the current projection-backed architecture.

**Architecture:** Keep user-facing API guidance in the docs and examples, and move architecture detail into the concept pages and `themis-eval` skill references. Reuse the existing example catalog by extending the agent example to exercise persisted trace metrics instead of creating a parallel example path.

**Tech Stack:** MkDocs Material, Python examples, local skill markdown, Ruff, MyPy, Pytest, uv

---

### Task 1: Plan the refresh scope

**Files:**
- Modify: `docs/quick-start/index.md`
- Modify: `docs/concepts/architecture.md`
- Modify: `docs/concepts/storage-and-resume.md`
- Modify: `docs/guides/plugins.md`
- Modify: `docs/guides/analyze-results.md`
- Modify: `docs/guides/compare-and-export.md`
- Modify: `docs/guides/examples.md`
- Modify: `docs/tutorials/hello-world.md`
- Modify: `.agents/skills/themis-eval/SKILL.md`
- Modify: `.agents/skills/themis-eval/references/getting-started.md`
- Modify: `.agents/skills/themis-eval/references/plugins-and-specs.md`
- Modify: `.agents/skills/themis-eval/references/results-and-ops.md`
- Modify: `examples/10_agent_eval.py`

**Step 1: Inspect current docs, examples, and skills**

Run: `rg -n "MetricRefSpec|trace_scores|aggregate_trace|aggregate_corpus|iter_trace_scores" docs examples .agents/skills/themis-eval themis`
Expected: Existing coverage appears in the plugin guide and runtime APIs, with gaps around shorthand inputs and projection architecture.

**Step 2: Write the refresh plan into touched surfaces**

Document:
- metric strings vs `MetricRefSpec(...)`
- `trace_scores` and trace metric config examples
- `aggregate(...)` vs `aggregate_trace(...)` vs `aggregate_corpus(...)`
- projection-backed storage and resume implications

**Step 3: Commit**

```bash
git add docs/plans/2026-03-24-docs-trace-metrics-refresh.md
git commit -m "docs(plans): add trace metrics refresh plan"
```

### Task 2: Update docs for the new metric and analysis surfaces

**Files:**
- Modify: `docs/quick-start/index.md`
- Modify: `docs/concepts/architecture.md`
- Modify: `docs/concepts/storage-and-resume.md`
- Modify: `docs/guides/plugins.md`
- Modify: `docs/guides/analyze-results.md`
- Modify: `docs/guides/compare-and-export.md`
- Modify: `docs/guides/examples.md`
- Modify: `docs/tutorials/hello-world.md`

**Step 1: Patch docs with the new narrative**

Add:
- shorthand metric refs for `ScoreSpec.metrics` and `TraceScoreSpec.metrics`
- guidance on when to use configured `MetricRefSpec`
- trace-score persistence and result inspection
- architecture notes on projections, overlays, and trace rows

**Step 2: Run focused docs tests**

Run: `uv run pytest tests/docs/test_docs_consistency.py tests/docs/test_public_docstrings.py tests/docs/test_documented_workflows.py tests/docs/test_example_display_paths.py -q`
Expected: PASS

**Step 3: Commit**

```bash
git add docs/quick-start/index.md docs/concepts/architecture.md docs/concepts/storage-and-resume.md docs/guides/plugins.md docs/guides/analyze-results.md docs/guides/compare-and-export.md docs/guides/examples.md docs/tutorials/hello-world.md
git commit -m "docs: refresh metrics and result analysis guides"
```

### Task 3: Extend the agent example for persisted trace metrics

**Files:**
- Modify: `examples/10_agent_eval.py`

**Step 1: Add trace-score coverage to the example**

Implement:
- catalog trace metric registration
- `SliceSpec.trace_scores`
- `result.aggregate_trace(...)` or `result.iter_trace_scores(...)` output

**Step 2: Run the example and its focused tests**

Run: `uv run python examples/10_agent_eval.py`
Expected: Prints candidate score aggregates and trace score aggregates/events.

Run: `uv run pytest tests/docs/test_documented_workflows.py tests/orchestration/test_orchestrator.py tests/benchmark/test_runtime.py -q`
Expected: PASS

**Step 3: Commit**

```bash
git add examples/10_agent_eval.py
git commit -m "docs(examples): show persisted trace scoring"
```

### Task 4: Update the themis-eval skill references

**Files:**
- Modify: `.agents/skills/themis-eval/SKILL.md`
- Modify: `.agents/skills/themis-eval/references/getting-started.md`
- Modify: `.agents/skills/themis-eval/references/plugins-and-specs.md`
- Modify: `.agents/skills/themis-eval/references/results-and-ops.md`

**Step 1: Patch the skill content**

Add:
- trace metrics and `TraceScoreSpec`
- metric-ref shorthand guidance
- result-surface selection guidance
- projection-backed architecture and quickcheck positioning

**Step 2: Validate docs build and skill markdown indirectly**

Run: `uv run mkdocs build --strict`
Expected: PASS

**Step 3: Commit**

```bash
git add .agents/skills/themis-eval/SKILL.md .agents/skills/themis-eval/references/getting-started.md .agents/skills/themis-eval/references/plugins-and-specs.md .agents/skills/themis-eval/references/results-and-ops.md
git commit -m "docs(skills): sync themis-eval guidance with trace metrics"
```

### Task 5: Run the full local quality pass and finalize

**Files:**
- Modify: `uv.lock` only if a tool changes it accidentally; restore before commit

**Step 1: Run repo checks**

Run:
- `uv run ruff format --check docs examples .agents/skills themis tests`
- `uv run ruff check docs examples .agents/skills themis tests`
- `uv run mypy themis tests`
- `uv run pytest -q`
- `uv run python scripts/ci/run_examples.py`
- `uv run mkdocs build --strict`
- `uv build`
- `uv run python scripts/ci/check_built_package.py`

Expected: PASS, or fix regressions before proceeding.

**Step 2: Restore incidental lockfile churn**

Run: `git restore --worktree uv.lock`
Expected: `uv.lock` no longer appears in `git status` unless intentionally changed.

**Step 3: Final commit**

```bash
git add docs examples .agents/skills tests themis
git commit -m "docs: update guides for trace metrics and projections"
```
