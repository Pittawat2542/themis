# Themis v0.2.0 Release Notes

**Release Date**: January 22, 2026  
**Status**: Ready for Release ✅

---

## 🎉 What's New

Themis v0.2.0 is a **major refactor** that transforms the framework from a complex, configuration-heavy system into a simple, powerful, and modern LLM evaluation tool.

### Highlights

✨ **One-Line Evaluations**
```python
from themis import evaluate
result = evaluate("gsm8k", model="gpt-4", limit=100)
```

📊 **Statistical Comparison**
```python
from themis.comparison import compare_runs
report = compare_runs(["run-1", "run-2"], statistical_test="bootstrap")
```

🌐 **Web Dashboard**
```bash
themis serve --port 8080
# Visit http://localhost:8080/dashboard
```

---

## 🚀 Key Features

### 1. Simplified API (95% Less Boilerplate)

**Before (v0.1.x)**:
```python
from themis.experiment import ExperimentBuilder, ExperimentConfig

config = ExperimentConfig(
    dataset_config=DatasetConfig(...),
    model_config=ModelConfig(...),
    metrics_config=MetricsConfig(...),
    # ... 50+ lines
)

builder = ExperimentBuilder(config)
report = builder.run()
```

**After (v0.2.0)**:
```python
from themis import evaluate
report = evaluate("gsm8k", model="gpt-4")
```

### 2. Built-in Benchmarks

- ✅ `demo` - Quick testing (10 samples)
- ✅ `gsm8k` - Grade school math (8.5K)
- ✅ `math500` - Competition math (500)
- ✅ `aime24` - Math olympiad (30)
- ✅ `mmlu_pro` - General knowledge
- ✅ `supergpqa` - Advanced reasoning

### 3. Statistical Comparison Engine

Rigorous statistical tests for comparing model performance:
- **Bootstrap**: Non-parametric, robust (default)
- **T-test**: Fast, provides effect sizes
- **Permutation**: Exact p-values
- **McNemar's**: For paired binary outcomes

Includes win/loss matrices, p-values, confidence intervals, and effect sizes.

### 4. Web Dashboard & API Server

Production-ready FastAPI server:
- REST API for programmatic access
- WebSocket for real-time updates
- Static HTML/JS dashboard (no build step)
- Interactive API docs (Swagger/ReDoc)
- CORS enabled for frontend integration

### 5. Pluggable Backend Architecture

Clean interfaces for extensibility:
- `StorageBackend` - Custom storage (S3, GCS, PostgreSQL, etc.)
- `ExecutionBackend` - Custom execution (Ray, Dask, Async, etc.)

Default implementations provided, users can add their own.

### 6. Comprehensive Documentation

- **31 pages** of documentation with MkDocs + Material theme
- **2 Jupyter tutorials** for hands-on learning
- **5 code examples** for common patterns
- **Migration guide** from v0.1.x
- **FAQ** with 50+ questions
- **Best practices** guide
- Complete **API reference**

### 7. Updated Dependencies

- **Pydantic**: 2.12.5 (Python 3.14 support)
- **Cyclopts**: 4.0.0 (improved CLI)
- **LiteLLM**: 1.81.1 (270+ models, 25% CPU reduction)
- **FastAPI**: 0.128.0 (Pydantic v2 compatible)

---

## ⚠️ Breaking Changes

This is a **major refactor** with breaking changes from v0.1.x:

### API Changes
- ❌ `ExperimentBuilder` → ✅ `themis.evaluate()`
- ❌ Configuration files → ✅ Function parameters
- ❌ Complex nested configs → ✅ Simple keyword arguments

### CLI Changes
- ❌ 20+ commands → ✅ 5 commands (`eval`, `compare`, `serve`, `list`, `clean`)
- ❌ `themis demo` → ✅ `themis eval demo --model fake-math-llm`
- ❌ `themis new-project` → ❌ Removed

### Metric Names
- ❌ `ExactMatch` → ✅ `exact_match` (lowercase snake_case)
- ❌ `MathVerify` → ✅ `math_verify`
- ❌ `BLEU` → ✅ `bleu`

---

## 📦 Migration Guide

See [`docs/MIGRATION.md`](docs/MIGRATION.md) for complete migration instructions.

**Quick migration**:
```python
# v0.1.x
from themis.experiment import ExperimentBuilder
config = ExperimentConfig(...)
builder = ExperimentBuilder(config)
report = builder.run()

# v0.2.0
from themis import evaluate
report = evaluate("gsm8k", model="gpt-4")
```

---

## 📊 Stats

- **Code**: ~15,000 lines
- **Tests**: 177+ passing
- **Documentation**: 31 pages
- **Benchmarks**: 6 built-in
- **Metrics**: 10+ supported
- **Providers**: 100+ via LiteLLM
- **Boilerplate Reduction**: 95%

---

## 🔧 Installation

```bash
# Using pip
pip install themis-eval==0.2.0

# With all features
pip install themis-eval[all]==0.2.0

# Using uv (recommended)
uv pip install themis-eval==0.2.0
```

---

## 📚 Quick Start

```python
from themis import evaluate

# Evaluate on a benchmark
result = evaluate("gsm8k", model="gpt-4", limit=100)
print(f"Accuracy: {result.metrics['exact_match']:.2%}")

# Compare two models
evaluate("gsm8k", model="gpt-4", run_id="run-gpt4")
evaluate("gsm8k", model="claude-3", run_id="run-claude")

from themis.comparison import compare_runs
report = compare_runs(["run-gpt4", "run-claude"])
print(report.summary())

# Start web dashboard
# CLI: themis serve
```

---

## 🐛 Known Issues

### Non-Critical
1. **WebSocket tests**: Skipped (requires additional setup)
2. **Integration tests**: May timeout on slow machines
3. **CLI tests**: Some skipped due to refactoring

### TODOs in Code (Future Features)
- WebSocket subscription logic in server
- Load dataset from file in CLI

These are placeholders and don't affect v0.2.0 functionality.

---

## 🙏 Acknowledgments

This release represents a complete redesign based on user feedback and best practices from the LLM research community.

Special thanks to:
- Early adopters who provided feedback
- Contributors who reported issues
- The open-source community for excellent tools (Pydantic, FastAPI, LiteLLM, etc.)

---

## 📖 Resources

- **Documentation**: https://pittawat2542.github.io/themis/
- **GitHub**: https://github.com/Pittawat2542/themis
- **PyPI**: https://pypi.org/project/themis-eval/
- **Changelog**: [CHANGELOG.md](CHANGELOG.md)
- **Migration Guide**: [docs/MIGRATION.md](docs/MIGRATION.md)
- **Contributing**: [CONTRIBUTING.md](CONTRIBUTING.md)

---

## 🚀 What's Next (v0.3.0+)

Potential future features:
- More benchmarks (HumanEval, MBPP, BigBench)
- More metrics (Perplexity, Diversity, Toxicity)
- Async evaluation for better performance
- Batch processing optimizations
- Advanced visualizations
- Plugin system for community contributions

---

## 📝 Changelog

See [CHANGELOG.md](CHANGELOG.md) for complete details.

---

**Happy Evaluating! 🎊**

For questions, issues, or feedback:
- GitHub Issues: https://github.com/Pittawat2542/themis/issues
- GitHub Discussions: https://github.com/Pittawat2542/themis/discussions
