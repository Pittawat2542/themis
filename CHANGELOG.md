# Changelog

All notable changes to Themis will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-01-22

### Added
- Simple `themis.evaluate()` one-liner API for quick evaluations
- 6 built-in benchmark presets (demo, gsm8k, math500, aime24, mmlu_pro, supergpqa)
- Comprehensive NLP metrics (BLEU, ROUGE, BERTScore, METEOR)
- Code generation metrics (Pass@K, CodeBLEU, ExecutionAccuracy)
- Statistical comparison engine with t-test, bootstrap, and permutation tests
- Win/loss matrices for multi-run comparisons
- FastAPI server with REST API and WebSocket support
- Web dashboard for viewing and comparing runs
- Pluggable backend interfaces for custom storage and execution
- 31 pages of comprehensive documentation with MkDocs
- 2 interactive Jupyter notebook tutorials
- 5 runnable code examples
- Migration guide from v0.1.x
- FAQ with 50+ questions
- Best practices guide
- Complete API reference documentation
- CI/CD pipeline with GitHub Actions

### Changed
- **BREAKING**: Simplified CLI from 20+ commands to 5 essential commands (`eval`, `compare`, `serve`, `list`, `clean`)
- **BREAKING**: Replaced `ExperimentBuilder` with simple `themis.evaluate()` function
- Improved storage architecture with atomic writes and smart cache invalidation
- Refactored preset system for better extensibility
- Updated all documentation for clarity and completeness
- Updated dependencies:
  - Pydantic: 2.7 → 2.12.5 (Python 3.14 support)
  - Cyclopts: 2.9 → 4.0.0 (improved CLI)
  - LiteLLM: 1.79.0 → 1.81.1 (270+ models, 25% CPU reduction)
  - FastAPI: 0.115.0 → 0.128.0 (Pydantic v2 compatible)

### Fixed
- Storage V2 lifecycle bug where `start_run()` wasn't called by orchestrator
- Optional integration imports (wandb, huggingface_hub) made truly optional
- Statistical test edge cases (perfect consistency, zero variance)
- CLI parameter validation issues
- CLI tests updated for new command structure

### Removed
- Complex configuration file system (replaced with simple parameters)
- Old multi-command CLI structure
- Deprecated v0.1.x APIs

### Major Changes
- Complete architecture refactor for simplicity and extensibility
- Breaking changes from v1.x - see Migration Guide

### Migration from v1.x
- ExperimentBuilder replaced with `themis.evaluate()`
- Configuration files no longer needed for simple use cases
- CLI commands simplified and renamed
- See [Migration Guide](docs/MIGRATION.md) for details

## [1.x] - Previous Versions

See git history for previous releases.

---

## Version Support

| Version | Supported          | Python |
|---------|--------------------|--------|
| 0.2.x   | ✅ Active          | 3.12+  |
| 0.1.x   | ⚠️ Maintenance     | 3.12+  |
| < 0.1   | ❌ End of Life     | 3.11+  |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to contribute to Themis.

## License

MIT License - see [LICENSE](LICENSE) for details.
