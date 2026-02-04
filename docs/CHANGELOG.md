# Changelog

All notable changes to Themis will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Simple `themis.evaluate()` one-liner API for quick evaluations
- 19 built-in benchmark presets (demo, gsm8k, math500, aime24, aime25, amc23, olympiadbench, beyondaime, mmlu-pro, supergpqa, gpqa, sciq, medmcqa, med_qa, commonsense_qa, piqa, social_i_qa, coqa, gsm-symbolic)
- Comprehensive NLP metrics (BLEU, ROUGE, BERTScore, METEOR)
- Code generation metrics (Pass@K, CodeBLEU, ExecutionAccuracy)
- Statistical comparison engine with t-test, bootstrap, and permutation tests
- Win/loss matrices for multi-run comparisons
- FastAPI server with REST API and WebSocket support
- Web dashboard for viewing and comparing runs
- Pluggable backend interfaces for custom storage and execution
- Extensive documentation with Jupyter tutorials
- Interactive API documentation at `/docs` endpoint

### Changed
- Simplified CLI from 20+ commands to 7 focused commands (`demo`, `eval`, `compare`, `share`, `serve`, `list`, `clean`)
- Improved storage architecture with atomic writes and smart cache invalidation
- Refactored preset system for better extensibility
- Updated all documentation for clarity and completeness

### Fixed
- Storage V2 lifecycle bug where `start_run()` wasn't called by orchestrator
- Import errors in backend interfaces
- Statistical test edge cases (perfect consistency, zero variance)
- CLI parameter validation issues

## [2.0.0] - TBD

### Major Changes
- Complete architecture refactor for simplicity and extensibility
- Breaking changes from v1.x - see Migration Guide

### Migration from v1.x
- ExperimentBuilder replaced with `themis.evaluate()`
- Configuration files no longer needed for simple use cases
- CLI commands simplified and renamed

## [1.x] - Previous Versions

See git history for previous releases.

---

## Version Support

| Version | Supported          | Python |
|---------|--------------------|--------|
| 2.0.x   | ✅ Active          | 3.12+  |
| 1.x     | ⚠️ Maintenance     | 3.11+  |
| < 1.0   | ❌ End of Life     | 3.10+  |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to contribute to Themis.

## License

MIT License - see [LICENSE](LICENSE) for details.
