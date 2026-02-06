# Contributing to Themis

Thank you for your interest in contributing to Themis! We welcome contributions from the community.

This guide covers:
- Development setup
- Code style and standards
- Testing requirements
- Pull request process
- Areas where we need help

## Getting Started

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (recommended for dependency management)

### Installation

1. **Fork the repository** on GitHub.
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/themis.git
   cd themis
   ```
3. **Install dependencies** using `uv`:
   ```bash
   uv sync --all-extras --dev
   ```
   This will create a virtual environment and install all necessary dependencies, including development tools.

## Development Workflow

### Running Tests

We use `pytest` for testing. To run the full test suite:

```bash
uv run pytest
```

To run a specific test file:

```bash
uv run pytest tests/generation/test_strategies.py
```

To run with coverage:

```bash
uv run pytest --cov=themis --cov-report=html
```

### Code Style

- **Python Version**: 3.12+
- **Formatting**: We follow PEP 8.
- **Type Hinting**: All code should be fully type-hinted and pass static analysis.
- **Docstrings**: Please include docstrings for all public modules, classes, and functions.

Run local quality checks before opening a PR:

```bash
# Formatting and linting for files you changed
uv run ruff format --check <changed_python_files>
uv run ruff check <changed_python_files>

# Baseline syntax/runtime lint used by CI
uv run ruff check --select E9,F63,F7 themis tests

# Docs build must stay strict-clean
uv run mkdocs build --strict
```

To prevent commit-time Ruff regressions, enable the repository pre-commit hook:

```bash
git config core.hooksPath .githooks
```

This hook runs `scripts/ci/check_staged_python.sh`, which validates staged Python
files with `ruff format --check` and `ruff check` before each commit.

### Project Structure

- `themis/`: Source code
- `tests/`: Test suite
- `examples-simple/`: Runnable example scripts
- `docs/`: Documentation

## Submitting Changes

1. **Create a new branch** for your feature or bugfix:
   ```bash
   git checkout -b feature/my-new-feature
   ```
2. **Make your changes** and commit them with clear, descriptive messages.
3. **Run tests** to ensure your changes don't break existing functionality.
4. **Push your branch** to your fork:
   ```bash
   git push origin feature/my-new-feature
   ```
5. **Open a Pull Request** against the `main` branch of the original repository.
   - Provide a clear title and description of your changes.
   - Link to any relevant issues.

## Versioning & Releases

- The project follows SemVer-compatible release tags (for example `vX.Y.Z` and post releases like `vX.Y.Z.postN`).
- Keep version metadata in sync across:
  - `pyproject.toml`
  - `CHANGELOG.md`
  - `docs/CHANGELOG.md`
  - `CITATION.cff`
- Validate release metadata locally:
  ```bash
  uv run python scripts/ci/validate_release.py --tag vX.Y.Z
  ```
- Generate release body from changelog (used by automation):
  ```bash
  uv run python scripts/ci/extract_release_notes.py --tag vX.Y.Z --changelog CHANGELOG.md --output /tmp/release_notes.md
  ```

## Reporting Issues

If you find a bug or have a feature request, please open an issue on GitHub. Provide as much detail as possible, including steps to reproduce the issue.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
