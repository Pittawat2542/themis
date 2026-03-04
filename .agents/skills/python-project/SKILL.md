---
name: python-project-conventions
description: Best practices and conventions for developing modern Python applications.
---

# Python Project Conventions

## Overview

Use modern Python features, strict typing, and structured configurations to build robust, scalable applications. Incorporates core rules from the **Google Python Style Guide**, adapted as a standard baseline.

**Core principle:** Strong typing, explicit dependencies, consistent naming, and composable architectures lead to maintainable code.

**Violating these conventions introduces technical debt and breaks architectural consistency.**

## When to Use

**Always:**
- Creating new Python projects
- Adding new modules or packages to existing projects
- Refactoring existing codebase layers
- Reviewing pull requests

## Code Style & Formatting

**MANDATORY Rules:**
- **Max line length:** 80 characters. Use implicit line joining within parentheses/brackets instead of backslash `\` escapes.
- **Modern Python:** Use `list[str]`, `X | None`, and modern standard library features where available.
- **Strict type hints:** Expected everywhere (`py.typed` included for libraries).
- **Docstrings:** Use the **Google Docstring Format** (`Args:`, `Returns:`, `Yields:`, `Raises:`). Summary line must be within the 80-char limit.
- **Tooling:** Default to `ruff` for linting/formatting and `mypy` for static type checking, unless project specifies alternatives like `black` or `pyright`.

<Good>
```python
from __future__ import annotations
from typing import Any

def process_data(data: dict[str, Any], limit: int | None = None) -> list[str]:
    """Process data with a strict limit.
    
    Args:
        data: The payload to process.
        limit: Optional maximum number of items.
        
    Returns:
        A list of processed string IDs.
        
    Raises:
        ValueError: If the limit is negative.
    """
    pass
```
Strict typing, explicit optional, Google style docstring, modern imports.
</Good>

<Bad>
```python
def process_data(data, limit=None):
    # Process the data
    pass
```
No types, weak docstring, missing Future imports.
</Bad>

## Naming Conventions

**MANDATORY Rules:**
- `module_name`, `package_name`
- `ClassName`, `ExceptionName` (must end with `Error`)
- `method_name`, `function_name`, `function_parameter_name`
- `instance_var_name`, `local_var_name`
- `GLOBAL_CONSTANT_NAME` 

Use a leading underscore `_internal_name` for "protected" or "private" visibility at the module, class, or instance level. Do not use private names from other modules.

## Directory Structure & Organization

**MANDATORY:**
Use a highly predictable and opinionated repository structure based on the modern `src` layout.

```text
my-project/
├── .github/workflows/   # CI/CD pipelines
├── docs/                # MkDocs/Sphinx documentation site
├── scripts/             # Maintenance, deployment, or utility scripts
├── src/                 # Source code root (prevents accidental sys.path imports)
│   └── my_pkg/          # Top-level importable package
│       ├── __init__.py
│       ├── core/        # Domain logic and entities
│       ├── api/         # Public-facing interfaces
│       ├── config/      # Configuration schemas
│       └── utils/       # Shared helpers
├── tests/               # Pytest suite
│   ├── integration/     # Slow, database/network tests
│   └── unit/            # Fast, isolated tests
├── .gitignore           # Ignored files
├── pyproject.toml       # Single-source of truth for project metadata
└── README.md            # Project entry point
```

**Rules:**
1. **Always use a `src/` layout.** This isolates code and ensures tests run against the installed package, not the local directory.
2. Put internal tooling or one-off tasks in `scripts/`, not scattered in the root.
3. Keep the root directory clean. Avoid dumping raw Python files there (except `setup.py` if legacy).

## In-File Structure & Organization

**MANDATORY:**
Every Python file should follow a predictable, top-to-bottom logical flow. Do not scatter imports or constants throughout the middle of the file.

### Standard Layout

```python
"""Module docstring describing the purpose of this file."""

from __future__ import annotations

# 1. Standard Library Imports
import os
import sys
from typing import Any

# 2. Third-Party Imports
import pydantic
import requests

# 3. First-Party (Local) Imports
from my_pkg.core import entities
from my_pkg.utils import helpers

# 4. Type Aliases & TypeVars (if any)
_PayloadDict = dict[str, Any]

# 5. Module-Level Constants
DEFAULT_TIMEOUT = 30
_INTERNAL_CACHE_SIZE = 128

# 6. Global Setup / Singletons (Avoid if possible)
_logger = logging.getLogger(__name__)

# 7. Helper / Internal Functions
def _parse_internal(data: _PayloadDict) -> str:
    pass

# 8. Public Classes
class DataProcessor:
    pass

# 9. Public Functions
def process(data: _PayloadDict) -> None:
    pass

# 10. Execution Block (Only if intended to be run as a script)
if __name__ == "__main__":
    process({"key": "value"})
```

**Rules:**
1. **Dunder Future First:** Always place `from __future__ import annotations` immediately after the module docstring.
2. **Import Sorting:** Strictly enforce the 3-block import rule (Stdlib, Third-party, First-party). Using `ruff` with the `I` (isort) rule automates this.
3. **Internal Helpers Top-Down:** Define internal helper functions (`_hidden`) *before* the public classes/functions that use them. This makes the public API easier to find at the bottom of the file (or easily discoverable via IDE collapse).

## Project Dependencies & Metadata

**MANDATORY:**
- Choose a modern dependency resolver (`uv`, `poetry`, or `pipenv`)
- Use `pyproject.toml` (PEP 621) for project metadata

**Imports:**
- Use **absolute imports** specifying the full module path. Avoid relative imports beyond one level up and never use `import *`.
- Group imports into: 1) standard library, 2) third-party, and 3) local application.

**Dependencies:**
1. Keep core `dependencies` lightweight.
2. Isolate heavy dependencies into `[project.optional-dependencies]` (e.g., `dev`, `docs`, `server`).
3. If necessary for performance, use `__getattr__` in `__init__.py` to lazy-load heavy submodules to minimize initial import time.

## Evaluation & Language Idioms

**Default Arguments:** 
Never use mutable objects (like lists or dicts) as default argument values in function definitions. Use `None` and instantiate inside the function.

**True/False Evaluation:**
Use standard implicit boolean rules for emptiness. 
- Use `if not users:` instead of `if len(users) == 0:`
- Use `if items:` instead of `if items != []:`
- Use `if flag:` instead of `if flag == True:` (and `if not flag:` for `False`).

<Good>
```python
def load_assets(items: list[str] | None = None):
    if items is None:
        items = []
        
    if not items:
        print("No items to load")
```
</Good>

<Bad>
```python
def load_assets(items: list[str] = []):
    if len(items) == 0:
        print("No items to load")
```
</Bad>

## Error Handling & Logging

**Exceptions:**
Create a robust hierarchy starting from a base `PackageError(Exception)`. Domain errors inherit from both the custom base *and* the relevant stdlib exception.
- Exception class names must end in `Error`.
- Never use a bare `except:` block unless re-raising or building a top-level isolation point.

<Good>
```python
class ProjectBaseError(Exception):
    """Base exception for all errors in the project."""

class ConfigurationError(ProjectBaseError, ValueError):
    """Invalid configuration."""
```
</Good>

**Logging:**
Centralize logging configuration. Log with `rich` or standard `logging.StreamHandler` for development aesthetics, structured JSON for production. Support a custom `TRACE` level if debugging requires verbosity.

## Configuration Management

**Ditch flat dictionaries.** Use `dataclasses` alongside `omegaconf`/`hydra-core`, `pydantic`, or similar tools to build strongly typed, composable configurations.

<Good>
```python
from dataclasses import dataclass, field

@dataclass
class RetryConfig:
    max_retries: int = 3
    delay: float = 0.5

@dataclass
class SystemConfig:
    retry: RetryConfig = field(default_factory=RetryConfig)
```
</Good>

## Testing Patterns

**MANDATORY Framework:** `pytest` paired with `pytest-cov`.

- Define shared, modular fixtures in `tests/conftest.py`.
- Yield mocked backend dependencies via fixtures instead of patching repetitively across tests.
- Maintain a top-level `test_smoke.py` for baseline coverage (verifying imports and instantiation without failing).

## Performance & Scalability

- **Concurrency:** Standardize around `concurrent.futures.ThreadPoolExecutor` for concurrent I/O.
- **Resilience:** Blend with `tenacity` or a similar robust tool for clean exception swallowing, backoff, and retries.
- **Async:** Use `async`/`await` patterns for dedicated server components (e.g., FastAPI, WebSockets).

## Verification Checklist

Before considering a Python feature complete:

- [ ] Line length <= 80 chars, absolute imports, no mutable default arguments.
- [ ] Variables/Functions are `snake_case`, Classes are `PascalCase`, Constants are `UPPER_SNAKE_CASE`.
- [ ] Passes project linter (e.g., `ruff check .`) and formatter (`ruff format .`).
- [ ] Passes static type checking (e.g., `mypy .`) with strict typing enabled.
- [ ] Smoke tests pass (`pytest tests/test_smoke.py` or equivalent).
- [ ] Core dependencies remain lightweight; heavy dependencies are optional.
- [ ] Config changes use typed structures (dataclasses or pydantic).
- [ ] Domain exceptions end in `Error` and inherit from both the project base exception and standard library equivalents.
