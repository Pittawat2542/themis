---
name: python-documentation
description: Best practices and conventions for structuring and writing Python project documentation.
---

# Python Project Documentation

## Overview

Documentation is a first-class citizen in software development. A standard convention is to rely on **MkDocs** (with the Material theme) for the site structure and **mkdocstrings** or **Sphinx** to auto-generate API references directly from Python source code.

**Core principle:** Documentation must be comprehensive, easy to navigate, and natively integrated with the code (single source of truth for APIs).

**Violating these conventions leads to fragmented, outdated, or hard-to-read documentation.**

## When to Use

**Always:**
- Adding new user-facing features or modules
- Writing or refactoring public APIs
- Creating tutorials or conceptual guides
- Reviewing pull requests for documentation completeness

## Site Structure & Configuration

**MANDATORY (if using MkDocs):**
- Use `mkdocs.yml` at the project root.
- Navigation (`nav:`) must be logically grouped into:
  - **Home / Introduction**
  - **Getting Started** (Installation, Quick Start, Core Concepts)
  - **Guides** (Task-oriented walk-throughs and tutorials)
  - **Reference** (API references, CLI references)
  - **Architecture** (Philosophy, Design principles)
  - **Project** (Changelog, Contributing guidelines)

<Good>
```yaml
nav:
  - Getting Started:
      - Installation: getting-started/installation.md
      - Quick Start: getting-started/quickstart.md
  - Reference:
      - Python API:
          - MyModule: api/my_module.md
```
Clear separation of task-oriented guides (Getting Started) and information-oriented reference (API).
</Good>

## API Documentation

**MANDATORY:**
- Auto-generate API documentation from code using a tool like the `mkdocstrings` plugin with the `python` handler or `Sphinx autodoc`.
- Docstrings in the code **must** strictly follow the **Google Style** format (`Args:`, `Returns:`, `Raises:`, `Yields:`). Alternative standard is **NumPy**, but ensure consistency.
- API reference files should primarily consist of the directive to auto-populate from source, wrapped with helpful surrounding context.

<Good>
```markdown
# MyModule API

Primary entry point for the module functionality.

::: project_name.api.my_module
    options:
      show_root_heading: false
      show_source: true
```
Surrounds auto-generated docs with a human-readable title and introduction.
</Good>

## Markdown Extensions & Formatting

Use markdown extensions to enhance readability.

- **Fenced Code Blocks:** Always specify the language (`python`, `bash`, `yaml`, etc.).
- **Tabs:** Use tabbed interfaces for mutually exclusive choices (e.g., pip vs. uv installation routines).
- **Admonitions:** Use standard admonitions (`note`, `tip`, `warning`, `danger`) to highlight important context.
- **Diagrams:** Use Mermaid blocks or equivalent for architecture and system flow diagrams.

<Good>
```markdown
!!! warning "Breaking Change"
    The execution backend API changed in version 2.0.0.
```
Uses MkDocs native admonition syntax to clearly flag a warning.
</Good>

## Documentation Content Guidelines

- **Tutorials:** Must be goal-oriented. Show the user exactly what they will build or accomplish, provide the code, and explain the "why".
- **How-to Guides:** Must be problem-oriented. Answer a specific question ("How to configure logging").
- **Explanations:** Concept-oriented. Clarify architecture, design decisions, and background context.
- **Reference:** Information-oriented. API specs, CLI flags, configuration fields. Keep it dry and accurate.

## Verification Checklist

Before considering documentation complete:

- [ ] Does the documentation site build strictly (e.g., `mkdocs build --strict`) without warnings or broken links?
- [ ] Are new Python public functions/classes documented with project-conventions docstrings?
- [ ] Is the newly introduced concept added to the navigation index under the appropriate category?
- [ ] Are code examples in markdown accurate and syntactically correct?
- [ ] If changing architecture, have relevant diagrams and architectural documents been updated?
