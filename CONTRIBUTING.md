# Contributing to Themis

Themis uses the in-repo documentation system as part of the product surface. Contributor workflows should treat docs, examples, and public-surface reference coverage as release-critical.

Start here:

- Product docs overview: [docs/project/index.md](/Users/pittawat/projects/themis/docs/project/index.md)
- Docs architecture: [docs/project/docs-architecture.md](/Users/pittawat/projects/themis/docs/project/docs-architecture.md)
- Writing guide: [docs/project/writing-guide.md](/Users/pittawat/projects/themis/docs/project/writing-guide.md)
- Example authoring: [docs/project/example-authoring.md](/Users/pittawat/projects/themis/docs/project/example-authoring.md)
- Coverage expectations: [docs/project/adding-new-docs-coverage.md](/Users/pittawat/projects/themis/docs/project/adding-new-docs-coverage.md)
- Release/versioning policy: [docs/project/release-and-versioning.md](/Users/pittawat/projects/themis/docs/project/release-and-versioning.md)

Minimum local checks:

```bash
uv run pytest tests/test_docs_site.py tests/test_docs_examples.py tests/test_docs_inventory.py -q
uv run mkdocs build --strict
```
