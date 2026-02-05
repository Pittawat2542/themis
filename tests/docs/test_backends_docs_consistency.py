from __future__ import annotations

from pathlib import Path


def test_backends_readme_uses_current_evaluate_examples():
    text = Path("themis/backends/README.md").read_text(encoding="utf-8")

    # evaluate() takes benchmark_or_dataset as the first positional argument.
    assert "benchmark=" not in text
    assert 'evaluate(\n    "demo",' in text
    assert 'evaluate(\n    "math500",' in text


def test_backends_readme_mentions_storage_backend_constraints():
    text = Path("themis/backends/README.md").read_text(encoding="utf-8")

    assert "ExperimentStorage-compatible" in text
    assert "custom `StorageBackend` integration is still evolving" in text
