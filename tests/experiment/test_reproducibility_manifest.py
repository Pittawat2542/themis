"""Tests for reproducibility manifest generation and validation."""

from __future__ import annotations

import pytest

from themis.experiment.manifest import (
    build_reproducibility_manifest,
    manifest_hash,
    validate_reproducibility_manifest,
)


def test_manifest_has_required_fields():
    manifest = build_reproducibility_manifest(
        model="fake-math-llm",
        provider="fake",
        provider_options={"api_key": "test"},
        sampling={"temperature": 0.0, "top_p": 0.95, "max_tokens": 128},
        num_samples=2,
        evaluation_config={
            "metrics": ["themis.evaluation.metrics.exact_match.ExactMatch:ExactMatch"],
            "extractor": "themis.evaluation.extractors.IdentityExtractor",
        },
        seeds={"sampling_seed": 7},
    )

    assert manifest["schema_version"] == "1"
    assert manifest["model"]["identifier"] == "fake-math-llm"
    assert manifest["model"]["provider"] == "fake"
    assert "package_versions" in manifest
    assert "themis-eval" in manifest["package_versions"]
    assert isinstance(manifest["git_commit_hash"], str)
    assert manifest["git_commit_hash"]


def test_manifest_hash_is_deterministic():
    manifest = {
        "schema_version": "1",
        "model": {"identifier": "m", "provider": "p", "provider_options": {}},
        "sampling": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 16},
        "num_samples": 1,
        "evaluation": {"metrics": ["a"], "extractor": "x"},
        "seeds": {"sampling_seed": None},
        "package_versions": {"themis-eval": "1.0.0"},
        "git_commit_hash": "abc123",
    }
    reordered = {
        "git_commit_hash": "abc123",
        "package_versions": {"themis-eval": "1.0.0"},
        "seeds": {"sampling_seed": None},
        "evaluation": {"extractor": "x", "metrics": ["a"]},
        "num_samples": 1,
        "sampling": {"max_tokens": 16, "top_p": 1.0, "temperature": 0.0},
        "model": {"provider_options": {}, "provider": "p", "identifier": "m"},
        "schema_version": "1",
    }

    assert manifest_hash(manifest) == manifest_hash(reordered)


def test_manifest_validation_rejects_incomplete_payload():
    bad_manifest = {
        "schema_version": "1",
        "model": {"identifier": "m", "provider": "p", "provider_options": {}},
    }
    with pytest.raises(ValueError, match="Missing required manifest fields"):
        validate_reproducibility_manifest(bad_manifest)
