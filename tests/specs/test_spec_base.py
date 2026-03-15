from themis.specs.base import SpecBase


class CustomSpec(SpecBase):
    """Mock spec for testing base functionality."""

    name: str
    value: int


def test_spec_base_hashing():
    spec = CustomSpec(name="test", value=42)

    # Check default schema_version
    assert spec.schema_version == "1.0"

    # Check compute_hash produces 64 char string
    hash_val = spec.compute_hash()
    assert len(hash_val) == 64

    # spec_hash property should be 12 chars
    assert len(spec.spec_hash) == 12
    assert spec.spec_hash == hash_val[:12]


def test_spec_base_does_not_expose_legacy_validate_semantic_shim():
    spec = CustomSpec(name="test", value=42)

    assert hasattr(spec, "_validate_semantic") is False
    assert hasattr(spec, "validate_semantic") is False
