from datetime import datetime, timezone
from pydantic import BaseModel, Field
from themis.types.hashable import HashableMixin


class MockSpec(HashableMixin, BaseModel):
    name: str
    items: list[str]
    count: int
    created_at: datetime = Field(json_schema_extra={"exclude_from_hash": True})


def test_hashable_mixin_canonical_dict():
    # Canonical JSON requires sorted keys, stable list order, no NaN, UTC ISO8601
    dt = datetime(2026, 3, 8, 12, 0, 0, tzinfo=timezone.utc)
    spec = MockSpec(name="test", items=["c", "a", "b"], count=42, created_at=dt)

    canonical = spec.canonical_dict()

    # Excluded fields should not be present
    assert "created_at" not in canonical

    # Expected dictionary items
    assert canonical == {
        "count": 42,
        "items": ["c", "a", "b"],  # list order is stable (not sorted)
        "name": "test",
    }


def test_hashable_mixin_compute_hash():
    dt1 = datetime(2026, 3, 8, 12, 0, 0, tzinfo=timezone.utc)
    dt2 = datetime(2026, 3, 8, 13, 0, 0, tzinfo=timezone.utc)

    spec1 = MockSpec(name="test", items=["a"], count=1, created_at=dt1)
    spec2 = MockSpec(name="test", items=["a"], count=1, created_at=dt2)
    spec3 = MockSpec(name="test2", items=["a"], count=1, created_at=dt1)

    # Changing excluded fields shouldn't change hash
    assert spec1.compute_hash() == spec2.compute_hash()

    # Changing included fields should change hash
    assert spec1.compute_hash() != spec3.compute_hash()

    # Short hash
    assert len(spec1.compute_hash(short=True)) == 12


def test_hashable_mixin_nested_models():
    class NestedSpec(HashableMixin, BaseModel):
        value: str

    class ParentSpec(HashableMixin, BaseModel):
        nested: NestedSpec

    spec = ParentSpec(nested=NestedSpec(value="inner"))
    canonical = spec.canonical_dict()
    assert canonical == {"nested": {"value": "inner"}}
