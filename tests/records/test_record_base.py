import pytest
from datetime import datetime, timezone
import themis
from themis.types.enums import RecordStatus
from themis.records.base import RecordBase
from themis.records.error import ErrorRecord
from themis.records.provenance import ProvenanceRecord
from themis.types.enums import ErrorCode, ErrorWhere


class MockRecord(RecordBase):
    data: str


def test_record_base_defaults():
    record = MockRecord(spec_hash="abc", data="test")
    assert record.spec_hash == "abc"
    assert record.status == RecordStatus.OK
    assert record.error is None
    assert record.provenance is None
    assert isinstance(record.created_at, datetime)
    assert record.created_at.tzinfo == timezone.utc


def test_record_base_with_error():
    err = ErrorRecord(
        code=ErrorCode.PROVIDER_TIMEOUT,
        message="timeout",
        retryable=True,
        where=ErrorWhere.INFERENCE,
    )
    record = MockRecord(
        spec_hash="abc", data="failed", status=RecordStatus.ERROR, error=err
    )
    assert record.status == RecordStatus.ERROR
    assert record.error is not None
    assert record.error.code == ErrorCode.PROVIDER_TIMEOUT


def test_record_immutability():
    record = MockRecord(spec_hash="abc", data="test")
    with pytest.raises(Exception):  # Pydantic ValidationError for frozen=True
        record.data = "new_data"


def test_record_base_accepts_hydrated_provenance() -> None:
    provenance = ProvenanceRecord(
        themis_version=themis.__version__,
        git_commit="abc123",
        python_version="3.12.0",
        platform="macOS",
        library_versions={"openai": "1.0.0"},
        model_endpoint_meta={"resolved_model": "gpt-4o-mini-2026-03-01"},
    )

    record = MockRecord(spec_hash="abc", data="test", provenance=provenance)

    assert record.provenance == provenance
