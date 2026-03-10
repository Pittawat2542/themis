from themis.types.enums import ErrorCode, ErrorWhere
from themis.errors.exceptions import (
    ThemisError,
    InferenceError,
    RetryableProviderError,
)
from themis.records.error import ErrorRecord


def test_themis_error_base():
    err = ThemisError(
        code=ErrorCode.PROVIDER_TIMEOUT,
        message="Request timed out",
        details={"model": "gpt-4"},
    )
    assert err.code == ErrorCode.PROVIDER_TIMEOUT
    assert err.message == "Request timed out"
    assert err.details == {"model": "gpt-4"}
    assert "Request timed out" in str(err)


def test_inference_error_hierarchy():
    err = RetryableProviderError(
        code=ErrorCode.PROVIDER_RATE_LIMIT, message="Rate limited"
    )
    assert isinstance(err, InferenceError)
    assert isinstance(err, ThemisError)
    assert err.code == ErrorCode.PROVIDER_RATE_LIMIT


def test_error_record_model():
    record = ErrorRecord(
        code=ErrorCode.PARSE_ERROR,
        type="JsonParseError",
        message="Failed to parse output",
        retryable=False,
        where=ErrorWhere.EXTRACTOR,
        details={"raw": "bad json"},
    )

    assert record.code == ErrorCode.PARSE_ERROR
    assert record.where == ErrorWhere.EXTRACTOR
    # Fingerprint should be automatically computed based on code, where, message
    # SHA-256 of {"code": "parse_error", "message": "Failed to parse output", "where": "extractor"}
    assert len(record.fingerprint) == 12


def test_error_record_fingerprint_stability():
    record1 = ErrorRecord(
        code=ErrorCode.PARSE_ERROR,
        message="Failed",
        retryable=False,
        where=ErrorWhere.EXTRACTOR,
    )

    record2 = ErrorRecord(
        code=ErrorCode.PARSE_ERROR,
        message="Failed",
        retryable=False,
        where=ErrorWhere.EXTRACTOR,
        details={"ignored_in_hash": True},
    )

    record3 = ErrorRecord(
        code=ErrorCode.PROVIDER_TIMEOUT,
        message="Failed",
        retryable=True,
        where=ErrorWhere.INFERENCE,
    )

    # Details and type are not part of the fingerprint, so hash should match
    assert record1.fingerprint == record2.fingerprint

    # Meaningful fields should change the fingerprint
    assert record1.fingerprint != record3.fingerprint


def test_error_record_fingerprint_uses_only_code_message_and_where():
    cause = ErrorRecord(
        code=ErrorCode.STORAGE_READ,
        message="disk unavailable",
        retryable=False,
        where=ErrorWhere.STORAGE,
    )
    baseline = ErrorRecord(
        code=ErrorCode.PROVIDER_TIMEOUT,
        message="timed out",
        retryable=False,
        where=ErrorWhere.INFERENCE,
    )
    enriched = ErrorRecord(
        code=ErrorCode.PROVIDER_TIMEOUT,
        type="ProviderTimeout",
        message="timed out",
        retryable=True,
        where=ErrorWhere.INFERENCE,
        details={"provider": "openai"},
        cause_chain=[cause],
    )

    assert enriched.compute_hash(short=True) == enriched.fingerprint
    assert enriched.fingerprint == baseline.fingerprint
    assert enriched.cause_chain == [cause]
