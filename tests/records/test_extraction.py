from themis.records.extraction import ExtractionRecord
from themis.types.enums import RecordStatus, ErrorCode, ErrorWhere
from themis.records.error import ErrorRecord


def test_extraction_record_success():
    record = ExtractionRecord(
        spec_hash="extr1",
        extractor_id="json_regex",
        success=True,
        parsed_answer={"key": "value"},
    )
    assert record.status == RecordStatus.OK
    assert record.success is True
    assert record.parsed_answer == {"key": "value"}


def test_extraction_record_failure():
    err = ErrorRecord(
        code=ErrorCode.PARSE_ERROR,
        message="Could not find JSON block",
        retryable=False,
        where=ErrorWhere.EXTRACTOR,
    )
    record = ExtractionRecord(
        spec_hash="extr2",
        status=RecordStatus.ERROR,
        extractor_id="json_regex",
        success=False,
        error=err,
    )
    assert record.status == RecordStatus.ERROR
    assert record.success is False
    assert record.error.code == ErrorCode.PARSE_ERROR
