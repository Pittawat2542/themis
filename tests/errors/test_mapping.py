from __future__ import annotations

from themis.errors.exceptions import InferenceError
from themis.errors.mapping import map_exception_to_error_record
from themis.types.enums import ErrorCode, ErrorWhere


def test_map_exception_to_error_record_preserves_context_and_cause_chain() -> None:
    try:
        try:
            raise TimeoutError("socket timed out")
        except TimeoutError as cause:
            raise InferenceError(
                code=ErrorCode.PROVIDER_TIMEOUT,
                message="provider timed out",
                details={"endpoint": "responses"},
            ) from cause
    except InferenceError as exc:
        record = map_exception_to_error_record(
            exc,
            where=ErrorWhere.INFERENCE,
            provider="openai",
            model_id="gpt-4o-mini",
            candidate_id="cand-1",
            attempt=2,
        )

    assert record.code == ErrorCode.PROVIDER_TIMEOUT
    assert record.where == ErrorWhere.INFERENCE
    assert record.details["provider"] == "openai"
    assert record.details["model_id"] == "gpt-4o-mini"
    assert record.details["candidate_id"] == "cand-1"
    assert record.details["attempt"] == 2
    assert record.details["endpoint"] == "responses"
    assert len(record.cause_chain) == 1
    assert record.cause_chain[0].message == "TimeoutError: socket timed out"
