"""Helpers for converting exceptions into persisted ``ErrorRecord`` values."""

from __future__ import annotations

import traceback
from typing import Any

from themis.errors.exceptions import (
    ExtractionError,
    InferenceError,
    MetricError,
    StorageError,
    ThemisError,
)
from themis.records.error import ErrorRecord
from themis.types.enums import ErrorCode, ErrorWhere
from themis.types.json_types import JSONDict


_RETRYABLE_CODES = {
    ErrorCode.PROVIDER_RATE_LIMIT,
}


def map_exception_to_error_record(
    exc: BaseException,
    *,
    where: ErrorWhere | None = None,
    provider: str | None = None,
    model_id: str | None = None,
    candidate_id: str | None = None,
    attempt: int | None = None,
) -> ErrorRecord:
    """Maps an exception into a normalized error record for storage and reporting."""

    resolved_where = where or _infer_where(exc)
    code, message, details = _extract_error_fields(exc, resolved_where)
    merged_details = dict(details)
    merged_details["traceback"] = traceback.format_exception_only(type(exc), exc)[
        -1
    ].strip()
    if provider is not None:
        merged_details["provider"] = provider
    if model_id is not None:
        merged_details["model_id"] = model_id
    if candidate_id is not None:
        merged_details["candidate_id"] = candidate_id
    if attempt is not None:
        merged_details["attempt"] = attempt

    causes = []
    seen: set[int] = {id(exc)}
    current = exc.__cause__ or exc.__context__
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        causes.append(_map_cause(current, resolved_where))
        current = current.__cause__ or current.__context__

    return ErrorRecord(
        code=code,
        type=type(exc).__name__,
        message=message,
        retryable=_is_retryable(exc, code),
        where=resolved_where,
        details=merged_details,
        cause_chain=causes,
    )


def _map_cause(exc: BaseException, where: ErrorWhere) -> ErrorRecord:
    return ErrorRecord(
        code=_default_code_for_where(where),
        type=type(exc).__name__,
        message=f"{type(exc).__name__}: {exc}",
        retryable=False,
        where=where,
        details={},
        cause_chain=[],
    )


def _extract_error_fields(
    exc: BaseException, where: ErrorWhere
) -> tuple[ErrorCode, str, JSONDict]:
    if isinstance(exc, ThemisError):
        return exc.code, exc.message, _json_dict(exc.details)
    return _default_code_for_where(where), f"{type(exc).__name__}: {exc}", {}


def _default_code_for_where(where: ErrorWhere) -> ErrorCode:
    return {
        ErrorWhere.INFERENCE: ErrorCode.PROVIDER_UNAVAILABLE,
        ErrorWhere.EXTRACTOR: ErrorCode.PARSE_ERROR,
        ErrorWhere.METRIC: ErrorCode.METRIC_COMPUTATION,
        ErrorWhere.STORAGE: ErrorCode.STORAGE_WRITE,
        ErrorWhere.PLANNER: ErrorCode.SCHEMA_MISMATCH,
        ErrorWhere.EXECUTOR: ErrorCode.PROVIDER_UNAVAILABLE,
    }.get(where, ErrorCode.PROVIDER_UNAVAILABLE)


def _infer_where(exc: BaseException) -> ErrorWhere:
    if isinstance(exc, InferenceError):
        return ErrorWhere.INFERENCE
    if isinstance(exc, ExtractionError):
        return ErrorWhere.EXTRACTOR
    if isinstance(exc, MetricError):
        return ErrorWhere.METRIC
    if isinstance(exc, StorageError):
        return ErrorWhere.STORAGE
    return ErrorWhere.EXECUTOR


def _is_retryable(exc: BaseException, code: ErrorCode) -> bool:
    return code in _RETRYABLE_CODES


def _json_dict(details: dict[str, Any] | None) -> JSONDict:
    payload: JSONDict = {}
    if not details:
        return payload
    for key, value in details.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            payload[key] = value
        elif isinstance(value, list):
            payload[key] = value
        elif isinstance(value, dict):
            payload[key] = value
        else:
            payload[key] = str(value)
    return payload
