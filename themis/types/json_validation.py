from __future__ import annotations

import json

from pydantic import TypeAdapter, ValidationError

from themis.errors.exceptions import SpecValidationError, StorageError
from themis.types.enums import ErrorCode
from themis.types.json_types import JSONDict, JSONValueType

_JSON_VALUE_ADAPTER: TypeAdapter[JSONValueType] = TypeAdapter(JSONValueType)
_JSON_DICT_ADAPTER: TypeAdapter[JSONDict] = TypeAdapter(JSONDict)


def format_validation_path(loc: tuple[object, ...]) -> str:
    parts: list[str] = []
    for item in loc:
        if isinstance(item, int):
            parts.append(f"[{item}]")
            continue
        text = str(item)
        if not parts:
            parts.append(text)
        else:
            parts.append(f".{text}")
    return "".join(parts)


def format_validation_error(exc: ValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return str(exc)
    first_error = errors[0]
    location = first_error.get("loc", ())
    path = format_validation_path(
        tuple(location) if isinstance(location, tuple | list) else ()
    )
    message = str(first_error.get("msg", str(exc)))
    return f"{path}: {message}" if path else message


def validate_json_value(value: object, *, label: str) -> JSONValueType:
    try:
        return _JSON_VALUE_ADAPTER.validate_python(value)
    except ValidationError as exc:
        raise SpecValidationError(
            code=ErrorCode.SCHEMA_MISMATCH,
            message=f"{label} must be JSON-serializable ({format_validation_error(exc)}).",
        ) from exc


def validate_json_dict(value: object, *, label: str) -> JSONDict:
    try:
        return _JSON_DICT_ADAPTER.validate_python(value)
    except ValidationError as exc:
        raise SpecValidationError(
            code=ErrorCode.SCHEMA_MISMATCH,
            message=f"{label} must be a JSON object ({format_validation_error(exc)}).",
        ) from exc


def dump_storage_json_bytes(value: object, *, label: str) -> bytes:
    try:
        validated = _JSON_VALUE_ADAPTER.validate_python(value)
    except ValidationError as exc:
        raise StorageError(
            code=ErrorCode.STORAGE_WRITE,
            message=f"{label} must be JSON-serializable ({format_validation_error(exc)}).",
        ) from exc
    return json.dumps(validated, sort_keys=True, separators=(",", ":")).encode("utf-8")
