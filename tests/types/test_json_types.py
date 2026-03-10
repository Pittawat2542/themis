from pydantic import JsonValue
from themis.types.json_types import JSONScalar, JSONValueType, ParsedValue


def test_json_scalar_type():
    assert JSONScalar is JsonValue


def test_json_value_type():
    assert JSONValueType is JsonValue


def test_parsed_value_type():
    assert ParsedValue is JsonValue
