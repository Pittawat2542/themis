from collections.abc import Mapping
from typing import TypeAlias

from pydantic import JsonValue

JSONScalar: TypeAlias = JsonValue
JSONValueType: TypeAlias = JsonValue
ParsedValue: TypeAlias = JsonValue
JSONDict: TypeAlias = dict[str, JsonValue]
JSONList: TypeAlias = list[JsonValue]
JSONMapping: TypeAlias = Mapping[str, JsonValue]
