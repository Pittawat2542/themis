from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel

SHORT_HASH_LENGTH = 12


class HashableMixin:
    """
    Reusable mixin for models that need canonical hashing.

    Shared by spec and error models so the same hashing rules apply anywhere
    identity or deduplication depends on serialized content.
    """

    def canonical_dict(self) -> dict[str, Any]:
        """
        Returns a dict of hash-included fields with sorted keys,
        stable list order, no NaN/Inf, datetimes normalized to UTC ISO8601.
        """
        assert isinstance(self, BaseModel)

        # We need a custom serializer for datetime and complex structures
        def _serialize_value(val: Any) -> Any:
            if isinstance(val, datetime):
                # Ensure UTC before ISO format
                if val.tzinfo is None:
                    # Treat naive datetimes as UTC (in practice they should be aware)
                    val = val.replace(tzinfo=timezone.utc)
                else:
                    val = val.astimezone(timezone.utc)
                # output format: YYYY-MM-DDTHH:MM:SS+00:00 (ISO8601)
                return val.isoformat()
            if isinstance(val, Enum):
                return val.value
            if isinstance(val, BaseModel):
                if hasattr(val, "canonical_dict"):
                    return val.canonical_dict()
                return {k: _serialize_value(v) for k, v in val.model_dump().items()}
            if isinstance(val, dict):
                return {k: _serialize_value(v) for k, v in val.items()}
            if isinstance(val, list):
                return [_serialize_value(v) for v in val]
            return val

        # Extract only fields that are not explicitly excluded from hashing.
        schema_props = self.__class__.model_fields
        included_data = {}
        for field_name, field_info in schema_props.items():
            # Respect both native Pydantic exclusion and the project-specific
            # `json_schema_extra={"exclude_from_hash": True}` convention.
            exclude_from_hash = False

            # 1. Pydantic native exclude
            if field_info.exclude:
                exclude_from_hash = True

            # 2. Project convention mapped into json_schema_extra
            if field_info.json_schema_extra and isinstance(
                field_info.json_schema_extra, dict
            ):
                if field_info.json_schema_extra.get("exclude_from_hash"):
                    exclude_from_hash = True

            # 3. Direct attribute (some wrappers might use it)
            if (
                hasattr(field_info, "exclude_from_hash")
                and field_info.exclude_from_hash
            ):
                exclude_from_hash = True

            if not exclude_from_hash:
                raw_val = getattr(self, field_name)
                included_data[field_name] = _serialize_value(raw_val)

        return included_data

    def compute_hash(self, *, short: bool = False) -> str:
        """
        SHA-256 of canonical JSON.
        When short=True, returns first 12 hex characters for human-friendly display.
        """
        canonical = self.canonical_dict()
        # `allow_nan=False` keeps hashes stable by rejecting NaN and Infinity.
        json_str = json.dumps(
            canonical, allow_nan=False, sort_keys=True, separators=(",", ":")
        )
        hash_hex = hashlib.sha256(json_str.encode("utf-8")).hexdigest()

        if short:
            return hash_hex[:SHORT_HASH_LENGTH]
        return hash_hex
