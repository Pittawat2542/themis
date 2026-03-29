"""Component reference models and builtin registry."""

from __future__ import annotations

from typing import Any

from themis.core.base import HashableModel


class ComponentRef(HashableModel):
    component_id: str
    version: str
    fingerprint: str


BUILTIN_COMPONENT_REFS: dict[str, ComponentRef] = {
    "generator/demo": ComponentRef(
        component_id="generator/demo",
        version="1.0",
        fingerprint="builtin-generator-demo-fingerprint",
    ),
    "reducer/demo": ComponentRef(
        component_id="reducer/demo",
        version="1.0",
        fingerprint="builtin-reducer-demo-fingerprint",
    ),
    "parser/demo": ComponentRef(
        component_id="parser/demo",
        version="1.0",
        fingerprint="builtin-parser-demo-fingerprint",
    ),
    "metric/demo": ComponentRef(
        component_id="metric/demo",
        version="1.0",
        fingerprint="builtin-metric-demo-fingerprint",
    ),
}


def component_ref_from_value(value: Any) -> ComponentRef:
    if isinstance(value, ComponentRef):
        return value
    if isinstance(value, str):
        try:
            return BUILTIN_COMPONENT_REFS[value]
        except KeyError as exc:
            raise ValueError(f"Unknown builtin component: {value}") from exc

    try:
        component_id = getattr(value, "component_id")
        version = getattr(value, "version")
        fingerprint = value.fingerprint()
    except AttributeError as exc:
        raise TypeError(
            "Components must be a known builtin component string or expose "
            "component_id, version, and fingerprint()."
        ) from exc
    return ComponentRef(
        component_id=component_id,
        version=version,
        fingerprint=fingerprint,
    )
