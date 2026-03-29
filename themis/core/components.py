"""Component reference models and manifest-backed builtin registry."""

from __future__ import annotations

from typing import Any

from themis.catalog.registry import component_specs
from themis.core.base import HashableModel


class ComponentRef(HashableModel):
    component_id: str
    version: str
    fingerprint: str


BUILTIN_COMPONENT_REFS: dict[str, ComponentRef] = {
    component_id: ComponentRef(
        component_id=component_id,
        version=spec.version,
        fingerprint=spec.fingerprint,
    )
    for component_id, spec in component_specs().items()
}


def component_ref_from_value(value: Any) -> ComponentRef:
    if isinstance(value, ComponentRef):
        return value
    if isinstance(value, str):
        try:
            return BUILTIN_COMPONENT_REFS[value]
        except KeyError as exc:
            from themis.catalog.registry import _unknown_component_message

            raise ValueError(_unknown_component_message(value)) from exc

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
