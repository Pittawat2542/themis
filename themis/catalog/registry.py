"""Unified registry for builtin and plugin-provided components."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from difflib import get_close_matches
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from themis.catalog.loaders import load_symbol, load_toml


@dataclass(frozen=True)
class ComponentSpec:
    component_id: str
    kind: str
    target: str
    version: str
    fingerprint: str


_MANIFEST_ROOT = Path(__file__).with_name("manifests")
_COMPONENT_MANIFEST = _MANIFEST_ROOT / "components.toml"
_COMPONENT_ENTRY_POINT_GROUP = "themis.components"
_REGISTERED_COMPONENT_SPECS: dict[str, ComponentSpec] = {}


def component_specs() -> dict[str, ComponentSpec]:
    """Return every known component from builtins, entry points, and registrations."""

    return _merge_specs(
        _manifest_component_specs(),
        discover_plugins(),
        _REGISTERED_COMPONENT_SPECS,
    )


def discover_plugins(
    *, group: str = _COMPONENT_ENTRY_POINT_GROUP
) -> dict[str, ComponentSpec]:
    """Discover installed Python entry-point components."""

    entry_points = importlib_metadata.entry_points()
    selected = entry_points.select(group=group)

    specs: dict[str, ComponentSpec] = {}
    for entry_point in selected:
        spec = _coerce_component_spec(entry_point.load())
        _add_unique_spec(specs, spec)
    return specs


def register_component(spec: ComponentSpec | Mapping[str, object]) -> ComponentSpec:
    """Register a component for the current process."""

    normalized = _coerce_component_spec(spec)
    if normalized.component_id in component_specs():
        raise ValueError(f"Duplicate component id: {normalized.component_id}")
    _REGISTERED_COMPONENT_SPECS[normalized.component_id] = normalized
    return normalized


def _manifest_component_specs() -> dict[str, ComponentSpec]:
    payload = load_toml(_COMPONENT_MANIFEST)
    specs: dict[str, ComponentSpec] = {}
    for component_id, entry in payload.get("components", {}).items():
        specs[component_id] = ComponentSpec(
            component_id=component_id,
            kind=str(entry["kind"]),
            target=str(entry["target"]),
            version=str(entry["version"]),
            fingerprint=str(entry["fingerprint"]),
        )
    return specs


def list_component_ids(*, kind: str | None = None) -> list[str]:
    component_ids = sorted(component_specs())
    if kind is None:
        return component_ids
    return [
        component_id
        for component_id in component_ids
        if component_specs()[component_id].kind == kind
    ]


def load_component(component_id: str, *, kind: str | None = None) -> object:
    spec = get_component_spec(component_id, kind=kind)
    loaded = load_symbol(spec.target)
    return loaded() if isinstance(loaded, type) else loaded


def get_component_spec(component_id: str, *, kind: str | None = None) -> ComponentSpec:
    try:
        spec = component_specs()[component_id]
    except KeyError as exc:
        raise ValueError(_unknown_component_message(component_id)) from exc
    if kind is not None and spec.kind != kind:
        raise ValueError(f"Component {component_id} is not a {kind}; found {spec.kind}")
    return spec


def builtin_component_refs() -> dict[str, Any]:
    from themis.core.components import ComponentRef

    return {
        component_id: ComponentRef(
            component_id=component_id,
            version=spec.version,
            fingerprint=spec.fingerprint,
        )
        for component_id, spec in component_specs().items()
    }


def load(name: str) -> object:
    return load_component(name)


def _unknown_component_message(component_id: str) -> str:
    suggestions = get_close_matches(
        component_id, component_specs().keys(), n=3, cutoff=0.5
    )
    if suggestions:
        return f"Unknown component: {component_id}. Did you mean: {', '.join(suggestions)}?"
    return f"Unknown component: {component_id}"


def _coerce_component_spec(value: ComponentSpec | Mapping[str, object]) -> ComponentSpec:
    if isinstance(value, ComponentSpec):
        return value
    if isinstance(value, Mapping):
        required_fields = {
            "component_id",
            "kind",
            "target",
            "version",
            "fingerprint",
        }
        missing = sorted(required_fields.difference(value))
        if missing:
            raise ValueError(
                "Component spec mapping is missing required fields: "
                + ", ".join(missing)
            )
        return ComponentSpec(
            component_id=str(value["component_id"]),
            kind=str(value["kind"]),
            target=str(value["target"]),
            version=str(value["version"]),
            fingerprint=str(value["fingerprint"]),
        )
    raise TypeError(
        "Component entry points must return ComponentSpec or a component spec mapping"
    )


def _merge_specs(*spec_groups: Mapping[str, ComponentSpec]) -> dict[str, ComponentSpec]:
    merged: dict[str, ComponentSpec] = {}
    for specs in spec_groups:
        for spec in specs.values():
            _add_unique_spec(merged, spec)
    return merged


def _add_unique_spec(specs: dict[str, ComponentSpec], spec: ComponentSpec) -> None:
    if spec.component_id in specs:
        raise ValueError(f"Duplicate component id: {spec.component_id}")
    specs[spec.component_id] = spec
