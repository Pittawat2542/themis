"""Manifest-backed registry for builtin components and benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import get_close_matches
from functools import lru_cache
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


@lru_cache(maxsize=1)
def component_specs() -> dict[str, ComponentSpec]:
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
    return [component_id for component_id in component_ids if component_specs()[component_id].kind == kind]


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
    from themis.core.components import BUILTIN_COMPONENT_REFS

    return BUILTIN_COMPONENT_REFS


def load(name: str) -> object:
    return load_component(name)


def _unknown_component_message(component_id: str) -> str:
    suggestions = get_close_matches(component_id, component_specs().keys(), n=3, cutoff=0.5)
    if suggestions:
        return f"Unknown builtin component: {component_id}. Did you mean: {', '.join(suggestions)}?"
    return f"Unknown builtin component: {component_id}"
