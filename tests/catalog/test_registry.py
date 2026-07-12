from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest

from themis.catalog import load as load_catalog_component
from themis.catalog.registry import (
    ComponentSpec,
    builtin_component_refs,
    component_specs,
    discover_plugins,
    list_component_ids,
    register_component,
)
from themis.core.components import component_ref_from_value
from themis.core.contexts import GenerationContext
from themis.core.models import Case, Candidate
from themis.core.protocols import Generator


class PluginMetric:
    pass


@dataclass(frozen=True)
class FakeEntryPoint:
    name: str
    loaded: object
    group: str = "themis.components"

    def load(self) -> object:
        return self.loaded


class FakeEntryPoints(list[FakeEntryPoint]):
    def select(self, *, group: str) -> list[FakeEntryPoint]:
        return [entry_point for entry_point in self if entry_point.group == group]


@pytest.mark.asyncio
async def test_manifest_registry_loads_builtin_components_with_stable_refs() -> None:
    component = cast(Generator, load_catalog_component("builtin/demo_generator"))
    component_ref = component_ref_from_value("builtin/demo_generator")

    result = await component.generate(
        Case(
            case_id="case-1", input={"question": "2+2"}, expected_output={"answer": "4"}
        ),
        GenerationContext(run_id="run-1", case_id="case-1", seed=7),
    )

    assert isinstance(result, Candidate)
    assert result.final_output == {"answer": "4"}
    assert component_ref == builtin_component_refs()["builtin/demo_generator"]
    assert "builtin/demo_generator" in list_component_ids(kind="generator")


def test_manifest_registry_rejects_unknown_components_with_suggestions() -> None:
    with pytest.raises(ValueError, match="builtin/demo_generator"):
        component_ref_from_value("builtin/demo_generatr")


def test_registry_discovers_entry_point_components(monkeypatch: pytest.MonkeyPatch) -> None:
    import themis.catalog.registry as registry

    plugin_spec = ComponentSpec(
        component_id="plugin/example_metric",
        kind="metric",
        target=f"{__name__}:PluginMetric",
        version="1.0",
        fingerprint="plugin-example-metric@1.0",
    )
    monkeypatch.setattr(
        registry.importlib_metadata,
        "entry_points",
        lambda: FakeEntryPoints([FakeEntryPoint("example_metric", plugin_spec)]),
    )

    discovered = discover_plugins()

    assert discovered == {"plugin/example_metric": plugin_spec}
    assert component_specs()["plugin/example_metric"] == plugin_spec
    assert "plugin/example_metric" in list_component_ids(kind="metric")
    assert isinstance(load_catalog_component("plugin/example_metric"), PluginMetric)


def test_registry_accepts_explicit_component_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import themis.catalog.registry as registry

    monkeypatch.setattr(registry, "_REGISTERED_COMPONENT_SPECS", {})
    monkeypatch.setattr(
        registry.importlib_metadata,
        "entry_points",
        lambda: FakeEntryPoints([]),
    )
    registered = ComponentSpec(
        component_id="local/example_generator",
        kind="generator",
        target="themis.catalog.components:DemoGenerator",
        version="1.0",
        fingerprint="local-example-generator@1.0",
    )

    register_component(registered)

    assert component_specs()["local/example_generator"] == registered
    with pytest.raises(ValueError, match="Duplicate component id"):
        register_component(registered)
