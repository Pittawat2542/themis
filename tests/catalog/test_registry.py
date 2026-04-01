from __future__ import annotations

import pytest

from themis.catalog import load as load_catalog_component
from themis.catalog.registry import builtin_component_refs, list_component_ids
from themis.core.components import component_ref_from_value
from themis.core.contexts import GenerateContext
from themis.core.models import Case, GenerationResult


@pytest.mark.asyncio
async def test_manifest_registry_loads_builtin_components_with_stable_refs() -> None:
    component = load_catalog_component("builtin/demo_generator")
    component_ref = component_ref_from_value("builtin/demo_generator")

    result = await component.generate(
        Case(
            case_id="case-1", input={"question": "2+2"}, expected_output={"answer": "4"}
        ),
        GenerateContext(run_id="run-1", case_id="case-1", seed=7),
    )

    assert isinstance(result, GenerationResult)
    assert result.final_output == {"answer": "4"}
    assert component_ref == builtin_component_refs()["builtin/demo_generator"]
    assert "builtin/demo_generator" in list_component_ids(kind="generator")


def test_manifest_registry_rejects_unknown_components_with_suggestions() -> None:
    with pytest.raises(ValueError, match="builtin/demo_generator"):
        component_ref_from_value("builtin/demo_generatr")
