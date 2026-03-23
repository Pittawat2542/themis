"""Catalog preview render helpers."""

from __future__ import annotations

from themis import BenchmarkDefinition, BenchmarkDefinitionConfig
from themis.types.json_types import JSONDict

from ..datasets._prompts import _prompt_messages_from_context
from ._shared import _json_dict


def render_healthbench_preview(
    definition: BenchmarkDefinition,
    config: BenchmarkDefinitionConfig,
    sample: dict[str, object],
) -> list[JSONDict]:
    del definition, config
    return [
        _json_dict(
            {
                "prompt_variant_id": "healthbench-default",
                "messages": _prompt_messages_from_context(sample),
                "follow_up_turns": [],
            },
            label="healthbench preview",
        )
    ]


def render_context_prompt_preview(
    prompt_variant_id: str,
    sample: dict[str, object],
) -> list[JSONDict]:
    return [
        _json_dict(
            {
                "prompt_variant_id": prompt_variant_id,
                "messages": _prompt_messages_from_context(sample),
                "follow_up_turns": [],
            },
            label="context prompt preview",
        )
    ]
