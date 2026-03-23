"""Catalog definitions for HumanEval and HumanEval+."""

from __future__ import annotations

from dataclasses import replace

from themis import BenchmarkDefinition

from ...common import (
    build_humaneval_benchmark,
    register_humaneval,
    register_humaneval_plus,
    summarize_humaneval,
    summarize_humaneval_plus,
)
from .dataset import (
    DEFAULT_HUMANEVAL_PLUS_VERSION,
    BuiltinHumanEvalDatasetProvider,
    parse_humaneval_variants,
)


def _preview_rows(_definition: BenchmarkDefinition) -> list[dict[str, object]]:
    return [
        {
            "task_id": "HumanEval/0",
            "entry_point": "add",
            "prompt": (
                "def add(a: int, b: int) -> int:\n"
                '    """Return the sum of two integers."""\n'
            ),
            "canonical_solution": "    return a + b\n",
            "base_input": [[1, 2], [4, 5]],
            "plus_input": [[10, 20]],
            "atol": 0.0,
        }
    ]


DEFINITION = BenchmarkDefinition(
    benchmark_id="humaneval",
    family="catalog",
    primary_metric_id="humaneval_pass_rate",
    requires_judge=False,
    metadata={
        "dataset_id": "evalplus/HumanEvalPlus",
        "split": "test",
        "variant": "base",
        "mini": False,
        "noextreme": False,
        "version": DEFAULT_HUMANEVAL_PLUS_VERSION,
    },
    builder=build_humaneval_benchmark,
    registrar=register_humaneval,
    summarizer=summarize_humaneval,
    dataset_provider_factory=lambda definition, huggingface_loader=None: (
        BuiltinHumanEvalDatasetProvider(
            mini=bool(definition.metadata.get("mini", False)),
            noextreme=bool(definition.metadata.get("noextreme", False)),
            version=str(
                definition.metadata.get("version", DEFAULT_HUMANEVAL_PLUS_VERSION)
            ),
            score_variant=str(definition.metadata.get("variant", "base")),
            huggingface_loader=huggingface_loader,
        )
    ),
    preview_rows_loader=_preview_rows,
)


PLUS_DEFINITION = BenchmarkDefinition(
    benchmark_id="humaneval_plus",
    family="catalog",
    primary_metric_id="humaneval_plus_pass_rate",
    requires_judge=False,
    metadata={
        "dataset_id": "evalplus/HumanEvalPlus",
        "split": "test",
        "variant": "plus",
        "mini": False,
        "noextreme": False,
        "version": DEFAULT_HUMANEVAL_PLUS_VERSION,
    },
    builder=build_humaneval_benchmark,
    registrar=register_humaneval_plus,
    summarizer=summarize_humaneval_plus,
    dataset_provider_factory=lambda definition, huggingface_loader=None: (
        BuiltinHumanEvalDatasetProvider(
            mini=bool(definition.metadata.get("mini", False)),
            noextreme=bool(definition.metadata.get("noextreme", False)),
            version=str(
                definition.metadata.get("version", DEFAULT_HUMANEVAL_PLUS_VERSION)
            ),
            score_variant=str(definition.metadata.get("variant", "plus")),
            huggingface_loader=huggingface_loader,
        )
    ),
    preview_rows_loader=_preview_rows,
)


def build_humaneval_definition(
    base_benchmark_id: str, raw_name: str
) -> BenchmarkDefinition:
    mini, noextreme, version = parse_humaneval_variants(base_benchmark_id, raw_name)
    template = DEFINITION if base_benchmark_id == "humaneval" else PLUS_DEFINITION
    tokens: list[str] = []
    if mini:
        tokens.append("mini")
    if noextreme:
        tokens.append("noextreme")
    if version != DEFAULT_HUMANEVAL_PLUS_VERSION or tokens:
        tokens.append(version)
    benchmark_id = base_benchmark_id
    if tokens:
        benchmark_id = f"{base_benchmark_id}:{','.join(tokens)}"
    return replace(
        template,
        benchmark_id=benchmark_id,
        metadata={
            **template.metadata,
            "mini": mini,
            "noextreme": noextreme,
            "version": version,
        },
    )
