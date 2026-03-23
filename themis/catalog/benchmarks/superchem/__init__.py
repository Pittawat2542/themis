"""Catalog definition for SuperChem."""

from __future__ import annotations

from dataclasses import replace

from themis import BenchmarkDefinition, BenchmarkDefinitionConfig
from themis.benchmark.specs import BenchmarkSpec
from themis.types.json_types import JSONDict

from ...common.builders import build_mcq_benchmark
from ...common.previews import render_context_prompt_preview
from ...common.registration import register_mcq
from ...common.summaries import summarize_mcq
from .dataset import BuiltinSuperChemDatasetProvider

_SUPPORTED_SUPERCHEM_LANGUAGES = ("en", "zh")


def _preview_rows(_definition: BenchmarkDefinition) -> list[dict[str, object]]:
    return [
        {
            "uuid": "chem-1",
            "field": "chemistry",
            "question_type": "multiple_choice",
            "question_en": "What is shown?",
            "question_zh": "图中显示了什么？",
            "question_images": ["/tmp/chem-1.png"],
            "options_en": {"A": "Alpha", "B": "Beta"},
            "options_zh": {"A": "甲", "B": "乙"},
            "answer_en": ["B"],
            "answer_zh": ["B"],
        }
    ]


def _render_preview(
    definition: BenchmarkDefinition,
    config: BenchmarkDefinitionConfig,
    sample: dict[str, object],
) -> list[JSONDict]:
    del config
    return render_context_prompt_preview(
        prompt_variant_id=f"{definition.benchmark_id}-default",
        sample=sample,
    )


def _build_superchem_benchmark(
    definition: BenchmarkDefinition,
    config: BenchmarkDefinitionConfig,
) -> BenchmarkSpec:
    benchmark = build_mcq_benchmark(
        definition,
        config,
        expected_source_field="expected",
    )
    language = str(definition.metadata.get("language", "en"))
    return benchmark.model_copy(
        update={
            "slices": [
                slice_spec.model_copy(update={"dimensions": {"language": language}})
                for slice_spec in benchmark.slices
            ]
        }
    )


DEFINITION = BenchmarkDefinition(
    benchmark_id="superchem",
    family="catalog",
    primary_metric_id="choice_accuracy",
    requires_judge=False,
    metadata={
        "dataset_id": "ZehuaZhao/SUPERChem",
        "config_name": "default",
        "split": "train",
        "language": "en",
    },
    builder=_build_superchem_benchmark,
    registrar=register_mcq,
    summarizer=summarize_mcq,
    dataset_provider_factory=lambda _definition, huggingface_loader=None: (
        BuiltinSuperChemDatasetProvider(huggingface_loader=huggingface_loader)
    ),
    preview_renderer=_render_preview,
    preview_rows_loader=_preview_rows,
)


def supported_superchem_languages() -> tuple[str, ...]:
    return _SUPPORTED_SUPERCHEM_LANGUAGES


def build_superchem_definition(language: str) -> BenchmarkDefinition:
    return replace(
        DEFINITION,
        benchmark_id=f"superchem:{language}",
        metadata={
            **DEFINITION.metadata,
            "language": language,
        },
    )
