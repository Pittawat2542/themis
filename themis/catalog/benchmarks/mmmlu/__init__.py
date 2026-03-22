"""Catalog definition for MMMLU."""

from __future__ import annotations

from dataclasses import replace

from themis import BenchmarkDefinition

from ...common import build_mcq_benchmark, register_mcq, summarize_mcq
from ...datasets.common import (
    BuiltinDatasetProvider,
    CatalogNormalizedRows,
    _normalize_mmmlu_rows,
)

_SUPPORTED_MMMLU_CONFIG_NAMES = (
    "default",
    "AR_XY",
    "BN_BD",
    "DE_DE",
    "ES_LA",
    "FR_FR",
    "HI_IN",
    "ID_ID",
    "IT_IT",
    "JA_JP",
    "KO_KR",
    "PT_BR",
    "SW_KE",
    "YO_NG",
    "ZH_CN",
)


class BuiltinMMMLUDatasetProvider(BuiltinDatasetProvider):
    def normalize_loaded_rows(
        self,
        rows: list[dict[str, object]],
        dataset: object,
    ) -> CatalogNormalizedRows:
        return _normalize_mmmlu_rows(rows, dataset)


def _preview_rows(_definition: BenchmarkDefinition) -> list[dict[str, object]]:
    return [
        {
            "Unnamed: 0": 0,
            "Question": "What is 2 + 2?",
            "A": "1",
            "B": "4",
            "C": "3",
            "D": "5",
            "Answer": "B",
            "Subject": "math",
        }
    ]


DEFINITION = BenchmarkDefinition(
    benchmark_id="mmmlu",
    family="catalog",
    primary_metric_id="choice_accuracy",
    requires_judge=False,
    metadata={
        "dataset_id": "openai/MMMLU",
        "config_name": "default",
        "split": "test",
    },
    builder=lambda definition, config: build_mcq_benchmark(
        definition,
        config,
        expected_source_field="expected",
    ),
    registrar=register_mcq,
    summarizer=summarize_mcq,
    dataset_provider_factory=lambda _definition, huggingface_loader=None: (
        BuiltinMMMLUDatasetProvider(huggingface_loader=huggingface_loader)
    ),
    preview_rows_loader=_preview_rows,
)


def supported_mmmlu_config_names() -> tuple[str, ...]:
    return _SUPPORTED_MMMLU_CONFIG_NAMES


def build_mmmlu_definition(config_name: str) -> BenchmarkDefinition:
    return replace(
        DEFINITION,
        benchmark_id=f"mmmlu:{config_name}",
        metadata={
            **DEFINITION.metadata,
            "config_name": config_name,
        },
    )
